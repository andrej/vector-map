use crate::osm_pbf;
use crate::partially_compressed::PartiallyCompressedStream;
use crate::protobuf_helpers::{decode_field, decode_varint, decode_zigzag_varint, WireType};
use bytes::{Buf, Bytes, BytesMut};
use prost::Message;
use std::collections::VecDeque;
use std::io::{BufRead, Read, Seek};
use tracing::{span, trace, Level}; // Bytes are a shared reference slice to a reference-counted buffer; essentially &[u8] that enforces lifetimes at runtime
                                   // (Note we can't use static (compile-time) lifetimes for our use case, because the underlying buffer changes over time as we read data, so we have run-time "temporal" lifetimes, whereas Rust can only enforce lifetimes for a certain code scope, and the scopes needed to update the buffer and read the references would necessarily overlap.)

// --------------------------------------------------------------------------
// Public Interface
// --------------------------------------------------------------------------

/// An efficient stream reader for OSM PBF files. The decoding is implemented in a streaming
/// fashion, reading the minimum number of bytes required to decode the next entity, leading to
/// very low memory usage irrespective of input file size. Since only small numbers of bytes are
/// read, for efficiency, the underlying stream should probably be buffered.
#[derive(Debug)]
pub struct OsmStream<R: Read + BufRead> {
    stream: PartiallyCompressedStream<R>,
    state: OsmStreamState,
}

#[derive(Debug)]
pub enum RawOsmEntity {
    Node(RawOsmNode),
    Way(RawOsmWay),
    Relation(RawOsmRelation),
}

#[derive(Debug, Clone, Default)]
pub struct RawOsmNode {
    pub id: u64,
    pub lat: i64,
    pub lon: i64,
    pub kv: Vec<(Bytes, Bytes)>,
}

impl RawOsmNode {
    /// Converts the encoded latitude and longitude to actual coordinates in degrees.
    /// Returns (latitude, longitude) as a tuple of f64 values.
    /// 
    /// Formulas:
    /// - latitude = .000000001 * (lat_offset + (granularity * lat))
    /// - longitude = .000000001 * (lon_offset + (granularity * lon))
    pub fn decode_coords(&self, lat_offset: i64, lon_offset: i64, granularity: i32) -> (f64, f64) {
        let lat = 0.000000001 * ((lat_offset + (granularity as i64 * self.lat)) as f64);
        let lon = 0.000000001 * ((lon_offset + (granularity as i64 * self.lon)) as f64);
        (lat, lon)
    }
}

/// An OSM node with decoded floating-point coordinates.
#[derive(Debug, Clone)]
pub struct OsmNode {
    pub id: u64,
    pub lat: f64,
    pub lon: f64,
    pub kv: Vec<(Bytes, Bytes)>,
}

#[derive(Debug)]
pub struct RawOsmWay {}

#[derive(Debug)]
pub struct RawOsmRelation {}

impl<'a, R: Read + BufRead> OsmStream<R> {
    pub fn new(stream: R) -> Self {
        Self {
            stream: PartiallyCompressedStream::from_uncompressed(stream),
            state: OsmStreamState::ReadingBlobHeader,
        }
    }
}

impl<R: Read + BufRead + Seek> OsmStream<R> {
    /// Create an iterator that yields BlobInfo for each blob in the file.
    /// Each BlobInfo contains position and size information that can be used to create
    /// independent "reduced" OsmStream readers via `BlobInfo::create_reader()`.
    /// This is useful for parallel processing of blobs.
    pub fn blobs<'a>(&'a mut self) -> BlobIterator<'a, R> {
        BlobIterator::new(self)
    }

    pub fn raw_nodes<'a>(&'a mut self) -> RawNodeIterator<'a, R> {
        RawNodeIterator::new(self)
    }

    pub fn nodes<'a>(&'a mut self) -> NodeIterator<'a, R> {
        NodeIterator::new(self)
    }
}

/// Information about a blob in the PBF file, including its position and size.
/// This can be used to create independent OsmStream readers for parallel processing.
#[derive(Debug, Clone)]
pub struct BlobInfo {
    pub blob_type: String,
    /// File position where this blob starts (at the BlobHeader size field)
    pub position: u64,
    /// Total size of this blob including the 4-byte header size, BlobHeader, and Blob data
    pub size: u64,
}

/// Iterator that yields BlobInfo for each blob in the OSM PBF file.
///
/// Each BlobInfo contains position and size information that can be used to create
/// independent "reduced" OsmStream readers via `BlobInfo::create_reader()`.
/// This enables parallel processing of blobs, where each blob can be processed
/// independently by different threads.
pub struct BlobIterator<'a, R: Read + BufRead + Seek> {
    stream: &'a mut OsmStream<R>,
}

impl<'a, R: Read + BufRead + Seek> BlobIterator<'a, R> {
    pub fn new(stream: &'a mut OsmStream<R>) -> Self {
        Self { stream }
    }
}

impl<'a, R: Read + BufRead + Seek> Iterator for BlobIterator<'a, R> {
    type Item = Result<BlobInfo, std::io::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        // Get the current position before reading the blob header
        let start_pos = match self.stream.stream.stream_position() {
            Ok(pos) => pos,
            Err(e) => return Some(Err(e)),
        };

        // Read through to get to the ReadingBlobStart state
        loop {
            match &self.stream.state {
                OsmStreamState::End => return None,
                OsmStreamState::Uninitialized => {
                    return Some(Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "Stream is uninitialized",
                    )));
                }
                OsmStreamState::ReadingBlobHeader => {
                    if let Err(e) = self.stream.decode_next() {
                        return Some(Err(e));
                    }
                }
                OsmStreamState::ReadingBlobStart(header) => {
                    // Calculate the total size of this blob
                    // 4 bytes (header size) + header size + datasize
                    let current_pos = match self.stream.stream.stream_position() {
                        Ok(pos) => pos,
                        Err(e) => return Some(Err(e)),
                    };
                    let header_size = (current_pos - start_pos - 4) as i32;
                    let total_size = 4 + header_size as u64 + header.datasize as u64;

                    // Clone the relevant info before we modify the state
                    let blob_info = BlobInfo {
                        blob_type: header.r#type.clone(),
                        position: start_pos,
                        size: total_size,
                    };

                    // Skip this blob in the main stream
                    if let Err(e) = self.stream.skip_next() {
                        return Some(Err(e));
                    }

                    return Some(Ok(blob_info));
                }
                _ => {
                    if let Err(e) = self.stream.skip_next() {
                        return Some(Err(e));
                    }
                }
            }
        }
    }
}

pub struct RawNodeIterator<'a, R: Read + BufRead + Seek> {
    stream: &'a mut OsmStream<R>,
}

impl<'a, R: Read + BufRead + Seek> RawNodeIterator<'a, R> {
    pub fn new(stream: &'a mut OsmStream<R>) -> Self {
        Self { stream }
    }
}

impl<'a, R: Read + BufRead + Seek> Iterator for RawNodeIterator<'a, R> {
    type Item = RawOsmNode;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            self.stream.decode_next().unwrap();
            match &self.stream.state {
                OsmStreamState::ReadingBlob {
                    state:
                        ReadingBlobState::ReadingBlobData(
                            ReadingBlobDataState::DecodingPrimitiveGroup {
                                state:
                                    DecodingPrimitiveGroupState::DecodingDenseNodes {
                                        state: DecodingDenseNodesState::Decoding { node, .. },
                                        ..
                                    },
                                ..
                            },
                        ),
                    ..
                } => {
                    return Some(node.clone());
                }
                OsmStreamState::End => {
                    return None;
                }
                _ => {
                    // Continue until state is one of the above two.
                    continue;
                }
            }
        }
    }
}

pub struct NodeIterator<'a, R: Read + BufRead + Seek> {
    stream: &'a mut OsmStream<R>,
}

impl<'a, R: Read + BufRead + Seek> NodeIterator<'a, R> {
    pub fn new(stream: &'a mut OsmStream<R>) -> Self {
        Self { stream }
    }
}

impl<'a, R: Read + BufRead + Seek> Iterator for NodeIterator<'a, R> {
    type Item = OsmNode;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            self.stream.decode_next().unwrap();
            match &self.stream.state {
                OsmStreamState::ReadingBlob {
                    state:
                        ReadingBlobState::ReadingBlobData(
                            ReadingBlobDataState::DecodingPrimitiveGroup {
                                state:
                                    DecodingPrimitiveGroupState::DecodingDenseNodes {
                                        state: DecodingDenseNodesState::Decoding { node, .. },
                                        ..
                                    },
                                info,
                                ..
                            },
                        ),
                    ..
                } => {
                    let (lat, lon) = node.decode_coords(info.lat_offset, info.lon_offset, info.granularity);
                    return Some(OsmNode {
                        id: node.id,
                        lat,
                        lon,
                        kv: node.kv.clone(),
                    });
                }
                OsmStreamState::End => {
                    return None;
                }
                _ => {
                    // Continue until state is one of the above two.
                    continue;
                }
            }
        }
    }
}

// --------------------------------------------------------------------------
// Private Implementation
// --------------------------------------------------------------------------

/// As the OsmStream progresses through the input stream, it transitions through a series of states,
/// each of which can have its own sub-states. Throughout reading, for each state, we can choose
/// to either skip ahead in a state or decode the data according to the current state. For example,
/// to decode DenseNode entities, we decode until we reach the DenseNodes decoding state, then
/// extract the entities from that state. Whenever we encounter a non-DenseNodes state, we can skip
/// decoding that state.
///
/// ## Read vs. Decode
///
/// Any state or function named "read" will consume some bytes from the underlying stream.
/// Functions named "decode" will interpret bytes that have already been read and that are
/// contained within the current state.
///
/// ## State Transition Diagram
///
/// ```text
/// OsmStreamState::ReadingBlobHeader ←────────────────────────────────────────────────────────┐
///    │      ↓                                                                                │
///    │   OsmStreamState::ReadingBlobStart                                                    │
///    │      ↓                                                                                │
///    │   OsmStreamState::ReadingBlob                                                         │
///    │      │                                                                                │
///    │      ├─→ ReadingBlobState::ReadingHeaderBlock ────────────────────────────────────┐   │
///    │      │                                                                            │   │
///    │      └─→ ReadingBlobState::ReadingBlobData                                        │   │
///    │             │                                                                     │   │
///    │             └─→ ReadingBlobDataState::Start                                       │   │
///    │                    ↓ (always)                                                     │   │
///    │                 ReadingBlobDataState::DecodingPrimitiveGroup                      │   │
///    │                    │                                                              │   │
///    │                    └─→ DecodingPrimitiveGroupState::Start ←───────────────────┐   │   │
///    │                           │                                                   │   │   │
///    │                           ├─→ DecodingPrimitiveGroupState::DecodingDenseNodes │   │   │
///    │                           │      │                                            │   │   │
///    │                           │      └─→ DecodingDenseNodesState::Start           │   │   │
///    │                           │             ↓ (always)                            │   │   │
///    │                           │          DecodingDenseNodesState::Decoding  ←─┐   │   │   │
///    │                           │             │    │ (yields entities)          │   │   │   │
///    │                           │             │    └────────────────────────────┘   │   │   │
///    │                           │             ↓                                     │   │   │
///    │                           │          DecodingDenseNodesState::End ────────────┘   │   │
///    │                           │                                                       │   │
///    │                           ↓   (after all groups)                                  │   │
///    │                        DecodingPrimitiveGroupState::End                           │   │
///    │                           ↓                                                       │   │
///    │                 ReadingBlobDataState::End                                         │   │
///    │                    │                                                              │   │
///    │                    │   ┌──────────────────────────────────────────────────────────┘   │
///    │                    ↓   ↓                                                              │
///    │         ReadingBlobState::End ────────────────────────────────────────────────────────┘
///    │
///    ↓ (after all blobs)
/// OsmStreamState::End
/// ```
#[derive(Debug, Default)]
enum OsmStreamState {
    #[default]
    Uninitialized,
    ReadingBlobHeader,
    ReadingBlobStart(osm_pbf::BlobHeader),
    ReadingBlob {
        state: ReadingBlobState,
        /// Size in bytes of the underlying potentially compressed stream
        compressed_size: usize,
        /// Total number of bytes that can be read from self.stream in this state
        decompressed_size: usize,
        skip_at_end: usize,
    },
    End,
}

#[derive(Debug)]
enum ReadingBlobState {
    ReadingHeaderBlock,
    ReadingBlobData(ReadingBlobDataState),
    End,
}

#[derive(Debug)]
enum ReadingBlobDataState {
    Start,
    DecodingPrimitiveGroup {
        state: DecodingPrimitiveGroupState,
        heap: Bytes,
        remaining_groups: VecDeque<Bytes>, // remaining groups on the heap to decode (subslices of heap)
        remaining_slice: Bytes, // remaining subslice of the current group's bytes on the heap to decode (subslices of heap)
        string_table: Vec<Bytes>, // subslices of heap
        info: PrimitiveGroupInfo,
    },
    End,
}

#[derive(Debug)]
enum DecodingPrimitiveGroupState {
    Start,
    DecodingDenseNodes {
        slice: Bytes, // subslice of heap being decoded
        state: DecodingDenseNodesState,
    },
    End,
}

#[derive(Debug)]
struct PrimitiveGroupInfo {
    granularity: i32,
    lat_offset: i64,
    lon_offset: i64,
    date_granularity: i32,
}

#[derive(Debug)]
enum DecodingDenseNodesState {
    Start,
    Decoding {
        ids_slice: Bytes,
        lats_slice: Bytes,
        lons_slice: Bytes,
        keys_vals_slice: Bytes,
        node: RawOsmNode,
    },
    End,
}

impl<R: Read + BufRead + Seek> OsmStream<R> {
    fn decode_next(&mut self) -> Result<(), std::io::Error> {
        match &mut self.state {
            OsmStreamState::Uninitialized => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Stream is uninitialized",
                ));
            }
            OsmStreamState::ReadingBlobHeader => {
                self.state = Self::read_blob_header(&mut self.stream)?
            }
            OsmStreamState::ReadingBlobStart(header) => {
                self.state = Self::read_blob_start(&mut self.stream, header)?
            }
            OsmStreamState::ReadingBlob {
                state: reading_blob_state,
                compressed_size: _,
                decompressed_size,
                skip_at_end,
            } => match reading_blob_state {
                ReadingBlobState::ReadingHeaderBlock => {
                    *reading_blob_state =
                        Self::read_header_block(&mut self.stream, *decompressed_size)?
                }
                ReadingBlobState::ReadingBlobData(reading_blob_data_state) => {
                    match reading_blob_data_state {
                        ReadingBlobDataState::Start => {
                            *reading_blob_data_state =
                                Self::read_blob_data_start(&mut self.stream, *decompressed_size)?;
                        }
                        ReadingBlobDataState::DecodingPrimitiveGroup {
                            state: decoding_primitive_group_state,
                            heap: _,
                            remaining_groups,
                            remaining_slice,
                            string_table,
                            info: _,
                        } => match decoding_primitive_group_state {
                            DecodingPrimitiveGroupState::Start => {
                                *decoding_primitive_group_state =
                                    Self::decode_primitive_group_start(
                                        remaining_slice,
                                        remaining_groups,
                                    )?;
                            }
                            DecodingPrimitiveGroupState::DecodingDenseNodes {
                                slice,
                                state: decoding_dense_nodes_state,
                            } => match decoding_dense_nodes_state {
                                DecodingDenseNodesState::Start => {
                                    *decoding_dense_nodes_state =
                                        Self::decode_dense_nodes_start(slice)?
                                }
                                DecodingDenseNodesState::Decoding { .. } => {
                                    Self::decode_dense_nodes_single(
                                        decoding_dense_nodes_state,
                                        string_table,
                                    )?;
                                }
                                DecodingDenseNodesState::End => {
                                    *decoding_primitive_group_state =
                                        DecodingPrimitiveGroupState::Start;
                                }
                            },
                            DecodingPrimitiveGroupState::End => {
                                *reading_blob_data_state = ReadingBlobDataState::End;
                            }
                        },
                        ReadingBlobDataState::End => {
                            *reading_blob_state = ReadingBlobState::End;
                        }
                    }
                }
                ReadingBlobState::End => {
                    self.state = Self::end_reading_blob(&mut self.stream, *skip_at_end)?;
                }
            },
            OsmStreamState::End => {
                // Nothing more to decode
                return Ok(());
            }
        };
        Ok(())
    }

    fn skip_next(&mut self) -> Result<(), std::io::Error> {
        match &mut self.state {
            OsmStreamState::ReadingBlobHeader => {
                self.state = OsmStreamState::End;
                Ok(())
            }
            OsmStreamState::ReadingBlobStart(header) => {
                // Read blob header, but have not yet started reading any blob data -- skip entire blob
                self.stream
                    .seek(std::io::SeekFrom::Current(header.datasize as i64))?;
                self.state = OsmStreamState::ReadingBlobHeader;
                Ok(())
            }
            OsmStreamState::ReadingBlob {
                state: reading_blob_state,
                compressed_size,
                decompressed_size: _,
                skip_at_end,
                ..
            } => match reading_blob_state {
                ReadingBlobState::ReadingHeaderBlock
                | ReadingBlobState::ReadingBlobData(ReadingBlobDataState::Start) => {
                    // Started reading the raw_size field of the blob and located at the start of the data field contents;
                    // but have not read any data yet. Must skip the lenght of the compressed data in the original stream.
                    if self.stream.is_compressed() {
                        self.stream.disable_compression();
                    }
                    self.stream.seek(std::io::SeekFrom::Current(
                        *compressed_size as i64 + *skip_at_end as i64,
                    ))?;
                    *reading_blob_state = ReadingBlobState::End;
                    Ok(())
                }
                ReadingBlobState::ReadingBlobData(reading_blob_data_state) => {
                    match reading_blob_data_state {
                        ReadingBlobDataState::Start => {
                            unreachable!("Should have been handled above");
                        }
                        ReadingBlobDataState::DecodingPrimitiveGroup {
                            state: ref mut decoding_primitive_group_state,
                            heap,
                            ref mut remaining_groups,
                            remaining_slice: _,
                            string_table: _,
                            info: _,
                        } => match decoding_primitive_group_state {
                            DecodingPrimitiveGroupState::Start => {
                                // Pop a group from the groups queue to skip it
                                let Some(_) = remaining_groups.pop_front() else {
                                    *decoding_primitive_group_state =
                                        DecodingPrimitiveGroupState::End;
                                    return Ok(());
                                };
                                Ok(())
                            }
                            DecodingPrimitiveGroupState::DecodingDenseNodes { .. } => {
                                Err(std::io::Error::new(
                                    std::io::ErrorKind::InvalidData,
                                    "Skipping dense nodes not implemented",
                                ))
                            }
                            DecodingPrimitiveGroupState::End => {
                                *reading_blob_data_state = ReadingBlobDataState::End;
                                Ok(())
                            }
                        },
                        ReadingBlobDataState::End => {
                            *reading_blob_state = ReadingBlobState::End;
                            Ok(())
                        }
                    }
                }
                ReadingBlobState::End => {
                    self.state = Self::end_reading_blob(&mut self.stream, *skip_at_end)?;
                    Ok(())
                }
            },
            _ => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Cannot skip in current state",
            )),
        }
    }

    fn read_blob_header(
        stream: &mut PartiallyCompressedStream<R>,
    ) -> Result<OsmStreamState, std::io::Error> {
        let _span = span!(Level::TRACE, "read_blob_header").entered();

        let mut size_buf = [0u8; 4];
        let n_read = stream.read(&mut size_buf)?;
        if 0 == n_read {
            return Ok(OsmStreamState::End);
        }
        let header_size = u32::from_be_bytes(size_buf) as usize;
        trace!("Header size: {}", header_size);

        let mut header_buf = vec![0u8; header_size];
        stream.read_exact(&mut header_buf)?;
        let blob_header = osm_pbf::BlobHeader::decode(&header_buf[..])?;
        trace!("Decoded BlobHeader: {:?}", blob_header);

        // Update state
        Ok(OsmStreamState::ReadingBlobStart(blob_header))
    }

    /// Read the first part of a Blob message (raw_size and compression type).
    /// We implement this manually because prost cannot decode partial messages, but we only want the start to be able to read sequentially from the compressed stream after this.
    fn read_blob_start(
        stream: &mut PartiallyCompressedStream<R>,
        header: &osm_pbf::BlobHeader,
    ) -> Result<OsmStreamState, std::io::Error> {
        let _span = span!(Level::TRACE, "read_blob_start").entered();
        /*
           message Blob {
               optional int32 raw_size = 2; // When compressed, the uncompressed size

               oneof data {
                   bytes raw = 1; // No compression

                   // Possible compressed versions of the data.
                   bytes zlib_data = 3;

                   // For LZMA compressed data (optional)
                   bytes lzma_data = 4;

                   // Formerly used for bzip2 compressed data. Deprecated in 2010.
                   bytes OBSOLETE_bzip2_data = 5 [deprecated=true]; // Don't reuse this tag number.

                   // For LZ4 compressed data (optional)
                   bytes lz4_data = 6;

                   // For ZSTD compressed data (optional)
                   bytes zstd_data = 7;
               }
           }
        */

        let mut data_field_number = None;
        let mut compressed_data_len: Option<usize> = None;
        let mut data_start_pos: Option<u64> = None;
        let mut decompressed_data_len: Option<usize> = None;

        // Make sure the reader encounters EOF at the end of the Blob data (don't read into next blob)
        let mut limited_reader = stream.by_ref().take(header.datasize as u64);
        let mut n_read_total: usize = 0;
        let mut raw_size_field_skip_len = 0;
        let mut seek_back = false;

        // There are at most two fields in the Blob message (raw_size and data)
        // We will iterate, then break, until we hit the data field (ignoring raw_size if needed)
        for i in 0..=1 {
            let mut field_number: u32 = 0;
            let mut wire_type: WireType = WireType::Varint(0);
            let n_read = match decode_field(&mut limited_reader, &mut field_number, &mut wire_type)?
            {
                Some(n) => n,
                None => break, // reached end of Blob message
            };
            n_read_total += n_read;
            match (field_number, wire_type) {
                (2, WireType::Varint(raw_size)) => {
                    decompressed_data_len = Some(raw_size as usize);
                    if i == 1 {
                        // The raw_size field follows _after_ the data; once done processing the data, we'll need to skip it to get to the next message
                        raw_size_field_skip_len = n_read;
                        seek_back = true;
                    }
                }
                (1..=7, WireType::LengthDelimited(len)) => {
                    data_field_number = Some(field_number);
                    compressed_data_len = Some(len as usize);
                    if i == 0 {
                        // Raw_size field might still follow; skip over the data
                        data_start_pos = Some(limited_reader.stream_position()?);
                        limited_reader.seek(std::io::SeekFrom::Current(len as i64))?;
                    }
                }
                (num, typ) => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Unexpected field in Blob: ({}, {:?})", num, typ),
                    ))
                }
            }
        }

        let (Some(data_field_number), Some(compressed_data_len)) =
            (data_field_number, compressed_data_len)
        else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Missing data field in Blob",
            ));
        };
        if n_read_total + compressed_data_len != header.datasize as usize {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Corrupt file: encoded blob data size plus length field does not match datasize in BlobHeader"));
        }

        if seek_back {
            // The raw_size field follows the data; get back to it so the subsequent states can decode the data
            let data_start_pos =
                data_start_pos.expect("data_start_pos must be set if seek_back is true");
            stream.seek(std::io::SeekFrom::Start(data_start_pos))?;
        }

        // Enable compression and get decompressed data size
        let decompressed_data_len = match data_field_number {
            1 => {
                // If the data stream is uncompressed, the raw_size field is optional, so it might not have been set...
                if let Some(decompressed_data_len) = decompressed_data_len {
                    // ...if it is set anyways, the decompressed size must match the compressed size
                    if decompressed_data_len != compressed_data_len {
                        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Corrupt file: for an uncompressed Blob, decompressed size must match compressed size, but it does not"));
                    }
                }
                compressed_data_len
            }
            3 => {
                let Some(decompressed_data_len) = decompressed_data_len else {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "Corrupt file: Missing required raw_size field in compressed Blob",
                    ));
                };
                stream.enable_zlib_compression(compressed_data_len);
                decompressed_data_len
            }
            _ => panic!("Unsupported compression type"),
        };

        trace!("Decoded Blob start: data_field_number={:?}, compressed_data_len={:?}, decompressed_data_len={:?}", data_field_number, compressed_data_len, decompressed_data_len);

        // Update state
        let skip_at_end: usize = raw_size_field_skip_len;
        if header.r#type == "OSMHeader" {
            Ok(OsmStreamState::ReadingBlob {
                state: ReadingBlobState::ReadingHeaderBlock,
                compressed_size: compressed_data_len,
                decompressed_size: decompressed_data_len,
                skip_at_end,
            })
        } else if header.r#type == "OSMData" {
            Ok(OsmStreamState::ReadingBlob {
                state: ReadingBlobState::ReadingBlobData(ReadingBlobDataState::Start),
                compressed_size: compressed_data_len,
                decompressed_size: decompressed_data_len,
                skip_at_end,
            })
        } else {
            Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Unknown Blob type: {}", header.r#type),
            ))
        }
    }

    #[inline]
    fn end_reading_blob(
        stream: &mut PartiallyCompressedStream<R>,
        skip_at_end: usize,
    ) -> Result<OsmStreamState, std::io::Error> {
        let _span = span!(Level::TRACE, "end_reading_blob").entered();
        if stream.is_compressed() {
            stream.disable_compression();
        }
        if skip_at_end > 0 {
            stream.seek(std::io::SeekFrom::Current(skip_at_end as i64))?;
        }
        Ok(OsmStreamState::ReadingBlobHeader)
    }

    fn read_header_block(
        stream: &mut PartiallyCompressedStream<R>,
        size: usize,
    ) -> Result<ReadingBlobState, std::io::Error> {
        let _span: span::EnteredSpan = span!(Level::TRACE, "read_header_block").entered();

        let mut header_buf = vec![0u8; size];
        stream.read_exact(&mut header_buf)?;
        let header_block = osm_pbf::HeaderBlock::decode(&header_buf[..])?;
        trace!("Decoded HeaderBlock: {:?}", header_block);

        Ok(ReadingBlobState::End)
    }

    /// Reads one complete PrimitiveBlock from the stream.
    /// Since the fields in a PrimitiveBlock can be in any order, and we require the stringtable and other metadata (granularities, offsets, ...)
    /// to make sense of the data in the PrimitiveGroups, and since we cannot seek in the compressed stream, we need to read the entire PrimitiveBlock into memory first.
    /// This is the only part of the OsmStream that "buffers" data in memory, and is the principal contributor to memory usage of the OsmStream.
    fn read_blob_data_start(
        stream: &mut PartiallyCompressedStream<R>,
        size: usize,
    ) -> Result<ReadingBlobDataState, std::io::Error> {
        let _span = span!(Level::TRACE, "read_blob_data_start").entered();
        /*
            message PrimitiveBlock {
                required StringTable stringtable = 1;
                repeated PrimitiveGroup primitivegroup = 2;

                // Granularity, units of nanodegrees, used to store coordinates in this block.
                optional int32 granularity = 17 [default=100];

                // Offset value between the output coordinates and the granularity grid in units of nanodegrees.
                optional int64 lat_offset = 19 [default=0];
                optional int64 lon_offset = 20 [default=0];

                // Granularity of dates, normally represented in units of milliseconds since the 1970 epoch.
                optional int32 date_granularity = 18 [default=1000];
            }
        */

        // We want to first read the stringtable and the granularity and lat/lon offset fields (if any)
        // before starting to decode the PrimitiveGroup, as we'll need that info for decoding the entities.
        let mut granularity: i32 = 100;
        let mut lat_offset: i64 = 0;
        let mut lon_offset: i64 = 0;
        let mut date_granularity: i32 = 1000;

        let mut n_read: usize = 0;
        let mut field_number: u32 = 0;
        let mut wire_type: WireType = WireType::Varint(0);
        let limited_stream = &mut stream.by_ref().take(size as u64);

        // Heap to store decompressed bytes of all PrimitiveGroups as well as the stringtable
        // The size here is slightly overprovisioned (includes other fields), but that's probably better than to keep reallocating it for each primitivegroup we encounter
        // We want to first read the stringtable and the granularity and lat/lon offset fields (if any)
        // before starting to decode the PrimitiveGroup, as we'll need that info for decoding the entities.
        let mut heap = BytesMut::with_capacity(size);

        let mut group_indices: VecDeque<std::ops::Range<usize>> = VecDeque::new();
        let mut string_table_loc = None;
        let mut cur_heap_offset = 0;

        while let Some(n_read_this_iter) =
            decode_field(limited_stream, &mut field_number, &mut wire_type)?
        {
            n_read += n_read_this_iter;
            trace!(
                "Blob data field: number={}, wire_type={:?}, n_read={:?}",
                field_number,
                wire_type,
                n_read
            );
            match (field_number, &wire_type) {
                (1..=2, WireType::LengthDelimited(len)) => {
                    let start = cur_heap_offset;
                    let end = start + *len as usize;
                    heap.resize(end, 0);
                    // Read raw bytes directly into the heap to avoid extra copies.
                    limited_stream.read_exact(&mut heap[start..end])?;
                    cur_heap_offset = end;
                    if field_number == 1 {
                        // stringtable
                        string_table_loc = Some(start..end);
                    } else if field_number == 2 {
                        // primitivegroup
                        group_indices.push_back(start..end);
                    }
                }
                (17, WireType::Varint(val)) => granularity = *val as i32,
                (19, WireType::Varint(val)) => lat_offset = *val as i64,
                (20, WireType::Varint(val)) => lon_offset = *val as i64,
                (18, WireType::Varint(val)) => date_granularity = *val as i32,
                (num, typ) => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Unexpected field in Blob data: ({}, {:?})", num, typ),
                    ))
                }
            }
        }

        let heap = heap.freeze();

        let mut string_table: Vec<Bytes> = Vec::new();
        if let Some(range) = string_table_loc {
            let string_table_heap = heap.slice(range.start..range.end);
            let mut slice_reader = &string_table_heap[..];
            let mut field_number: u32 = 0;
            let mut wire_type = WireType::Varint(0);
            let mut pos = 0;
            while let Some(n_read) =
                decode_field(&mut slice_reader, &mut field_number, &mut wire_type)?
            {
                trace!(
                    "StringTable field: number={}, wire_type={:?}, n_read={:?}",
                    field_number,
                    wire_type,
                    n_read
                );
                if field_number != 1 {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Unexpected field number in StringTable: {}", field_number),
                    ));
                }
                match wire_type {
                    WireType::LengthDelimited(len) => {
                        let start = range.start + pos + n_read;
                        let end = start + len as usize;
                        string_table.push(heap.slice(start..end));
                        pos += n_read + len as usize;
                        // Advance slice_reader past the data we just consumed
                        slice_reader = &slice_reader[len as usize..];
                    }
                    _ => {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            "Expected length-delimited wire type for StringTable entry",
                        ));
                    }
                }
            }
        }

        let mut groups: VecDeque<Bytes> = VecDeque::new();
        for range in group_indices {
            let group_slice = heap.slice(range.start..range.end);
            groups.push_back(group_slice);
        }
        // Start with the first slice
        let remaining_slice = groups[0].clone();

        Ok(ReadingBlobDataState::DecodingPrimitiveGroup {
            state: DecodingPrimitiveGroupState::Start,
            heap: heap,
            remaining_groups: groups,
            remaining_slice: remaining_slice,
            string_table: string_table,
            info: PrimitiveGroupInfo {
                granularity: granularity,
                lat_offset: lat_offset,
                lon_offset: lon_offset,
                date_granularity: date_granularity,
            },
        })
    }

    fn decode_primitive_group_start(
        remaining_slice: &mut Bytes,
        remaining_groups: &mut VecDeque<Bytes>,
    ) -> Result<DecodingPrimitiveGroupState, std::io::Error> {
        let _span = span!(Level::TRACE, "decode_primitive_group").entered();
        /*
            message PrimitiveGroup {
                repeated Node nodes = 1;
                optional DenseNodes dense = 2;
                repeated Way ways = 3;
                repeated Relation relations = 4;
                repeated ChangeSet changesets = 5;
            }
        */

        // Done processing the current slice?
        if remaining_slice.is_empty() {
            // Pop the next PrimitiveGroup to decode, if any;
            // note that this also updates the state (groups shrinks by one, or if empty, we move to next blob header state)
            if let Some(next_slice) = remaining_groups.pop_front() {
                // Start with the next group.
                *remaining_slice = next_slice;
            } else {
                // No groups left; after processing all groups, continue with the next blob header.
                return Ok(DecodingPrimitiveGroupState::End);
            };
        }

        let mut field_number: u32 = 0;
        let mut wire_type = WireType::Varint(0);
        let Some(n_read) = decode_field(
            &mut (&remaining_slice as &[u8]),
            &mut field_number,
            &mut wire_type,
        )?
        else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "EOF while reading PrimitiveGroup field",
            ));
        };
        // Subslice to process in child states
        let subslice = remaining_slice.slice(n_read..);

        // Advance remaining slice
        remaining_slice.advance(
            n_read
                + match wire_type {
                    WireType::LengthDelimited(len) => len as usize,
                    _ => 0,
                },
        );

        match (field_number, &wire_type) {
            (2, WireType::LengthDelimited(_)) => {
                // Dense nodes
                Ok(DecodingPrimitiveGroupState::DecodingDenseNodes {
                    slice: subslice,
                    state: DecodingDenseNodesState::Start,
                })
            }
            (3, WireType::LengthDelimited(_)) => {
                // Ways
                // Not yet implemented, skip
                Ok(DecodingPrimitiveGroupState::Start)
            }
            (4, WireType::LengthDelimited(_)) => {
                // Relations
                // Not yet implemented, skip
                Ok(DecodingPrimitiveGroupState::Start)
            }
            (1, WireType::LengthDelimited(_)) => {
                // Nodes
                // Not yet implemented, skip
                Ok(DecodingPrimitiveGroupState::Start)
            }
            (5, WireType::LengthDelimited(_)) => {
                // Changesets
                // Not yet implemented, skip
                Ok(DecodingPrimitiveGroupState::Start)
            }
            _ => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Unexpected field in PrimitiveGroup: ({}, {:?})",
                    field_number, wire_type
                ),
            )),
        }
    }

    fn decode_dense_nodes_start(slice: &Bytes) -> Result<DecodingDenseNodesState, std::io::Error> {
        let _span = span!(Level::TRACE, "decode_dense_nodes_start").entered();

        let mut ids_slice = None;
        let mut lats_slice = None;
        let mut lons_slice = None;
        let mut keys_vals_slice = None;

        /*
            message DenseNodes {
                repeated sint64 id = 1 [packed = true]; // DELTA coded

                optional DenseInfo denseinfo = 5;

                repeated sint64 lat = 8 [packed = true]; // DELTA coded
                repeated sint64 lon = 9 [packed = true]; // DELTA coded

                // Special packing of keys and vals into one array. May be empty if all nodes in this block are tagless.
                repeated int32 keys_vals = 10 [packed = true];
            }
        */

        let mut pos: usize = 0;
        let mut field_number: u32 = 0;
        let mut wire_type = WireType::Varint(0);
        let slice_reader = &mut (&slice as &[u8]);
        while let Some(n_read) = decode_field(slice_reader, &mut field_number, &mut wire_type)? {
            pos += n_read;
            match (field_number, &wire_type) {
                (1, WireType::LengthDelimited(len)) => {
                    // ids
                    ids_slice = Some(slice.slice(pos..pos + (*len as usize)));
                }
                (8, WireType::LengthDelimited(len)) => {
                    // lats
                    lats_slice = Some(slice.slice(pos..pos + (*len as usize)));
                }
                (9, WireType::LengthDelimited(len)) => {
                    // lons
                    lons_slice = Some(slice.slice(pos..pos + (*len as usize)));
                }
                (10, WireType::LengthDelimited(len)) => {
                    // keys_vals
                    keys_vals_slice = Some(slice.slice(pos..pos + (*len as usize)));
                }
                _ => {}
            };
            pos += match &wire_type {
                WireType::LengthDelimited(len) => {
                    *slice_reader = &slice_reader[*len as usize..];
                    *len as usize
                }
                _ => 0,
            };
        }

        let (Some(ids_slice), Some(lats_slice), Some(lons_slice), Some(keys_vals_slice)) =
            (ids_slice, lats_slice, lons_slice, keys_vals_slice)
        else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Missing required field in DenseNodes",
            ));
        };

        Ok(DecodingDenseNodesState::Decoding {
            ids_slice,
            lats_slice,
            lons_slice,
            keys_vals_slice,
            node: RawOsmNode::default(),
        })
    }

    fn decode_dense_nodes_single(
        state: &mut DecodingDenseNodesState,
        string_table: &Vec<Bytes>,
    ) -> Result<(), std::io::Error> {
        let _span = span!(Level::TRACE, "decode_dense_nodes_single").entered();

        let DecodingDenseNodesState::Decoding {
            ids_slice,
            lats_slice,
            lons_slice,
            keys_vals_slice,
            node,
        } = state
        else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid state for decoding dense nodes single",
            ));
        };

        if ids_slice.is_empty() {
            *state = DecodingDenseNodesState::End;
            return Ok(());
        }

        // The following not only decodes the fields, but also advances the state slices and updates the last_* values
        node.kv.clear();
        node.id = Self::decode_delta_coded_field(ids_slice, node.id)?;
        node.lat = Self::decode_delta_coded_field(lats_slice, node.lat)?;
        node.lon = Self::decode_delta_coded_field(lons_slice, node.lon)?;

        // Key/value decoding
        enum KvState {
            Key,
            Value,
        };
        let mut kv_state = KvState::Key;
        loop {
            let mut key_or_val_idx: u64 = 0;
            let Some(n_read) = decode_varint(&mut &keys_vals_slice[..], &mut key_or_val_idx)?
            else {
                break;
            };
            keys_vals_slice.advance(n_read);
            if key_or_val_idx == 0 {
                break;
            }
            let Some(key_or_val) = string_table.get(key_or_val_idx as usize) else {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Key/Value index out of bounds: {}", key_or_val_idx),
                ));
            };
            kv_state = match kv_state {
                KvState::Key => {
                    node.kv.push((key_or_val.clone(), Bytes::new()));
                    KvState::Value
                }
                KvState::Value => {
                    node.kv.last_mut().unwrap().1 = key_or_val.clone();
                    KvState::Key
                }
            }
        }

        trace!("Decoded DenseNode: {:?}", node);

        Ok(())
    }

    fn decode_delta_coded_field<T>(slice: &mut Bytes, last_value: T) -> Result<T, std::io::Error>
    where
        T: std::ops::Add<Output = T> + TryFrom<i64> + Copy,
    {
        let mut delta: i64 = 0;
        let n_read = decode_zigzag_varint(&mut &slice[..], &mut delta)?;
        slice.advance(n_read);
        let decoded_value = last_value
            + T::try_from(delta).map_err(|_| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "Failed to convert delta")
            })?;
        Ok(decoded_value)
    }

    fn decode_entity(&mut self) -> Result<Option<RawOsmEntity>, std::io::Error> {
        Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Not implemented yet",
        ))
    }
}
