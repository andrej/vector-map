use std::f32::consts::E;
use std::io::{Read, Seek, BufRead};
use crate::osm_pbf;
use prost::Message;
use tracing::{Level, field, span, trace};
use crate::partially_compressed::PartiallyCompressedStream;
use crate::protobuf_helpers::{decode_field, WireType};
use std::collections::{HashMap, VecDeque};
use bytes::{Bytes, BytesMut};  // Bytes are a shared reference slice to a reference-counted buffer; essentially &[u8] that enforces lifetimes at runtime
// (Note we can't use static (compile-time) lifetimes for our use case, because the underlying buffer changes over time as we read data, so we have run-time "temporal" lifetimes, whereas Rust can only enforce lifetimes for a certain code scope, and the scopes needed to update the buffer and read the references would necessarily overlap.)

#[derive(Debug)]
pub struct OsmStream<R: Read + BufRead> {
    stream : PartiallyCompressedStream<R>,
    state: OsmStreamState,
}

#[derive(Debug)]
pub enum RawOsmEntity {
    Node(RawOsmNode),
    Way(RawOsmWay),
    Relation(RawOsmRelation),
}

#[derive(Debug)]
pub struct RawOsmNode {
    pub id: u64,
    pub lat: i64,
    pub lon: i64,
    pub kv: HashMap<Bytes, Bytes>
}

#[derive(Debug)]
pub struct RawOsmWay {}

#[derive(Debug)]
pub struct RawOsmRelation {}

#[derive(Debug, Default)]
enum OsmStreamState {
    #[default]
    Uninitialized,
    ReadingBlobHeader,
    ReadingBlobStart(osm_pbf::BlobHeader),
    ReadingBlob {
        state: ReadingBlobState,
        size: usize,
        skip_at_end: usize
    },
    End
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
        groups: VecDeque<Bytes>,  // subslices of heap
        string_table: Vec<Bytes>,  // subslices of heap
        info: PrimitiveGroupInfo,
    },
    End
}

#[derive(Debug)]
enum DecodingPrimitiveGroupState {
    Start,
    DecodingDenseNodes(Bytes),
    End,
}

#[derive(Debug)]
struct PrimitiveGroupInfo {
    granularity: i32,
    lat_offset: i64,
    lon_offset: i64,
    date_granularity: i32,
}

impl<'a, R: Read + BufRead> OsmStream<R> {
    pub fn new(stream: R) -> Self {
        Self {
            stream: PartiallyCompressedStream::from_uncompressed(stream),
            state: OsmStreamState::ReadingBlobHeader,
        }
    }
}

impl OsmStream<std::io::BufReader<std::fs::File>> {
    pub fn from_file(path : &std::path::Path) -> std::io::Result<Self> {
        // Note: It is critical that the BufReader is _inside_ the PotentiallyCompressedStream;
        // this ensures that buffering works correctly even when switching between compressed and uncompressed reads.
        // (Otherwise, the BufReader may buffer compressed data as uncompressed because it does not know at what point in the stream we will "switch on" the compression.)
        let file = std::io::BufReader::new(std::fs::File::open(path)?);
        Ok(Self::new(file))
    }
}

impl<R: Read + BufRead + Seek> OsmStream<R> {
    fn decode_until_entity(&mut self) -> Result<Option<RawOsmEntity>, std::io::Error> {
        // If we're not currently reading entities, get into the "reading entities" state
        while match self.state {
            OsmStreamState::End => false,
            _ => true,
        } {
            self.decode_next()?;
        }
        // Read the next entity, if any
        match self.state {
            OsmStreamState::End => self.decode_entity(),
            _ => Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "Not in ReadingEntities state")),
        }
    }

    fn decode_next(&mut self) -> Result<(), std::io::Error> {
        match &mut self.state {
            OsmStreamState::ReadingBlobHeader => { 
                self.state = Self::decode_blob_header(&mut self.stream)? 
            },
            OsmStreamState::ReadingBlobStart(header) => {
                self.state = Self::decode_blob_start(&mut self.stream, header)?
            },
            OsmStreamState::ReadingBlob { state: reading_blob_state , size, skip_at_end } => match reading_blob_state {
                ReadingBlobState::ReadingHeaderBlock => { 
                    *reading_blob_state = Self::decode_header_block(&mut self.stream, *size)? 
                },
                ReadingBlobState::ReadingBlobData(reading_blob_data_state) => match reading_blob_data_state {
                    ReadingBlobDataState::Start => { 
                        *reading_blob_data_state = Self::decode_blob_data_start(&mut self.stream, *size)?;
                    },
                    ReadingBlobDataState::DecodingPrimitiveGroup { state: decoding_primitive_group_state, heap, groups, string_table, info } => match decoding_primitive_group_state {
                        DecodingPrimitiveGroupState::Start => {
                            *decoding_primitive_group_state = Self::decode_primitive_group(groups)?;
                        },
                        DecodingPrimitiveGroupState::DecodingDenseNodes(slice) => {
                            *decoding_primitive_group_state = Self::decode_dense_nodes(slice)?;
                        },
                        DecodingPrimitiveGroupState::End => {
                            *reading_blob_data_state = ReadingBlobDataState::End;
                        }
                    },
                    ReadingBlobDataState::End => {
                        *reading_blob_state = ReadingBlobState::End;
                    }
                },
                ReadingBlobState::End => {
                    // Seek to next message and expect to read blob header next
                    self.stream.seek(std::io::SeekFrom::Current(*skip_at_end as i64))?;
                    self.state = OsmStreamState::ReadingBlobHeader;
                }
            },
            _ => { 
                return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Cannot decode next in current state"))
            }
        };
        Ok(())
    }

    fn decode_blob_header(stream: &mut PartiallyCompressedStream<R>) -> Result<OsmStreamState, std::io::Error> {
        let _span = span!(Level::TRACE, "decode_blob_header").entered();

        let mut size_buf = [0u8; 4];
        stream.read_exact(&mut size_buf)?;
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
    fn decode_blob_start(stream: &mut PartiallyCompressedStream<R>, header: &osm_pbf::BlobHeader) -> Result<OsmStreamState, std::io::Error> {
        let _span = span!(Level::TRACE, "decode_blob_start").entered();

        let mut data_field_number = None;
        let mut compressed_data_len: Option<u64> = None;
        let mut data_start_pos: Option<u64> = None;
        let mut uncompressed_data_len: Option<u64> = None;

        // Make sure the reader encounters EOF at the end of the Blob data (don't read into next blob)
        let mut limited_reader = stream.by_ref().take(header.datasize as u64);
        let mut raw_size_field_skip_len = 0;
        let mut seek_back = false;

        // There are at most two fields in the Blob message (raw_size and data)
        // We will iterate, then break, until we hit the data field (ignoring raw_size if needed)
        for i in 0..=1 {
            let mut field_number: u32 = 0;
            let mut wire_type: WireType = WireType::Varint(0);
            let n_read = match decode_field(&mut limited_reader, &mut field_number, &mut wire_type)? {
                Some(n) => n,
                None => break, // reached end of Blob message
            };
            match (field_number, wire_type) {
                (2, WireType::Varint(raw_size)) => {
                    uncompressed_data_len = Some(raw_size);
                    if i == 1 {
                        // The raw_size field follows _after_ the data; once done processing the data, we'll need to skip it to get to the next message
                        raw_size_field_skip_len = n_read;
                        seek_back = true;
                    }
                },
                (1..=7, WireType::LengthDelimited(len)) => {
                    data_field_number = Some(field_number);
                    compressed_data_len = Some(len);
                    data_start_pos = Some(limited_reader.stream_position()?);
                    if i == 0 {
                        // Raw_size field might still follow; skip over the data
                        limited_reader.seek(std::io::SeekFrom::Current(len as i64))?;
                    }
                },
                (num, typ) => {
                    return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, format!("Unexpected field in Blob: ({}, {:?})", num, typ)))
                },
            }
        }

        if data_start_pos.is_none() {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Missing data field in Blob"));
        }
        if seek_back {
            // The raw_size field follows the data; get back to it
            stream.seek(std::io::SeekFrom::Start(data_start_pos.unwrap()))?;
        }

        match data_field_number {
            None => return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Missing data_type field in Blob")),
            Some(1) => { 
                uncompressed_data_len = compressed_data_len;
            }
            Some(3) => {
                if uncompressed_data_len.is_none() {
                    return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Missing raw_size field in zlib-compressed Blob"));
                }
                stream.enable_zlib_compression(compressed_data_len.unwrap() as usize);
            },
            _ => panic!("Unsupported compression type"),
        };

        trace!("Decoded Blob start: data_field_number={:?}, compressed_data_len={:?}, uncompressed_data_len={:?}", data_field_number, compressed_data_len, uncompressed_data_len);

        // Update state
        let size = uncompressed_data_len.unwrap() as usize;
        let skip_at_end = raw_size_field_skip_len;
        if header.r#type == "OSMHeader" {
            Ok(OsmStreamState::ReadingBlob {
                state: ReadingBlobState::ReadingHeaderBlock,
                size,
                skip_at_end,
            })
        } else if header.r#type == "OSMData" {
            Ok(OsmStreamState::ReadingBlob {
                state: ReadingBlobState::ReadingBlobData(ReadingBlobDataState::Start),
                size,
                skip_at_end,
            })
        } else {
            Err(std::io::Error::new(std::io::ErrorKind::InvalidData, format!("Unknown Blob type: {}", header.r#type)))
        }
    }

    fn decode_header_block(stream: &mut PartiallyCompressedStream<R>, size: usize) -> Result<ReadingBlobState, std::io::Error> {
        let _span: span::EnteredSpan = span!(Level::TRACE, "decode_header_block").entered();

        let mut header_buf = vec![0u8; size];
        stream.read_exact(&mut header_buf)?;
        let header_block = osm_pbf::HeaderBlock::decode(&header_buf[..])?;
        trace!("Decoded HeaderBlock: {:?}", header_block);

        Ok(ReadingBlobState::End)
    }

    fn decode_blob_data_start(stream: &mut PartiallyCompressedStream<R>, size: usize) -> Result<ReadingBlobDataState, std::io::Error> {
        let _span = span!(Level::TRACE, "decode_blob_data_start").entered();
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
        let mut heap  = BytesMut::with_capacity(size);

        let mut group_indices: VecDeque<std::ops::Range<usize>> = VecDeque::new();
        let mut string_table_loc = None;
        let mut cur_heap_offset = 0;

        while let Some(n_read_this_iter) = decode_field(limited_stream, &mut field_number, &mut wire_type)? {
            n_read += n_read_this_iter;
            trace!("Blob data field: number={}, wire_type={:?}, n_read={:?}", field_number, wire_type, n_read);
            match (field_number, &wire_type) {
                (1..=2, WireType::LengthDelimited(len)) => {
                    let start = cur_heap_offset;
                    let end = start + *len as usize;
                    heap.resize(end, 0);
                    // Read raw bytes directly into the heap to avoid extra copies.
                    limited_stream.read_exact(&mut heap[start..end])?;
                    cur_heap_offset = end;
                    if field_number == 1 { // stringtable
                        string_table_loc = Some(start..end);
                    } else if field_number == 2 { // primitivegroup
                        group_indices.push_back(start..end);
                    }
                },
                (17, WireType::Varint(val)) =>  granularity = *val as i32,
                (19, WireType::Varint(val)) =>  lat_offset = *val as i64,
                (20, WireType::Varint(val)) =>  lon_offset = *val as i64,
                (18, WireType::Varint(val)) =>  date_granularity = *val as i32,
                (num, typ) => {
                    return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, format!("Unexpected field in Blob data: ({}, {:?})", num, typ)))
                },
            }
        }

        let heap = heap.freeze();

        let mut string_table: Vec<Bytes> = Vec::new();
        if let Some(range) = string_table_loc {
            let string_table_heap = heap.slice(range.start..range.end);
            for slice in string_table_heap.split(|&b| b == 0) {
                //let s = std::str::from_utf8(slice)
                //    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("Invalid UTF-8 in string table: {e}")))?;
                let substr_start_idx = range.start + (slice.as_ptr() as usize - string_table_heap.as_ptr() as usize);
                let substr_end_idx = substr_start_idx + slice.len();
                string_table.push(string_table_heap.slice(substr_start_idx..substr_end_idx));
            }
        }

        let mut groups: VecDeque<Bytes> = VecDeque::new();
        for range in group_indices {
            let group_slice = heap.slice(range.start..range.end);
            groups.push_back(group_slice);
        }

        Ok(ReadingBlobDataState::DecodingPrimitiveGroup {
            state: DecodingPrimitiveGroupState::Start,
            heap: heap,
            groups: groups,
            string_table: string_table,
            info: PrimitiveGroupInfo {
                granularity: 100,
                lat_offset: 0,
                lon_offset: 0,
                date_granularity: 1000,
            },
        })
    }

    fn decode_primitive_group(groups: &mut VecDeque<Bytes>) -> Result<DecodingPrimitiveGroupState, std::io::Error> {
        /*
            message PrimitiveGroup {
                repeated Node nodes = 1;
                optional DenseNodes dense = 2;
                repeated Way ways = 3;
                repeated Relation relations = 4;
                repeated ChangeSet changesets = 5;
            }
         */
        let _span = span!(Level::TRACE, "decode_primitive_group").entered();

        // Pop the next PrimitiveGroup to decode, if any;
        // note that this also updates the state (groups shrinks by one, or if empty, we move to next blob header state)
        let Some(mut slice) = groups.pop_front() else {
            // After processing all groups, continue with the next blob header.
            return Ok(DecodingPrimitiveGroupState::End);
        };

        let mut field_number: u32 = 0;
        let mut wire_type = WireType::Varint(0);
        let Some(n_read) = decode_field(&mut (&slice as &[u8]), &mut field_number, &mut wire_type)? else {
            return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "EOF while reading PrimitiveGroup field"));
        };
        let subslice = slice.slice(n_read..);

        match (field_number, &wire_type) {
            (2, WireType::LengthDelimited(_)) => { // Dense nodes
                Ok(DecodingPrimitiveGroupState::DecodingDenseNodes(subslice))
            },
            _ => {
                Err(std::io::Error::new(std::io::ErrorKind::InvalidData, format!("Unexpected field in PrimitiveGroup: ({}, {:?})", field_number, wire_type)))
            }
        }
    }

    fn decode_dense_nodes(slice: &Bytes) -> Result<DecodingPrimitiveGroupState, std::io::Error> {
        let _span = span!(Level::TRACE, "decode_dense_nodes").entered();

        let mut entities: Vec<RawOsmEntity> = Vec::new();

        // TODO: implement
        Ok(DecodingPrimitiveGroupState::Start)
    }

    fn decode_entity(&mut self) -> Result<Option<RawOsmEntity>, std::io::Error> {
        Err(std::io::Error::new(std::io::ErrorKind::Other, "Not implemented yet"))
    }
}

impl<R: Read + BufRead + Seek> Iterator for OsmStream<R> {
    type Item = RawOsmEntity;
    fn next(&mut self) -> Option<Self::Item> {
        self.decode_until_entity().unwrap()
    }
}
