use std::io::{Read, Seek, BufRead};
use crate::osm_pbf;
use prost::Message;
use tracing::{span, trace, Level};
use crate::partially_compressed::PartiallyCompressedStream;
use crate::protobuf_helpers::{decode_field, WireType};

#[derive(Debug)]
pub struct OsmStream<R: Read + BufRead> {
    stream : PartiallyCompressedStream<R>,
    state: OsmStreamState,
}

#[derive(Debug)]
pub enum OsmEntity {
    Node(osm_pbf::Node),
    Way(osm_pbf::Way),
    Relation(osm_pbf::Relation),
}

#[derive(Debug, Default)]
enum OsmStreamState {
    #[default]
    Uninitialized,
    ReadingBlobHeader,
    ReadingBlob(osm_pbf::BlobHeader),
    // Contains an absolute position to seek to after reading the blob data.
    ReadingHeaderBlock(usize, usize),
    ReadingBlobData(usize, usize),
    /// Heap buffer that contains all PrimitiveGroup messages concatenated, and a list of (start, end)
    /// ranges into that heap for each PrimitiveGroup. `next_index` is the next group to decode.
    DecodingPrimitiveGroup {
        heap: Vec<u8>,
        group_slices: Vec<(usize, usize)>,
        next_index: usize,
        stringtable: Option<osm_pbf::StringTable>,
        granularity: i32,
        lat_offset: i64,
        lon_offset: i64,
        date_granularity: i32,
    },
    ReadingEntities,
    End
}

#[derive(Debug)]
enum CompressionType {
    None,
    Zlib,
    Lzma,
    ObsoleteBzip2,
    Lz4,
    Zstd,
}

impl<R: Read + BufRead> OsmStream<R> {
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
    fn decode_until_entity(&mut self) -> Result<Option<OsmEntity>, std::io::Error> {
        // If we're not currently reading entities, get into the "reading entities" state
        while match self.state {
            OsmStreamState::ReadingEntities => false,
            _ => true,
        } {
            self.decode_next()?;
        }
        // Read the next entity, if any
        match self.state {
            OsmStreamState::ReadingEntities => self.decode_entity(),
            _ => Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "Not in ReadingEntities state")),
        }
    }

    fn decode_next(&mut self) -> Result<(), std::io::Error> {
        let entry_state = std::mem::take(&mut self.state);
        self.state = match entry_state {
            OsmStreamState::ReadingBlobHeader => { self.decode_blob_header()? },
            OsmStreamState::ReadingBlob(ref header) => { self.decode_blob_start(header)? },
            OsmStreamState::ReadingHeaderBlock(size, skip_at_end) => { self.decode_header_block(size, skip_at_end)? }
            OsmStreamState::ReadingBlobData(size, skip_at_end) => { self.decode_blob_data(size, skip_at_end)? },
            OsmStreamState::DecodingPrimitiveGroup { heap, group_slices, next_index, stringtable, granularity, lat_offset, lon_offset, date_granularity } => {
                self.decode_primitive_group(heap, group_slices, next_index, stringtable, granularity, lat_offset, lon_offset, date_granularity)?
            }
            _ => { return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Cannot decode next in current state"))}
        };
        Ok(())
    }

    fn decode_blob_header(&mut self) -> Result<OsmStreamState, std::io::Error> {
        let _span = span!(Level::TRACE, "decode_blob_header").entered();

        let mut size_buf = [0u8; 4];
        self.stream.read_exact(&mut size_buf)?;
        let header_size = u32::from_be_bytes(size_buf) as usize;
        trace!("Header size: {}", header_size);

        let mut header_buf = vec![0u8; header_size];
        self.stream.read_exact(&mut header_buf)?;
        let blob_header = osm_pbf::BlobHeader::decode(&header_buf[..])?;
        trace!("Decoded BlobHeader: {:?}", blob_header);

        Ok(OsmStreamState::ReadingBlob(blob_header))
    }

    /// Read the first part of a Blob message (raw_size and compression type).
    /// We implement this manually because prost cannot decode partial messages, but we only want the start to be able to read sequentially from the compressed stream after this.
    fn decode_blob_start(&mut self, header: &osm_pbf::BlobHeader) -> Result<OsmStreamState, std::io::Error> {
        let _span = span!(Level::TRACE, "decode_blob_start").entered();

        let mut data_field_number = None;
        let mut compressed_data_len: Option<u64> = None;
        let mut data_start_pos: Option<u64> = None;
        let mut uncompressed_data_len: Option<u64> = None;

        // Make sure the reader encounters EOF at the end of the Blob data (don't read into next blob)
        let mut limited_reader = self.stream.by_ref().take(header.datasize as u64);
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
            self.stream.seek(std::io::SeekFrom::Start(data_start_pos.unwrap()))?;
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
                self.stream.enable_zlib_compression(compressed_data_len.unwrap() as usize);
            },
            _ => panic!("Unsupported compression type"),
        };

        trace!("Decoded Blob start: data_field_number={:?}, compressed_data_len={:?}, uncompressed_data_len={:?}", data_field_number, compressed_data_len, uncompressed_data_len);

        if header.r#type == "OSMHeader" {
            Ok(OsmStreamState::ReadingHeaderBlock(uncompressed_data_len.unwrap() as usize, raw_size_field_skip_len))
        } else if header.r#type == "OSMData" {
            Ok(OsmStreamState::ReadingBlobData(uncompressed_data_len.unwrap() as usize, raw_size_field_skip_len))
        } else {
            Err(std::io::Error::new(std::io::ErrorKind::InvalidData, format!("Unknown Blob type: {}", header.r#type)))
        }
    }

    fn decode_header_block(&mut self, size: usize, skip_at_end: usize) -> Result<OsmStreamState, std::io::Error> {
        let _span = span!(Level::TRACE, "decode_header_block").entered();

        let mut header_buf = vec![0u8; size];
        self.stream.read_exact(&mut header_buf)?;
        let header_block = osm_pbf::HeaderBlock::decode(&header_buf[..])?;
        trace!("Decoded HeaderBlock: {:?}", header_block);

        // Seek to next message and expect to read blob header next
        self.stream.seek(std::io::SeekFrom::Current(skip_at_end as i64))?;
        Ok(OsmStreamState::ReadingBlobHeader)
    }

    fn decode_blob_data(&mut self, size: usize, skip_at_end: usize) -> Result<OsmStreamState, std::io::Error> {
        let _span = span!(Level::TRACE, "decode_blob_data").entered();
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
        let mut stringtable: Option<osm_pbf::StringTable> = None;
        let mut granularity: i32 = 100;
        let mut lat_offset: i64 = 0;
        let mut lon_offset: i64 = 0;
        let mut date_granularity: i32 = 1000;

        let mut n_read: usize = 0;
        let mut field_number: u32 = 0;
        let mut wire_type: WireType = WireType::Varint(0);
        let limited_stream = &mut self.stream.by_ref().take(size as u64);

        // Heap to store decompressed bytes of all PrimitiveGroups
        // The size here is overprovisioned (includes stringtable and other fields), but that's probably better than to keep reallocating it for each primitivegroup we encounter
        let mut heap: Vec<u8> = Vec::with_capacity(size);
        let mut group_slices: Vec<(usize, usize)> = Vec::new();
        let mut cur_group_start = 0;

        while let Some(n_read_this_iter) = decode_field(limited_stream, &mut field_number, &mut wire_type)? {
            n_read += n_read_this_iter;
            trace!("Blob data field: number={}, wire_type={:?}", field_number, wire_type);
            trace!("Total bytes read in Blob data so far: {}", n_read);
            match (field_number, &wire_type) {
                (1, WireType::LengthDelimited(len)) => { // stringtable
                    let mut stringtable_buf = vec![0u8; *len as usize];
                    limited_stream.read_exact(&mut stringtable_buf)?;
                    stringtable = Some(osm_pbf::StringTable::decode(&stringtable_buf[..])?);
                },
                (2, WireType::LengthDelimited(len)) => { // primitivegroup
                    // Read raw bytes directly into the heap to avoid extra copies.
                    let start = cur_group_start;
                    let end = start + *len as usize;
                    heap.resize(end, 0);
                    limited_stream.read_exact(&mut heap[start..end])?;
                    group_slices.push((start, end));
                    cur_group_start = end;
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

        if group_slices.is_empty() {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Expected at least one PrimitiveGroup in Blob data"));
        }

        // Advance stream past this blob (might have raw_size field after it to skip)
        self.stream.seek(std::io::SeekFrom::Current(skip_at_end as i64))?;
        Ok(OsmStreamState::DecodingPrimitiveGroup {
            heap,
            group_slices,
            next_index: 0,
            stringtable,
            granularity,
            lat_offset,
            lon_offset,
            date_granularity,
        })
    }

    fn decode_primitive_group(&mut self, heap: Vec<u8>, groups: Vec<(usize, usize)>, mut next_index: usize, _stringtable: Option<osm_pbf::StringTable>, _granularity: i32, _lat_offset: i64, _lon_offset: i64, _date_granularity: i32) -> Result<OsmStreamState, std::io::Error> {
        let _span = span!(Level::TRACE, "decode_primitive_group").entered();

        while next_index < groups.len() {
            let (start, end) = groups[next_index];
            let _pg = osm_pbf::PrimitiveGroup::decode(&heap[start..end])?;
            trace!("Decoded PrimitiveGroup {}/{}", next_index + 1, groups.len());
            next_index += 1;
        }

        // After processing all groups, continue with the next blob header.
        Ok(OsmStreamState::ReadingBlobHeader)
    }

    fn decode_entity(&mut self) -> Result<Option<OsmEntity>, std::io::Error> {
        Err(std::io::Error::new(std::io::ErrorKind::Other, "Not implemented yet"))
    }
}

impl<R: Read + BufRead + Seek> Iterator for OsmStream<R> {
    type Item = OsmEntity;
    fn next(&mut self) -> Option<Self::Item> {
        self.decode_until_entity().unwrap()
    }
}
