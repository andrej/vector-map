use std::io::{Read, Seek};
use crate::osm_pbf;
use prost::Message;
use crate::potentially_compressed::{PotentiallyCompressedStream, CompressionType};
use bytes;
use crate::protobuf_helpers::{decode_field, WireType};

pub struct OsmStream<R: Read + Seek> {
    stream : PotentiallyCompressedStream<R>,
    state: OsmStreamState,
}

#[derive(Debug)]
pub enum OsmEntity {
    Node(osm_pbf::Node),
    Way(osm_pbf::Way),
    Relation(osm_pbf::Relation),
}

enum OsmStreamState {
    ReadingBlobHeader,
    ReadingBlob(osm_pbf::BlobHeader),
    ReadingBlobData(usize),
    ReadingHeaderBlock(usize),
    ReadingEntities,
    End
}


impl<R: Read + Seek> OsmStream<R> {
    pub fn new(stream: R) -> Self {
        Self {
            stream: PotentiallyCompressedStream::from_uncompressed(stream),
            state: OsmStreamState::ReadingBlobHeader,
        }
    }
}

impl OsmStream<std::fs::File> {
    pub fn from_file(path : &std::path::Path) -> std::io::Result<Self> {
        let file = std::fs::File::open(path)?;
        Ok(Self::new(file))
    }
}

impl<R: Read + Seek> Iterator for OsmStream<R> {
    type Item = OsmEntity;
    fn next(&mut self) -> Option<Self::Item> {
        self.decode_next().unwrap()
    }
}

impl<R: Read + Seek> OsmStream<R> {
    fn decode_next(&mut self) -> Result<Option<OsmEntity>, std::io::Error> {
        // If we're not currently reading entities, get into the "reading entities" state
        while match self.state {
            OsmStreamState::ReadingEntities => false,
            _ => true,
        } {
            match self.state {
                OsmStreamState::ReadingBlobHeader => { self.decode_blob_header()?; },
                OsmStreamState::ReadingBlob(_) => { self.decode_blob_start()?; },
                OsmStreamState::ReadingHeaderBlock(_) => { self.decode_header_block()?; }
                _ => { }
            }
        };
        // Read the next entity, if any
        match self.state {
            OsmStreamState::ReadingEntities => self.decode_entity(),
            _ => Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "Not in ReadingEntities state")),
        }
    }

    fn decode_blob_header(&mut self) -> Result<(), std::io::Error> {
        println!("Decoding BlobHeader");
        const SIZE_SIZE: usize = 4;
        self.stream.ensure_bytes(SIZE_SIZE)?;
        let header_size: usize = 
            bytes::Buf::get_u32(&mut self.stream.bytes)
            .try_into().map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "Header size too large"))?;

        println!("Header size: {}", header_size);
        self.stream.ensure_bytes(header_size)?;
        let blob_header = osm_pbf::BlobHeader::decode(bytes::Buf::take(&mut self.stream.bytes, header_size))?;
        println!("Header size: {}", header_size);
        println!("Decoded BlobHeader: {:?}", blob_header);

        self.state = OsmStreamState::ReadingBlob(blob_header);
        Ok(())
    }

    fn decode_entity(&mut self) -> Result<Option<OsmEntity>, std::io::Error> {
        Err(std::io::Error::new(std::io::ErrorKind::Other, "Not implemented yet"))
    }

    /// Read the first part of a Blob message (raw_size and compression type).
    /// We implement this manually because prost cannot decode partial messages, but we only want the start to be able to read sequentially from the compressed stream after this.
    pub fn decode_blob_start(&mut self) -> Result<(Option<u64>, Option<CompressionType>), std::io::Error> {
        let mut raw_size = None;
        let mut data_type = None;

        for _ in 0..2 {  // There are at most two fields in the Blob message
            self.stream.ensure_bytes(10)?; // Max 10 bytes for a varint
            let (field_number, wire_type) = decode_field(&mut self.stream)?;
            match (field_number, wire_type) {
                (2, WireType::Varint(val)) => raw_size = Some(val),
                (_, WireType::LengthDelimited(len)) => {
                    match field_number {
                        1 => data_type = Some(CompressionType::None),
                        3 => data_type = Some(CompressionType::Zlib),
                        4 => return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Unsupported compression type")),
                        5 => return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Unsupported compression type")),
                        6 => return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Unsupported compression type")),
                        7 => return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Unsupported compression type")),
                        _ => return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, format!("Unexpected field ({}) in Blob", field_number))),
                    }
                    //self.stream.seek(std::io::SeekFrom::Current(len as i64))?;
                }
                _ => return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Unexpected field in Blob")),
            }
        }

        let header = match self.state {
            OsmStreamState::ReadingBlob(ref header) => header,
            _ => panic!("Expected state to be ReadingBlob"),
        };

        let size = 
            match data_type {
                None => return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Missing data_type field in Blob")),
                Some(CompressionType::None) => {
                    header.datasize as usize
                }
                Some(CompressionType::Zlib) => {
                    self.stream.switch_compression(CompressionType::Zlib, raw_size.unwrap() as usize);
                    raw_size.unwrap() as usize
                },
                _ => panic!("Unsupported compression type"),
            };

        if header.r#type == "OSMHeader" {
            self.state = OsmStreamState::ReadingHeaderBlock(size);
        }

        println!("Decoded Blob start: raw_size={:?}, data_type={:?}", raw_size, data_type);

        Ok((raw_size, data_type))
    }

    pub fn decode_header_block(&mut self) -> Result<(), std::io::Error> {
        let size = match self.state {
            OsmStreamState::ReadingHeaderBlock(size) => size,
            _ => panic!("Expected state to be ReadingBlobData"),
        };
        self.stream.ensure_bytes(size)?;
        let header_block = osm_pbf::HeaderBlock::decode(bytes::Buf::take(&mut self.stream.bytes, size))?;
        println!("Decoded HeaderBlock: {:?}", header_block);

        // After reading the header block, we expect to read entities next
        self.state = OsmStreamState::ReadingEntities;
        Ok(())
    }

}
