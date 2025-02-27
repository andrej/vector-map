/*
Keeping track of release build performance:

commit      comment                                 max_n_blobs     time
            non-streaming, decompress from buffer   64              2.21s
*/

use map_preparer::osm;
use prost::Message;

const BUFFER_SIZE: usize = 4096;
fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).ok_or(std::io::Error::new(std::io::ErrorKind::InvalidInput, "missing input file path command line argument"));
    let mut file = std::fs::File::open(path?)?;
    let mut buffered_file = std::io::BufReader::new(file); //new(file);
    let mut buffer: [u8; BUFFER_SIZE] = [0; BUFFER_SIZE];
    let mut n_blobs_read = 0;
    let max_n_blobs = 64;
    /// has_something[lon][lat] == 1 iff. something in the planet.pbf file is located between [lon,lon+1), and [lat,lat+1)
    let mut has_something: [[bool; 180]; 360];  
    while let Ok((size, blob_header)) = read_blob_header(&mut buffered_file) {
        if n_blobs_read >= max_n_blobs {
            break;
        }
        println!("Blob {:9}: Type: {}  Size: {}  Header size: {}", n_blobs_read, blob_header.r#type, blob_header.datasize, size);
        if let Ok((size, blob)) = read_blob(&mut buffered_file, &blob_header) {
            println!("Uncompressed size: {}  Data type: {:?}", blob.raw_size(), match blob.data {
                Some(osm::blob::Data::Raw(_)) => "raw",
                Some(osm::blob::Data::ZlibData(_)) => "zlib",
                Some(osm::blob::Data::Lz4Data(_)) => "lz4data",
                Some(osm::blob::Data::LzmaData(_)) => "lzmadata",
                Some(osm::blob::Data::ZstdData(_)) => "zstddata",
                Some(osm::blob::Data::ObsoleteBzip2Data(_)) => "ObsoleteBzip2Data",
                Some(_) => "unknown",
                None => ""
            });
            let raw_size = blob.raw_size() as usize;
            if let Some(osm::blob::Data::ZlibData(data)) = blob.data {
                let mut decompressor = flate2::bufread::ZlibDecoder::new(&data[..]);
                let mut decompressed = vec![0; raw_size]; 
                std::io::Read::read_exact(&mut decompressor, &mut decompressed).expect("couldn't decompress bytes");
                let osm_type: String = blob_header.r#type;
                if osm_type == "OSMHeader" {
                    let header = osm::HeaderBlock::decode(&decompressed as &[u8]).expect("couldn't read header block");
                    println!("Header: {:?}", header.bbox);
                } else if osm_type == "OSMData" {
                    let decoded_data = osm::PrimitiveBlock::decode(&decompressed as &[u8]).expect("couldn't read data block");
                    println!("Granularity: {:?}", decoded_data.granularity());
                }
            }
        } else {
            println!("error reading blob");
        }
        //buffered_file.seek_relative(blob_header.datasize as i64);
        n_blobs_read += 1;
    }
    println!("{}", n_blobs_read);
    Ok(())
}

/// Reads a blob header from the input stream and returns how many bytes total were read plus the blob header.
fn read_blob_header(in_stream: &mut impl std::io::Read) -> Result<(usize, osm::BlobHeader), OSMPBFParseError> {
    const SIZE_SIZE: usize = 4;
    let mut size_buffer: [u8; 4] = [0; 4];
    if SIZE_SIZE != std::io::Read::read(in_stream, &mut size_buffer).map_err(|e| { OSMPBFParseError::new(e) } )? {
        return Err(OSMPBFParseError { root_cause: None })
    }
    let size: usize = u32::from_be_bytes(size_buffer).try_into().map_err(|e| { OSMPBFParseError::new(e) })?;
    let mut blob_header_buffer = vec![0; size]; 
    read_all(in_stream, size, &mut blob_header_buffer).map_err(|_| { OSMPBFParseError { root_cause: None } })?;
    let blob_header = osm::BlobHeader::decode(&blob_header_buffer as &[u8]).map_err(|e| { OSMPBFParseError::new(e) })?;
    Ok((size, blob_header))
}

/// A single read system call may only read partially; this function keeps reading
fn read_all(in_stream: &mut impl std::io::Read, size: usize, buf: &mut [u8]) -> Result<(), ()> {
    // TODO: does Read::read_exact do exactly what this function does? If so, replace
    let mut unread_size: usize = size;
    loop {
        let read = 
            std::io::Read::read(in_stream, &mut buf[(size-unread_size)..size])
            .map_err(|_| { () } )?;
        if 0 == read || read > unread_size {
            return Err(())
        };
        unread_size -= read;
        if unread_size == 0 {
            break;
        }
    }
    Ok(())
}


/// Reads a blob
fn read_blob(in_stream: &mut impl std::io::Read, header: &osm::BlobHeader) -> Result<(usize, osm::Blob), OSMPBFParseError> {
    let compressed_size = header.datasize.try_into().map_err(OSMPBFParseError::new)?;
    let mut buffer = vec![0; compressed_size];
    read_all(in_stream, compressed_size, &mut buffer).map_err(|_| { OSMPBFParseError { root_cause: None } })?;
    let blob = osm::Blob::decode(&buffer as &[u8]).map_err(OSMPBFParseError::new)?;
    Ok((compressed_size, blob))
}

struct OSMPBFParseError {
    root_cause: Option<Box<dyn std::error::Error>>
}

impl OSMPBFParseError {
    fn new(root_cause: impl std::error::Error + 'static) -> Self { // the + 'static here constrains the lifetimes of any potential references contained within the error
        Self {
            root_cause: Some(Box::new(root_cause) as Box<dyn std::error::Error>)
        }
    }
}