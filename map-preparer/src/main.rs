use map_preparer::osm;
use prost::Message;

const BUFFER_SIZE: usize = 4096;
fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).ok_or(std::io::Error::new(std::io::ErrorKind::InvalidInput, "missing input file path command line argument"));
    let mut file = std::fs::File::open(path?)?;
    let mut buffered_file = std::io::BufReader::new(file); //new(file);
    let mut buffer: [u8; BUFFER_SIZE] = [0; BUFFER_SIZE];
    let mut n_read = 0;
    let max_n = 4;
    /// has_something[lon][lat] == 1 iff. something in the planet.pbf file is located between [lon,lon+1), and [lat,lat+1)
    let mut has_something: [[bool; 180]; 360];  
    while let Ok((size, blob_header)) = read_blob_header(&mut buffered_file) {
        if n_read > max_n {
            break;
        }
        println!("Blob {:9}: Type: {}  Size: {}  Header size: {}", n_read, blob_header.r#type, blob_header.datasize, size);
        if let Ok((size, blob)) = read_blob(&mut buffered_file, &blob_header) {
            println!("Uncompressed size: {}  Data type: {:?}", blob.raw_size(), blob.data.unwrap());
            //match blob.data {
            //    Some(osm::blob::Data::Raw(_)) => {
            //        println!("raw");
            //    }
            //    Some(osm::blob::Data::ZlibData(_)) => {
            //        println!("zlib data");
            //    }
            //    _ => {()}
            //};
        } else {
            println!("error reading blob");
        }
        //buffered_file.seek_relative(blob_header.datasize as i64);
        n_read += 1;
    }
    println!("{}", n_read);
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
    let mut blob_header_buffer = vec![0; size]; //Vec::<u8>:: Vec::<u8>::with_capacity(size);
    if size != std::io::Read::read(in_stream, &mut blob_header_buffer).map_err(|e| { OSMPBFParseError::new(e) } )? {
        return Err(OSMPBFParseError { root_cause: None })
    };
    //let header = osm::HeaderBlock::decode(&buffer[..])?;
    let blob_header = osm::BlobHeader::decode(&blob_header_buffer as &[u8]).map_err(|e| { OSMPBFParseError::new(e) })?;
    Ok((size, blob_header))
}

/// Reads a blob
fn read_blob(in_stream: &mut impl std::io::Read, header: &osm::BlobHeader) -> Result<(usize, osm::Blob), OSMPBFParseError> {
    let compressed_size = header.datasize.try_into().map_err(OSMPBFParseError::new)?;
    let mut buffer = vec![0; compressed_size];
    let read_bytes = std::io::Read::read(in_stream, &mut buffer).map_err(OSMPBFParseError::new)?;
    if compressed_size != read_bytes {
        println!("read {} bytes but wanted {}", read_bytes, compressed_size);
        return Err(OSMPBFParseError { root_cause: None })
    }
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