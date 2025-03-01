/*
Keeping track of release build performance:

commit      comment                                 max_n_blobs     time
f794dbdf515 non-streaming, decompress from buffer   64              2.21s
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
    let mut has_something = [[false; 180]; 360];  
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
                } else if osm_type == "OSMData" {
                    // OSM wiki: A primitive block contains _all_ the information to decompress the entities it contains
                    // -> I assume this means any referred IDs in relations/ways will be contained in this block
                    let decoded_data = osm::PrimitiveBlock::decode(&decompressed as &[u8]).expect("couldn't read data block");
                    let string_table: Vec<String> = decoded_data.stringtable.s.iter().map(|str| std::str::from_utf8(&str).expect("invalid string").to_string()).collect();
                    let lat_offset = decoded_data.lat_offset();
                    let lon_offset = decoded_data.lon_offset();
                    let granularity = decoded_data.granularity();
                    let coord_decode = |coord, offset| -> f64 {
                        1e-9 * ((offset + (granularity as i64) * coord) as f64)
                    };
                    // Each primitive group is one of Nodes, DenseNodes, Ways, Relations  -- not multiple
                    for group in decoded_data.primitivegroup.iter() {
                        for node in &group.nodes {
                            for (k, v) in node.keys.iter().zip(node.vals.iter()) {
                                println!("{:?}: {:?}", k, v);
                                println!("{:?}: {:?}", string_table[*k as usize], string_table[*v as usize]);
                            }
                        }
                        for way in &group.ways {
                            for (k, v) in way.keys.iter().zip(way.vals.iter()) {
                                println!("{:?}: {:?}", k, v);
                                println!("{:?}: {:?}", string_table[*k as usize], string_table[*v as usize]);
                            }
                        }
                        for relation in &group.relations {
                            for (k, v) in relation.keys.iter().zip(relation.vals.iter()) {
                                println!("{:?}: {:?}", k, v);
                                println!("{:?}: {:?}", string_table[*k as usize], string_table[*v as usize]);
                            }
                        }
                        if let Some(dense_nodes) = &group.dense {
                            let mut kv_iter = dense_nodes.keys_vals.iter();
                            let has_kv = dense_nodes.keys_vals.len() > 0;
                            let mut last_lat: i64 = 0;
                            let mut last_lon: i64 = 0;
                            for (raw_lat, raw_lon) in dense_nodes.lat.iter().zip(dense_nodes.lon.iter()) {
                                // OSM wiki: index=0 is used as a delimiter when encoding DenseNodes
                                // Each node's tags are encoded in alternating <keyid> <valid>
                                // As an exception, if no node in the current block has any key/value pairs, this array does not contain any delimiters, but is simply empty
                                let mut kv = std::collections::HashMap::<&String, &String>::new();
                                if has_kv {
                                    while let Some(k) = kv_iter.next() {
                                        if *k == 0 {
                                            break;
                                        }
                                        let k = &string_table[*k as usize];
                                        let v = &string_table[*kv_iter.next().expect("kv iter not multiple of 2 length? key with no value?") as usize];
                                        kv.insert(k, v);
                                    }
                                }
                                // raw_lat = x_2 - last_lat
                                // raw_lat + last_lat = x_2
                                let lat_i = last_lat + *raw_lat;
                                let lon_i = last_lon + *raw_lon;
                                let lat = coord_decode(lat_i, lat_offset);
                                let lon = coord_decode(lon_i, lon_offset);
                                has_something[(lon as isize + 180) as usize][(lat as isize + 90) as usize] = true;
                                //println!("{} {} {:?}", lat, lon, kv);
                                last_lat = lat_i;
                                last_lon = lon_i;
                            }
                        }
                    }
                }
            }
        } else {
            println!("error reading blob");
        }
        //buffered_file.seek_relative(blob_header.datasize as i64);
        n_blobs_read += 1;
    }
    println!("{}", n_blobs_read);
    for lat_i in 0..180 {
        for lon_i in 0..360 {
            if has_something[lon_i][lat_i] {
                print!("x");
            } else {
                print!(" ");
            }
        }
        println!("");
    }
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