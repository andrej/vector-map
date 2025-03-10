/*
Keeping track of release build performance:

commit      comment                                 max_n_blobs     time
f794dbdf515 non-streaming, decompress from buffer   64              2.21 s
3d9ad8b     decode dense nodes, map latlon          64              2.69 s
*/

/*
Plan:

 - streaming decoder
   - reads minimum number of bytes needed from input stream on demand
   - decompresses minimum number of bytes to read next item (i.e. read header
     first to determine how much to read, then only read length indicated in 
     header, etc.)
   - yields:
     - GroupStart bbox
     - Node
     - RelationStart
       - NodeRef
     - RelationEnd
     - WayStart
       - NodeRef
     - WayEnd
     - GroupEnd
     - in that sequence

 - hashmap of merged nodes:
   nodes that are within same granularity for current zoom level become merged
   to just one of their ids;
   even better, just use lat/lon tuple at selected zoom granularity as the ID;
   all properties become merged to that lat/lon at the zoom level
   -> basically binning of lat/lons for given zoom level (don't need higher 
      accuracy than what could be displayed in one pixel at given zoom level)
   -> at outermost zoom level, returned data should contain whole globe at a
      rough granularity
   -> at closer zoom levels, have multiple tiles; need to perform clamping

 - final target output:
   relations, ways arrays for each tile and zoom level we intend to support
   relations and ways should be separated out by some search criteria, e.g. 
   type=highway for efficient access

   way(type=highway) = {metadata, [(lat, lon), (lat, lon), (lat, lon) ...]} <- make lat lon i16s so 4 bytes for each tuple
   relation(type=highway) = {metadata, }

 - parallelization: parallel threads operate on different subsections of the
   input file / different start offsets. challenge: need to read some of the
   file to start at a sensible place (i.e. start at a primitive block);
   each thread produces its own outputs; finally, the outputs must be merged
   together; could potentially also parallelize the merging phase: parallel
   threads for each output to be produced
*/

/// See proto/fileformat.proto; we do not use the automatically generated Prost
/// bindings for this, because Prost only allows us to decode the whole message
/// at once, copying it in the process; we only want to read the minimum amount
/// to determine what the compression type and raw_size of a blob is, then let
/// the user read as much as they want (i.e. only one primitivegroup), instead
/// of decoding the whole thing at once. This also has a pipelining-like effect:
/// we don't have to block until the entire blob is decoded; instead, we can
/// start processing the first primitivegroup as soon as just enough bytes to
/// decompress it have been read
enum OSMBlobCompression {
        Raw,
        Zlib,
        Lzma,
        ObsoleteBzip2,
        Lz4,
        Zstd,
}

impl TryFrom<u32> for OSMBlobCompression {
    type Error = ();
    fn try_from(input: u32) -> Result<OSMBlobCompression, Self::Error> {
        use OSMBlobCompression::*;
        match input {
            1 => Ok(Raw),
            3 => Ok(Zlib),
            4 => Ok(Lzma),
            5 => Ok(ObsoleteBzip2),
            6 => Ok(Lz4),
            7 => Ok(Zstd),
            _ => Err(())
        }
    }
}

struct OSMBlob {
    header: osm::BlobHeader,
    raw_size: usize,
    compression: OSMBlobCompression
}

enum OSMStreamItem {
    Blob(OSMBlob),
    Group,
    Node,
    NodeRef,
    Relation,
    Way,
}

const STREAM_BUF_SIZE: usize = 8192; 
struct OSMStreamingReader<'a, InStream: std::io::Read + std::io::Seek> {
    in_stream: &'a mut InStream,
    // current is a stack of what we are currently parsing in the file along
    // with the anticipated size in the input stream of what we are currently
    // parsing; for example, if we are currently parsing a Relation, the stack
    // will consist of Blob->PrimitiveGroup->Group->Relation->NodeRef items,
    // each of which is a tuple (item, start_offset), where start_offset is the
    // byte offset from the input stream at which we started parsing this
    // object
    current: Vec<(usize, usize, OSMStreamItem)>,
    n_bytes_read: usize,
    /// Read only up to `max_len` bytes
    max_len: usize,
    bytes: bytes::BytesMut,
}

impl<'a, InStream: std::io::Read + std::io::Seek> OSMStreamingReader<'a, InStream> {

    pub fn new(in_stream: &'a mut InStream) -> Self {
        Self::new_with_max_len(in_stream, 0)
    }

    pub fn new_with_max_len(in_stream: &'a mut InStream, max_len: usize) -> Self {
        let mut this = Self {
            in_stream: in_stream,
            current: Vec::with_capacity(4),
            n_bytes_read: 0,
            max_len: max_len,
            bytes: bytes::BytesMut::with_capacity(STREAM_BUF_SIZE)
        };
        this.advance();
        this
    }

    pub fn advance(&mut self) {
        let stack_top = self.current.last();
        match stack_top {
            // 
            None => self.read_blob_head(),
            // 
            Some((_, _, OSMStreamItem::Blob(blob))) => {
                Ok(())
            }
            //
            Some(_) => {
                Ok(())
            }
        };
    }

    /// Read exactly `n` bytes into self.bytes, unless EOF occurs first, in
    /// which case we read upto EOF and return the number of bytes read (<n)
    fn _read_bytes_upto(&mut self, n: usize) -> Result<usize, OSMPBFParseError> {
        let mut buf: Vec<u8> = vec![0; n];
        let mut i = 0;
        let n_to_read = if self.max_len > 0 {
            usize::min(n, self.max_len - self.n_bytes_read)
        }  else {
            n
        };
        while let Ok(n_read_this_iter) = std::io::Read::read(self.in_stream, &mut buf[i..(n_to_read-i)]) {
            if n_read_this_iter == 0 {
                break;
            }
            i += n_read_this_iter;
        }
        self.n_bytes_read += i;
        bytes::BufMut::put(&mut self.bytes, &buf[0..i]);
        Ok(i)
    }

    /// Call this when you are done processing the bytes in self._bytes to re-
    /// use the memory. The buffer will be cleared for reuse, and any un-
    /// consumed bytes that have already been read (self.bytes.remaining()) will
    /// be copied to the start of the buffer.
    fn _reset_bytes(&mut self) {
        // let unconsumed = prost::bytes::Buf::copy_to_bytes(&mut self.bytes, prost::bytes::Buf::remaining(&self.bytes));
        // self.bytes.clear();
        // bytes::BufMut::put(&mut self.bytes, unconsumed);
        self.bytes = self.bytes.split_off(self.bytes.len() - prost::bytes::Buf::remaining(&self.bytes));
    }
    
    /// Reads a blob header from the input stream and store the blob header.
    /// Instead of decoding (and copying) the entire compressed contents, we then leave the buffer at that state to read
    /// sequentially
    fn read_blob_head(&mut self) -> Result<(), OSMPBFParseError> {
        let start_offset = self.n_bytes_read;
        let mut n_read: usize = 0;

        // Blob header
        const SIZE_SIZE: usize = 4;
        n_read += self._read_bytes_upto(SIZE_SIZE)?;
        let size: usize = bytes::Buf::get_u32(&mut self.bytes).try_into().map_err(OSMPBFParseError::new)?;
        n_read += self._read_bytes_upto(size)?;
        let blob_header = osm::BlobHeader::decode(&mut self.bytes).map_err(|e| { OSMPBFParseError::new(e) })?;

        // Blob itself
        // We decode only the first field (raw_size), followed by the metadata
        // of the second field, which will tell us which compression is used.
        // Since the field metadata can be up to 10 bytes (varint encoding) and
        // the first field is a 4-byte integer, we read upto 24 bytes.
        const SIZE_FIRST_TWO: usize = 24;
        n_read += self._read_bytes_upto(SIZE_FIRST_TWO)?;

        // raw_size field
        let (field_tag, wire_type) = prost::encoding::decode_key(&mut self.bytes).map_err(OSMPBFParseError::new)?;
        assert!(field_tag == 1);
        assert!(wire_type == prost::encoding::WireType::ThirtyTwoBit);
        let raw_size = i32::decode(&mut self.bytes).map_err(OSMPBFParseError::new)?;

        // oneof data
        let (field_tag, wire_type) = prost::encoding::decode_key(&mut self.bytes).map_err(OSMPBFParseError::new)?;
        let compression_type = field_tag.try_into().map_err(|_| OSMPBFParseError{root_cause: None})?;

        // total size of this blob, including header and size prefix
        let total_size = SIZE_SIZE + blob_header.datasize as usize;

        // Push onto the stack
        self.current.push(
            (  
                start_offset,
                total_size,
                OSMStreamItem::Blob(OSMBlob {
                    header: blob_header,
                    raw_size: raw_size.try_into().map_err(OSMPBFParseError::new)?,
                    compression: compression_type
                })
            )
        );

        Ok(())
    }

    /// Skip parsing the remainder of the current object on top of the stack
    fn skip(&mut self) -> bool {
        if let Some((top_start, top_size, _)) = self.current.last() {
            let remaining: i64 = (top_start + top_size) as i64 - self.n_bytes_read as i64;
            assert!(remaining >= 0);
            std::io::Seek::seek(&mut self.in_stream, std::io::SeekFrom::Current(remaining));
            return true
        }   
        false
    }

    fn read_group(&mut self) {

    }
    fn read_relation(&mut self) {

    }
    fn read_way(&mut self) {

    }

}

use map_preparer::osm;
use plotters::style::SizeDesc;
use prost::{bytes, Message};

const BUFFER_SIZE: usize = 4096;

/// lat, lon are in radians; returned values are x y where (0, 0) is in the top left corner
/// see: https://en.wikipedia.org/wiki/Web_Mercator_projection
fn web_mercator(lat: f64, lon: f64) -> (u64, u64) {
    /// lower right corner is at (2^zoom_level-1, 2^zoom_level-1)
    let zoom_level = 8;  //2^8 = 512
    use std::f64::consts::PI;

    (
        f64::floor(1.0/(2.0*PI) * f64::powi(2.0, zoom_level) * (PI + lon)) as u64,
        f64::floor(1.0/(2.0*PI) * f64::powi(2.0, zoom_level) * (PI - f64::ln(f64::tan(PI/4.0 + lat/2.0)))) as u64
    )
}
fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).ok_or(std::io::Error::new(std::io::ErrorKind::InvalidInput, "missing input file path command line argument"));
    let mut file = std::fs::File::open(path?)?;
    let mut buffered_file = std::io::BufReader::new(file); //new(file);

    let mut buffer: [u8; BUFFER_SIZE] = [0; BUFFER_SIZE];
    let mut n_blobs_read = 0;
    let mut n_bytes_read: usize = 0;
    let skip_n_blobs = 0;
    let max_n_blobs = 64;
    /// has_something[lon][lat] == 1 iff. something in the planet.pbf file is located between [lon,lon+1), and [lat,lat+1)
    let mut has_something = [[false; 181]; 361];  
    while let Ok((size, blob_header)) = read_blob_header(&mut buffered_file) {
        n_bytes_read += size;
        if n_blobs_read >= max_n_blobs {
            break;
        }
        println!("Blob {:9}: Type: {}  Size: {}  Header size: {}", n_blobs_read, blob_header.r#type, blob_header.datasize, size);
        let read_raw_blob = read_blob(&mut buffered_file, &blob_header);
        if n_blobs_read < skip_n_blobs {
            n_blobs_read += 1;
            continue;
        }
        if let Ok((size, blob)) = read_raw_blob {
            n_bytes_read += size;
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
    println!("{} blobs read ({} bytes)", n_blobs_read, n_bytes_read);
    plot(&has_something);
    Ok(())
}

fn plot(has_something: &[[bool; 181]; 361]) {
    use plotters::prelude::*;

    let root_area = BitMapBackend::new("plot.png", (512, 512))
    .into_drawing_area();
    root_area.fill(&WHITE).unwrap();
    let mut ctx = ChartBuilder::on(&root_area).build_cartesian_2d(0..512, 0..512).unwrap();
    ctx.configure_mesh().draw().unwrap();

    ctx.draw_series(
        has_something.iter().enumerate().filter_map(
            |(lon, ps)| Some(ps.iter().enumerate().filter_map(
                move |(lat, point)| {
                    if *point {
                        let (x, y) = web_mercator(f64::to_radians(lat as f64 - 90.0), f64::to_radians(lon as f64 - 180.0));
                        Some(Circle::new((x as i32, y as i32), 1, &RED))
                    } else {
                        None
                    }
                }
            ))
        ).flatten()
    ).unwrap(); 
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