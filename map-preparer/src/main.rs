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
     !! This won't be possible entirely due to the way the data format is
        structured. For example, a PrimitiveBlock has all PrimitiveGroups first
        in memory; only then do the granularity and lat_lon offset fields 
        follow. This is an issue since these latter fields are needed to 
        correctly compute lat/lons for the elements inside the PrimitiveGroups.
        Since everything is compressed, we also cannot cheaply seek further down
        in the file (wouldn't know the compressed offset)...
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

trait OSMStream : std::io::BufRead {} //+ std::io::Seek {}

enum InStream<RawT> {
    None,
    Raw(RawT),
    Zlib(flate2::bufread::ZlibDecoder<RawT>)
}

impl<RawT> InStream<RawT> {
    fn into_inner(self) -> Self {
        match self {
            InStream::None => InStream::None,
            InStream::Raw(x) => InStream::Raw(x),
            InStream::Zlib(x) => InStream::Raw(x.into_inner())
        }
    }
}

/// See proto/fileformat.proto; we do not use the automatically generated Prost
/// bindings for this, because Prost only allows us to decode the whole message
/// at once, copying it in the process; we only want to read the minimum amount
/// to determine what the compression type and raw_size of a blob is, then let
/// the user read as much as they want (i.e. only one primitivegroup), instead
/// of decoding the whole thing at once. This also has a pipelining-like effect:
/// we don't have to block until the entire blob is decoded; instead, we can
/// start processing the first primitivegroup as soon as just enough bytes to
/// decompress it have been read
#[derive(Clone, Copy)]
enum OSMBlobCompression {
        Raw,
        Zlib,
        Lzma,
        ObsoleteBzip2,
        Lz4,
        Zstd,
}
impl OSMBlobCompression {
    fn get_decompressor<'a, RawT>(&self, compressed_stream: RawT) -> InStream<RawT>
    where RawT: std::io::BufRead {
        match self {
            OSMBlobCompression::Raw => InStream::Raw(compressed_stream),
            OSMBlobCompression::Zlib => InStream::Zlib(flate2::bufread::ZlibDecoder::new(compressed_stream)),
            _ => panic!("unsupported compression type")
        }
    }
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

struct OSMBlock {
    block: osm::PrimitiveBlock,
    string_table: Vec<String>
}

enum OSMStreamItem {
    Blob(OSMBlob),
    Header(osm::HeaderBlock),
    Block(OSMBlock),
}


impl<RawT> InStream<RawT> 
where RawT: std::io::BufRead,
{
    fn dyn_readable_inner(&mut self) -> Box<&mut dyn std::io::Read> {
        match self {
            InStream::None => panic!("invalid state for InStream"),
            InStream::Raw(x) => Box::new(x),
            InStream::Zlib(x) => Box::new(x)
        }
    }
}

impl<RawT> InStream<RawT>
where RawT: std::io::BufRead + std::io::Seek
{
    fn dyn_seekable_inner(&mut self) -> Option<Box<&mut dyn std::io::Seek>> {
        match self{
            InStream::None => panic!("invalid state for InStream"),
            InStream::Raw(x) => Some(Box::new(x)),
            InStream::Zlib(_) => None // zlib streams are not seekable
        }
    }
}

const STREAM_BUF_SIZE: usize = 8192; 
struct OSMStreamingReader<RawT> 
where
{
    in_stream: InStream<RawT>,
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

impl<RawT> OSMStreamingReader<RawT> 
where RawT: std::io::BufRead + std::io::Seek
{

    pub fn new(in_stream: RawT) -> Self {
        Self::new_with_max_len(in_stream, 0)
    }

    pub fn new_with_max_len(in_stream: RawT, max_len: usize) -> Self {
        Self {
            in_stream: InStream::Raw(in_stream),
            current: Vec::with_capacity(4),
            n_bytes_read: 0,
            max_len: max_len,
            bytes: bytes::BytesMut::with_capacity(STREAM_BUF_SIZE),
        }
    }

    pub fn advance(&mut self) -> bool {
        let stack_top = self.current.last();
        match stack_top {
            // element at the top of the stack is completely parsed, pop it
            Some((start_offset, total_size, _)) if self.n_bytes_read >= start_offset + total_size => {
                self.current.pop();
                true
            }
            // start of stream/blob
            None => {
                self._deactivate_decompression();
                if self.read_blob_head().expect("parse error in blob head") {
                    if let Some((_, _, OSMStreamItem::Blob(OSMBlob { header: _, raw_size: _, compression: compression }))) = &self.current.last() {
                        self._activate_decompression(*compression);
                        true
                    } else {
                        panic!("after successful blob head parse, top of stack is not a blob?")
                    }
                } else {
                    false
                }
            },
            // 
            Some((_, _, OSMStreamItem::Blob(blob))) => {
                match blob.header.r#type.as_str() {
                    "OSMHeader" => { self.read_header_block().expect("parse error in header block") },
                    "OSMData" => { self.read_primitive_block().expect("parse error in primitive block") },
                    _ => false,
                }
            }
            // Unexpected state
            _ => panic!("unexpected parser state") 
        }
    }

    /// Attempt to make `n` bytes available for reading in self.bytes. If the
    /// internal cursor is not at the end of the available data by `n` or more,
    /// no data is read; otherwise, we read only the amount needed. The only
    /// case in which this function makes less than `n` bytes available for
    /// reading is if we encounter EOF beforehand.
    fn _read_bytes_upto(&mut self, mut n: usize) -> Result<usize, OSMPBFParseError> {
        let mut buf: Vec<u8> = vec![0; n];
        let mut i = 0;
        let n_remaining_in_buf = prost::bytes::Buf::remaining(&self.bytes);
        if n < n_remaining_in_buf {
            return Ok(n)
        } else {
            n -= n_remaining_in_buf;
            i += n_remaining_in_buf;
        }
        if self.max_len > 0 {
            n = usize::min( n, self.max_len - self.n_bytes_read)
        }
        while let Ok(n_read_this_iter) = std::io::Read::read(*self.in_stream.dyn_readable_inner(), &mut buf[i..n]) {
            if n_read_this_iter == 0 {
                break;
            }
            i += n_read_this_iter;
        }
        self.n_bytes_read += i;
        bytes::BufMut::put(&mut self.bytes, &buf[n_remaining_in_buf..i]);
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

    /// Swap the current in_stream with a decompressed one, based on the
    /// indicated compression type
    fn _activate_decompression(&mut self, compression_type: OSMBlobCompression) {
        let old_in_stream = std::mem::replace(&mut self.in_stream, InStream::None);
        if let InStream::Raw(raw_s) = old_in_stream {
            self.in_stream = compression_type.get_decompressor(raw_s);
        } else {
            panic!("Invalid stream state");
        }
    }

    /// Turn of decompression; call this after you are done reading from the 
    /// compressed block
    fn _deactivate_decompression(&mut self) {
        let old_in_stream = std::mem::replace(&mut self.in_stream, InStream::None);
        self.in_stream = old_in_stream.into_inner();
    }
    
    /// Reads a blob header from the input stream and store the blob header.
    /// Instead of decoding (and copying) the entire compressed contents, we then leave the buffer at that state to read
    /// sequentially
    fn read_blob_head(&mut self) -> Result<bool, OSMPBFParseError> {
        let start_offset = self.n_bytes_read;
        let mut n_read: usize = 0;

        // Blob header
        const SIZE_SIZE: usize = 4;
        n_read += self._read_bytes_upto(SIZE_SIZE)?;
        if n_read != SIZE_SIZE {
            // reached end of stream
            return Ok(false);
        }
        let header_size: usize = bytes::Buf::get_u32(&mut self.bytes).try_into().map_err(OSMPBFParseError::new)?;
        n_read += self._read_bytes_upto(header_size)?;
        let blob_header = osm::BlobHeader::decode(&mut self.bytes).map_err(|e| { OSMPBFParseError::new(e) })?;

        // Blob itself
        // We decode only the first field (raw_size), followed by the metadata
        // of the second field, which will tell us which compression is used.
        // Since the field metadata can be up to 10 bytes (varint encoding) and
        // the first field is also a varint up to 10 bytes in length, we read 
        // upto 20 bytes.
        const SIZE_FIRST_TWO: usize = 20;
        n_read += self._read_bytes_upto(SIZE_FIRST_TWO)?;

        // raw_size field
        let (field_tag, wire_type) = prost::encoding::decode_key(&mut self.bytes).map_err(OSMPBFParseError::new)?;
        assert!(field_tag == 2);
        assert!(wire_type == prost::encoding::WireType::Varint);
        let raw_size: i64 = prost::encoding::decode_varint(&mut self.bytes).map_err(OSMPBFParseError::new)?.try_into().map_err(OSMPBFParseError::new)?;

        // oneof data
        let (field_tag, wire_type) = prost::encoding::decode_key(&mut self.bytes).map_err(OSMPBFParseError::new)?;
        let compression_type: OSMBlobCompression = field_tag.try_into().map_err(|_| OSMPBFParseError{root_cause: None})?;

        // total size of this blob, including header and size prefix
        let total_size = SIZE_SIZE + header_size + blob_header.datasize as usize;

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
        Ok(true)
    }

    fn read_header_block(&mut self) -> Result<bool, OSMPBFParseError> {
        let start_offset = self.n_bytes_read;
        let mut to_read: usize = 0;
        if let Some((_, _, OSMStreamItem::Blob(blob))) = self.current.last() {
            to_read = blob.header.datasize.try_into().expect("blob datasize should be positive and fit into a usize");
        } else {
            panic!("read_primitive_block expects a blob on the parsing stack")
        }
        let n_read = self._read_bytes_upto(to_read).expect("something went wrong reading");
        if n_read != to_read {
            panic!("fewer bytes read ({}) than announced in blob header ({})", n_read, to_read);
        }
        let header = osm::HeaderBlock::decode(&mut self.bytes).expect("couldn't read header block");
        self.current.push((
            start_offset,
            to_read,
            OSMStreamItem::Header(header)
        ));
        Ok(true)
    }

    fn read_primitive_block(&mut self) -> Result<bool, OSMPBFParseError> {
        let start_offset = self.n_bytes_read;
        let mut to_read: usize = 0;
        if let Some((_, _, OSMStreamItem::Blob(blob))) = self.current.last() {
            to_read = blob.header.datasize.try_into().expect("blob datasize should be positive and fit into a usize");
        } else {
            panic!("read_primitive_block expects a blob on the parsing stack")
        }
        let n_read = self._read_bytes_upto(to_read).expect("something went wrong reading");
        if n_read != to_read {
            panic!("fewer bytes read ({}) than announced in blob header ({})", n_read, to_read);
        }
        let decoded_data = osm::PrimitiveBlock::decode(&mut self.bytes).expect("couldn't read data block");
        let string_table: Vec<String> = decoded_data.stringtable.s.iter().map(|str| std::str::from_utf8(&str).expect("invalid string").to_string()).collect();
        self.current.push(
            (
                start_offset,
                to_read,
                OSMStreamItem::Block(OSMBlock {
                    block: decoded_data,
                    string_table: string_table 
                })
            )
        );
        Ok(true)
    }

    /// Skip parsing the remainder of the current object on top of the stack
    fn skip(&mut self) {
        if let Some((top_start, top_size, _)) = self.current.last() {
            let remaining: i64 = (top_start + top_size) as i64 - self.n_bytes_read as i64;
            assert!(remaining >= 0);
            if let Some(mut seekable) = self.in_stream.dyn_seekable_inner() {
                std::io::Seek::seek(&mut *seekable, std::io::SeekFrom::Current(remaining));
                self.n_bytes_read += remaining as usize;
                return
            }
        }   
        panic!("could not seek")
    }

}

use std::{cell::{RefCell, RefMut}, marker::PhantomData, rc::Rc};

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

    let mut parser = OSMStreamingReader::new(buffered_file);

    /// has_something[lon][lat] == 1 iff. something in the planet.pbf file is located between [lon,lon+1), and [lat,lat+1)
    let mut has_something = [[false; 181]; 361];  
    while parser.advance() {
        println!("advanced");
        if !matches!(parser.current.last(), Some((_, _, OSMStreamItem::Block(_)))) {
            println!("not a block");
            continue;
        }
        if n_blobs_read >= max_n_blobs {
            break;
        }
        n_blobs_read += 1;

        let blob = match parser.current.get(parser.current.len()-2) {
            Some((_, _, OSMStreamItem::Blob(blob))) => blob,
            _ => panic!("no header on stack?")
        };
        let block = match parser.current.last() {
            Some((_, _, OSMStreamItem::Block(block))) => block,
            _ => panic!("unreachable")
        };

        println!("Blob {:9}: Type: {}  Size: {}  Header size: {}", n_blobs_read, blob.header.r#type, blob.header.datasize, blob.raw_size);

        println!("Uncompressed size: {}  Data type: {:?}", blob.header.datasize, match blob.compression {
            OSMBlobCompression::Raw => "raw",
            OSMBlobCompression::Zlib => "zlib",
            _ => "unknown"
        });

        if blob.header.r#type == "OSMData" {
            // OSM wiki: A primitive block contains _all_ the information to decompress the entities it contains
            // -> I assume this means any referred IDs in relations/ways will be contained in this block
            let string_table: &Vec<String> = &block.string_table;
            let lat_offset = block.block.lat_offset();
            let lon_offset = block.block.lon_offset();
            let granularity = block.block.granularity();
            let coord_decode = |coord, offset| -> f64 {
                1e-9 * ((offset + (granularity as i64) * coord) as f64)
            };
            // Each primitive group is one of Nodes, DenseNodes, Ways, Relations  -- not multiple
            for group in block.block.primitivegroup.iter() {
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

#[derive(Debug)]
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