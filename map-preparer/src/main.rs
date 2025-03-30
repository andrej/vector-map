/*
Keeping track of release build performance:

commit      comment                                 max_n_blobs     time
f794dbdf515 non-streaming, decompress from buffer   64              2.21 s
3d9ad8b     decode dense nodes, map latlon          64              2.69 s
b34613a     decode nodes with new struct, map       64              4.17 s  (battery power)
*/

trait IntoDecompressor {
    fn into_decompressor<'a>(self) -> Box<dyn std::io::Read>;
}

impl IntoDecompressor for osm::blob::Data {
    fn into_decompressor<'a>(self) -> Box<dyn std::io::Read> {
        match self {
            osm::blob::Data::Raw(data_vec) => Box::new(std::io::Cursor::new(data_vec)),
            osm::blob::Data::ZlibData(data_vec) => Box::new(flate2::bufread::ZlibDecoder::new(std::io::Cursor::new(data_vec))),
            _ => panic!("unsupported compression type")
        }
    }
}

const STREAM_BUF_MAX_HEADER_SIZE: usize = 64*1024; 
const STREAM_BUF_MAX_BLOB_SIZE: usize = 32*1024*1024; 
struct OSMStreamingReader<InStreamT> 
where InStreamT: std::io::BufRead
{
    in_stream: InStreamT,
    n_bytes_read: usize,
    /// Read only up to `max_len` bytes
    max_len: usize,
    bytes: bytes::BytesMut,
    current_blob_header: Option<osm::BlobHeader>,
    current_blob: Option<osm::Blob>
}

impl<InStreamT> OSMStreamingReader<InStreamT> 
where InStreamT: std::io::BufRead
{

    pub fn new(in_stream: InStreamT) -> Self {
        Self::new_with_max_len(in_stream, 0)
    }

    pub fn new_with_max_len(in_stream: InStreamT, max_len: usize) -> Self {
        Self {
            in_stream: in_stream,
            n_bytes_read: 0,
            max_len: max_len,
            bytes: bytes::BytesMut::with_capacity(STREAM_BUF_MAX_BLOB_SIZE),
            current_blob_header: None,
            current_blob: None
        }
    }

    /// Attempt to make `n` bytes available for reading in self.bytes. If the
    /// internal cursor is not at the end of the available data by `n` or more,
    /// no data is read; otherwise, we read only the amount needed. The only
    /// case in which this function makes less than `n` bytes available for
    /// reading is if we encounter EOF beforehand.
    fn _read_bytes_upto(&mut self, mut n: usize) -> Result<usize, OSMPBFParseError> {
        assert!(n < STREAM_BUF_MAX_BLOB_SIZE);
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
        while let Ok(n_read_this_iter) = std::io::Read::read(&mut self.in_stream, &mut buf[i..n]) {
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
    
    /// Reads a blob header from the input stream and store the blob header.
    /// Instead of decoding (and copying) the entire compressed contents, we then leave the buffer at that state to read
    /// sequentially
    fn read_blob_header(&mut self) -> Result<bool, OSMPBFParseError> {
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
        let blob_header = osm::BlobHeader::decode(&mut self.bytes).map_err( OSMPBFParseError::new)?;
        self._reset_bytes(); // all bytes have been read and decoded into blob_header

        self.current_blob_header = Some(blob_header);

        Ok(true)
    }

    /// Read a blob into a readable stream of decompressed bytes
    fn read_blob(&mut self) -> Result<(usize, Box<dyn std::io::Read>), OSMPBFParseError> {
        if let Some(blob_header) = &self.current_blob_header {
            let blob_size: usize = blob_header.datasize.try_into().map_err(OSMPBFParseError::new)?;
            self._read_bytes_upto(blob_size)?;
            let blob = osm::Blob::decode(&mut self.bytes).map_err(OSMPBFParseError::new)?;
            self._reset_bytes(); // all bytes have been read and decoded into blob
            let size: usize = blob.raw_size().try_into().map_err(OSMPBFParseError::new)?;
            Ok((size, IntoDecompressor::into_decompressor(blob.data.ok_or(OSMPBFParseError::from_str("blob has no associated data"))?)))
        } else {
            Err(OSMPBFParseError::from_str("read_blob called without prior call to read_blob_header"))
        }
    }
}

impl<InStreamT> Iterator for OSMStreamingReader<InStreamT> 
where InStreamT: std::io::BufRead {
    type Item = (osm::BlobHeader, usize, Box<dyn std::io::Read>);
    fn next(&mut self) -> Option<Self::Item> {
        let has_more = self.read_blob_header().expect("reading blob header failed");
        if !has_more {
            return None
        }
        let (stream_size, stream) = self.read_blob().expect("reading blob failed");
        let blob_header = std::mem::replace(&mut self.current_blob_header, None)
            .expect("should be unreachable: blob_header uninitialized after call to read_blob_header returned true");
        Some((blob_header, stream_size, stream))
    }
}

use std::{cell::{RefCell, RefMut}, iter::Map, marker::PhantomData, rc::Rc};

use map_preparer::osm;
use plotters::style::SizeDesc;
use prost::{bytes::{self, BytesMut}, Message};

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

fn read_string_table(stringtable: &osm::StringTable) -> Vec<&str> {
    let mut ret = Vec::with_capacity(stringtable.s.len());
    for item in &stringtable.s {
        ret.push(std::str::from_utf8(&item[..]).expect("invalid unicode string in stringtable"))
    }
    ret
}

#[derive(Eq, PartialEq, Hash)]
struct GroupLocalOSMString<'a> {
    string_table_id: usize,
    // Below is to ensure that the index above lives for as long as the string
    // table it references
    _marker: std::marker::PhantomData<&'a ()>
}

impl<'a> GroupLocalOSMString<'a> {
    fn from_index(string_table: &'a Vec<&str>, index: usize) -> GroupLocalOSMString<'a> {
        assert!(index < string_table.len());
        Self {
            string_table_id: index,
            _marker: std::marker::PhantomData
        }
    }

    fn from_string(string_table: &'a Vec<&str>, string: &str) -> Option<GroupLocalOSMString<'a>> {
        string_table.iter()
            .position(|x| *x == string)
            .map(|index| {
                Self {
                    string_table_id: index,
                    _marker: std::marker::PhantomData
                }
            })
    }

    fn str(&self, string_table: &'a Vec<&str>) -> &'a str {
        string_table[self.string_table_id]
    }
}

#[derive(Clone, Copy, Debug)]
struct Coord {
    lat: f64,
    lon: f64
}

fn kv_to_hashmap<'a>(string_table: &'a Vec<&str>, keys_vals: &Vec<i32>) -> std::collections::HashMap<GroupLocalOSMString<'a>, GroupLocalOSMString<'a>> {
    // OSM wiki: index=0 is used as a delimiter when encoding DenseNodes
    // Each node's tags are encoded in alternating <keyid> <valid>
    // As an exception, if no node in the current block has any key/value pairs, this array does not contain any delimiters, but is simply empty
    let mut kv_iter = keys_vals.iter();
    let has_kv = keys_vals.len() > 0;
    let mut kv = std::collections::HashMap::new();
    if has_kv {
        kv.reserve((keys_vals.len() + 1) / 2);
        while let Some(k) = kv_iter.next() {
            if *k == 0 {
                break;
            }
            let k = GroupLocalOSMString::from_index(
                &string_table, 
                *k as usize
            );
            let v = GroupLocalOSMString::from_index(
                &string_table, 
                *kv_iter.next().expect("kv iter not multiple of 2 length? key with no value?") as usize
            );
            kv.insert(k, v);
        }
    }
    kv
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).ok_or(std::io::Error::new(std::io::ErrorKind::InvalidInput, "missing input file path command line argument"));
    let mut file = std::fs::File::open(path?)?;
    let mut buffered_file = std::io::BufReader::new(file); //new(file);

    let mut buffer: [u8; BUFFER_SIZE] = [0; BUFFER_SIZE];
    let mut n_blobs_read = 0;
    let skip_n_blobs = 0;
    let max_n_blobs = 64;

    let mut parser = OSMStreamingReader::new(buffered_file);

    let mut buf = vec![0 as u8; STREAM_BUF_MAX_BLOB_SIZE];

    /// has_something[lon][lat] == 1 iff. something in the planet.pbf file is located between [lon,lon+1), and [lat,lat+1)
    //let mut has_something = [[false; 181]; 361];  
    let mut nodes = std::collections::HashMap::<usize, Coord>::new(); 
    let mut lines = Vec::<Vec<Coord>>::new();

    for (blob_header, blob_size, mut blob_stream) in &mut parser {
        if n_blobs_read >= max_n_blobs {
            break;
        }
        n_blobs_read += 1;

        println!("Blob {:9}:   Type: {}  Compressed Size: {}  Uncompressed Size: {}", n_blobs_read, blob_header.r#type, blob_header.datasize, blob_size);

        if blob_header.r#type == "OSMData" {
            assert!(blob_size <= STREAM_BUF_MAX_BLOB_SIZE);
            std::io::Read::read_exact(&mut blob_stream, &mut buf[0..blob_size]).expect("couldn't read raw primitive bytes");
            let block = osm::PrimitiveBlock::decode(&buf[0..blob_size]).expect("couldn't decode primitive");

            // OSM wiki: A primitive block contains _all_ the information to decompress the entities it contains
            // -> I assume this means any referred IDs in relations/ways will be contained in this block
            let string_table = read_string_table(&block.stringtable);
            let lat_offset = block.lat_offset();
            let lon_offset = block.lon_offset();
            let granularity = block.granularity();
            let coord_decode = |coord, offset| -> f64 {
                1e-9 * ((offset + (granularity as i64) * coord) as f64)
            };


            // Each primitive group is one of Nodes, DenseNodes, Ways, Relations  -- not multiple
            for group in block.primitivegroup.iter() {

                for way in &group.ways {
                    println!("found a way!");
                    let mut new_line = Vec::<Coord>::new();
                    new_line.reserve(way.refs.len());
                    let mut last_reference: i64 = 0;
                    for reference in &way.refs {
                        let node_id: usize = (last_reference + reference).try_into().expect("must result in a positive integer node ID");
                        assert!(nodes.contains_key(&node_id));
                        new_line.push(nodes[&node_id]);
                        last_reference = *reference;
                    }
                }

                //for relation in &group.relations {
                //}

                //for node in &group.nodes {
                //}

                if let Some(dense_nodes) = &group.dense {
                    nodes.reserve(dense_nodes.id.len());

                    let mut last_id: i64 = 0;
                    let mut last_lat: i64 = 0;
                    let mut last_lon: i64 = 0;
                    for (raw_id, (raw_lat, raw_lon)) in dense_nodes.id.iter().zip(dense_nodes.lat.iter().zip(dense_nodes.lon.iter())) {
                        let id = last_id + *raw_id;
                        let lat_i = last_lat + *raw_lat;
                        let lon_i = last_lon + *raw_lon;
                        let lat = coord_decode(lat_i, lat_offset);
                        let lon = coord_decode(lon_i, lon_offset);
                        if let Some(x) = nodes.insert(id as usize, Coord { lat, lon }) {
                            println!("duplicate entry for {}", id);
                        }
                        last_id = id;
                        last_lat = lat_i;
                        last_lon = lon_i;
                    }
                }
            }
        }

    }
    println!("{} blobs read ({} bytes)", n_blobs_read, parser.n_bytes_read);
    plot_points(&nodes);
    //plot_lines(&lines);
    Ok(())
}

fn plot_points(points: &std::collections::HashMap<usize, Coord>) {
    use plotters::prelude::*;

    let width = 512;
    let height = 512;

    let root_area = BitMapBackend::new("plot.png", (width, height))
    .into_drawing_area();
    root_area.fill(&WHITE).unwrap();
    let mut ctx = ChartBuilder::on(&root_area).build_cartesian_2d(0..512, 0..512).unwrap();
    ctx.configure_mesh().draw().unwrap();

    ctx.draw_series(
        points.iter().map(|(index, point)| { 
            let (x, y) = web_mercator(f64::to_radians(point.lat), f64::to_radians(point.lon));
            Circle::new(
                (x as i32, y as i32),
                1.0,
                &BLACK
            )
        })
    ).unwrap(); 

}

fn plot_lines(lines: &Vec<Vec<Coord>>) {
    use plotters::prelude::*;

    let width = 512;
    let height = 512;

    let root_area = BitMapBackend::new("plot.png", (width, height))
    .into_drawing_area();
    root_area.fill(&WHITE).unwrap();
    let mut ctx = ChartBuilder::on(&root_area).build_cartesian_2d(0..512, 0..512).unwrap();
    ctx.configure_mesh().draw().unwrap();

    for points in lines {
        ctx.draw_series(
            LineSeries::new(
                points.iter().map(|point| {
                    let (x, y) = web_mercator(f64::to_radians(point.lat), f64::to_radians(point.lon));
                    (x as i32, y as i32)
                }),
                &BLACK
            )
        ).unwrap(); 
    }

}

#[derive(Debug)]
struct OSMPBFParseError {
    root_cause: Option<Box<dyn std::error::Error>>, 
    str: Option<&'static str>
}

impl OSMPBFParseError {
    fn new(root_cause: impl std::error::Error + 'static) -> Self { // the + 'static here constrains the lifetimes of any potential references contained within the error
        Self {
            root_cause: Some(Box::new(root_cause) as Box<dyn std::error::Error>),
            str: None
        }
    }
    fn from_str(str: &'static str) -> Self {
        Self {
            root_cause: None,
            str: Some(str)
        }
    }
}