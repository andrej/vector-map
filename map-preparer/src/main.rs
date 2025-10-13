
mod potentially_compressed;
mod osm_stream;
mod osm_pbf;
mod protobuf_helpers;

use osm_stream::OsmStream;

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <osm_file>", args[0]);
        std::process::exit(1);
    }
    let filename = &args[1];
    let mut osmstream = OsmStream::from_file(std::path::Path::new(filename))?;
    for entity in osmstream {
        println!("{:?}", entity);
    }
    Ok(())
}
