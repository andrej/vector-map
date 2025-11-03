
mod partially_compressed;
mod osm_stream;
mod osm_pbf;
mod protobuf_helpers;
mod instrumented_reader;

use osm_stream::OsmStream;
use tracing_subscriber::FmtSubscriber;

const CHUNK_SIZE: usize = 8192;

fn main() -> std::io::Result<()> {
    let tracing_subscriber = FmtSubscriber::new();
    tracing::subscriber::set_global_default(tracing_subscriber).expect("setting default tracing subscriber failed");
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <osm_file>", args[0]);
        std::process::exit(1);
    }

    let filename = std::path::Path::new(&args[1]);
    let file = std::fs::File::open(filename)?;
    let metadata = std::fs::metadata(filename).unwrap();
    let file_size = metadata.len();
    let mut counting_reader = instrumented_reader::InstrumentedReader::with_callback(file, |bytes_read| {
        let percentage = (bytes_read as f64 / file_size as f64) * 100.0;
        print_progress(percentage, bytes_read);
    });
    let stream = std::io::BufReader::with_capacity(CHUNK_SIZE, &mut counting_reader);
    let mut osmstream = OsmStream::new(stream);

    for entity in osmstream {
        println!("{:?}", entity);
    }
    Ok(())
}

fn print_progress(percentage: f64, bytes_read: u64) {
    eprint!("\rProgress: {:>6.2}% ({:.2} MiB read)", percentage, bytes_read as f64 / (1024.0 * 1024.0));
}