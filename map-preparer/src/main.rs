
mod partially_compressed;
mod osm_stream;
mod osm_pbf;
mod protobuf_helpers;
mod instrumented_reader;

use clap::Parser;
use osm_stream::OsmStream;
use tracing_subscriber::FmtSubscriber;

const CHUNK_SIZE: usize = 8192;

#[derive(clap::Parser)]
struct ClArgs {
    /// Path to the OSM file
    #[arg()]
    osm_file: std::path::PathBuf,

    /// Just sequentially read the file without any processing to get a baseline speed measurement
    #[arg(short, long)]
    baseline: bool,

    /// Buffer chunk size in bytes
    #[arg(short = 'c', long = "chunk-size", default_value_t = 8192)]
    chunk_size: usize,
}

fn main() -> std::io::Result<()> {
    let tracing_subscriber = FmtSubscriber::new();
    tracing::subscriber::set_global_default(tracing_subscriber).expect("setting default tracing subscriber failed");
    let args = ClArgs::parse();

    let filename = std::path::Path::new(&args.osm_file);
    let file = std::fs::File::open(filename)?;
    let metadata = std::fs::metadata(filename).unwrap();
    let file_size = metadata.len();
    let start_time = std::time::Instant::now();
    let mut counting_reader = instrumented_reader::InstrumentedReader::with_callback(file, |bytes_read| {
        let percentage = (bytes_read as f64 / file_size as f64) * 100.0;
        print_progress(percentage, bytes_read, start_time);
    });
    let stream = std::io::BufReader::with_capacity(args.chunk_size, &mut counting_reader);

    if args.baseline {
        baseline(stream);
        return Ok(());
    }
    let mut osmstream = OsmStream::new(stream);

    for entity in osmstream {
        println!("{:?}", entity);
    }
    Ok(())
}

/// Just sequentially read the file without any processing to get a baseline speed measurement.
fn baseline<R: std::io::Read>(mut reader: R) {
    let mut buffer = [0u8; CHUNK_SIZE];
    while let Ok(n) = reader.read(&mut buffer) {
        if n == 0 {
            break;
        }
    }
}

fn print_progress(percentage: f64, bytes_read: u64, start_time: std::time::Instant) {
    let now = std::time::Instant::now();
    let throughput = bytes_read as f64 / now.duration_since(start_time).as_secs_f64();
    eprint!("\rProgress: {:>6.2}% ({:.2} MiB read, {:.2} MiB/s)", percentage, bytes_read as f64 / (1024.0 * 1024.0), throughput / (1024.0 * 1024.0));
}