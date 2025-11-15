
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

    /// Skip the first N blobs.
    #[arg(long, default_value_t = 0)]
    skip_blobs: usize,

    /// List blobs in the file without decoding them
    #[arg(long)]
    count_blobs: bool,

    /// List entities in the file
    #[arg(long)]
    count_entities: bool,

    /// Verbosity
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbosity: u8,
}

fn main() -> std::io::Result<()> {
    let args = ClArgs::parse();

    let tracing_subscriber = FmtSubscriber::builder().with_max_level(
        match args.verbosity {
            0 => tracing::Level::WARN,
            1 => tracing::Level::INFO,
            2 => tracing::Level::DEBUG,
            _ => tracing::Level::TRACE,
        }
    ).finish();
    tracing::subscriber::set_global_default(tracing_subscriber).expect("setting default tracing subscriber failed");

    let filename = std::path::Path::new(&args.osm_file);
    let file = std::fs::File::open(filename)?;
    let metadata = std::fs::metadata(filename).unwrap();
    let file_size = metadata.len();
    let start_time = std::time::Instant::now();
    let mut counting_reader = instrumented_reader::InstrumentedReader::with_callback(file, |bytes_read, file_pos| {
        let percentage = (file_pos as f64 / file_size as f64) * 100.0;
        print_progress(percentage, bytes_read, file_pos, start_time);
    });
    let stream = std::io::BufReader::with_capacity(args.chunk_size, &mut counting_reader);

    if args.baseline {
        baseline(stream);
        return Ok(());
    }
    
    let mut osmstream = OsmStream::new(stream);

    if args.skip_blobs > 0 {
        let mut blob_iterator = osmstream.blobs();
        for i in 0..args.skip_blobs {
            match blob_iterator.next() {
                Some(Ok(_)) => {},
                Some(Err(e)) => return Err(e),
                None => break,
            }
        }

    }

    if args.count_blobs {
        let mut blob_count = 0;
        let mut header_count = 0;
        let mut data_count = 0;
        
        for blob_result in osmstream.blobs() {
            let blob = blob_result?;
            blob_count += 1;
            match blob.blob_type.as_str() {
                "OSMHeader" => header_count += 1,
                "OSMData" => data_count += 1,
                _ => {}
            }
        }
        
        println!("\nSummary:");
        println!("  Total blobs: {}", blob_count);
        println!("  Header blobs: {}", header_count);
        println!("  Data blobs: {}", data_count);
        return Ok(());
    }

    if args.count_entities {
        let mut entity_count = 0;
        for entity in osmstream {
            entity_count += 1;
        }
        println!("\nTotal entities: {}", entity_count);
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

fn print_progress(percentage: f64, bytes_read: u64, file_pos: u64, start_time: std::time::Instant) {
    let now = std::time::Instant::now();
    let throughput = bytes_read as f64 / now.duration_since(start_time).as_secs_f64();
    let throughput_mib = throughput / (1024.0 * 1024.0);
    
    // Calculate progress bar width: 80 chars total - brackets - percentage - throughput - spaces
    // Format: "[...] XX.X% XXX.X MiB/s"
    // Reserve: 2 (brackets) + 1 (space) + 6 (percentage) + 1 (space) + 10 (throughput) = 20 chars
    let bar_width = 80 - 20;
    let progress_pos = ((percentage / 100.0) * bar_width as f64) as usize;
    let progress_pos = progress_pos.min(bar_width);
    
    // Build progress bar with 'o' at position and spaces elsewhere
    let mut bar = String::with_capacity(bar_width);
    for i in 0..bar_width {
        if i == progress_pos && progress_pos > 0 {
            bar.push('o');
        } else {
            bar.push(' ');
        }
    }
    
    eprint!("\r[{}] {:>5.2}% {:>6.1} MiB/s", bar, percentage, throughput_mib);
}