
mod partially_compressed;
mod osm_stream;
mod osm_pbf;
mod protobuf_helpers;
mod instrumented_reader;

use clap::Parser;
use osm_stream::OsmStream;
use tracing::info;
use tracing_subscriber::FmtSubscriber;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

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
    count_nodes: bool,

    /// Number of parallel threads for entity processing (0 = sequential)
    #[arg(short = 'j', long = "threads", default_value_t = 1)]
    threads: usize,

    /// Verbosity
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbosity: u8,
}

#[derive(Default, Clone)]
struct ThreadStatus {
    bytes_read: u64,
    file_pos: u64,
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
        for _ in 0..args.skip_blobs {
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

    if args.count_nodes {
        if args.threads == 0 {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "Thread count cannot be zero"));

        } else if args.threads == 1 {
            // Sequential processing
            let mut node_count = 0;
            for node in osmstream.nodes() {
                node_count += 1;
            }
            println!("\nTotal entities: {}", node_count);
        } else {
            // Parallel processing
            // Collect blob positions (skip headers and apply skip_blobs)
            let blob_infos: Vec<_> = osmstream.blobs()
                .filter_map(|r| r.ok())
                .filter(|b| b.blob_type == "OSMData")
                .skip(args.skip_blobs)
                .collect();
            
            let num_threads = args.threads.min(blob_infos.len());
            rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build_global()
                .unwrap();
            info!("Using {} threads for parallel processing of {} blobs", num_threads, blob_infos.len());
            
            // Thread positions for progress bar
            let thread_status = Arc::new(Mutex::new(vec![ThreadStatus::default(); num_threads]));
            let running = Arc::new(std::sync::atomic::AtomicBool::new(true));
            let running_clone = running.clone();
            
            // Spawn progress thread (as daemon)
            let file_size = metadata.len();
            let start_time = std::time::Instant::now();
            let thread_status_clone = thread_status.clone();
            let progress_handle = std::thread::spawn(move || {
                while running_clone.load(std::sync::atomic::Ordering::Relaxed) {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    let thread_status = thread_status_clone.lock().unwrap();
                    print_progress_parallel(&thread_status, file_size, start_time);
                }
            });
            
            // Process blobs in parallel
            let total: usize = blob_infos.par_iter().enumerate().map(|(idx, blob_info)| {
                let file = std::fs::File::open(&filename).unwrap();
                let thread_idx = idx % num_threads;
                let thread_status= thread_status.clone();
                
                // Wrap file in InstrumentedReader to track position
                let instrumented = instrumented_reader::InstrumentedReader::with_frequency(
                    file,
                    1024 * 1024, // Update every 1MB
                    move |bytes_read, file_pos| {
                        let mut thread_status = thread_status.lock().unwrap();
                        thread_status[thread_idx].file_pos = file_pos;
                        thread_status[thread_idx].bytes_read = bytes_read;
                    }
                );
                
                let reader = std::io::BufReader::with_capacity(args.chunk_size, instrumented);
                let mut blob_stream = blob_info.create_reader(reader).unwrap();
                
                let mut count = 0;
                for _entity in blob_stream.nodes() {
                    count += 1;
                }
                count
            }).sum();

            running.store(false, std::sync::atomic::Ordering::Relaxed);
            progress_handle.join().unwrap();
            
            println!("\nTotal entities: {}", total);
        }
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

fn print_progress_parallel(thread_status: &[ThreadStatus], file_size: u64, start_time: std::time::Instant) {
    let now = std::time::Instant::now();
    let elapsed = now.duration_since(start_time).as_secs_f64();
    
    // Calculate average progress
    let total_read: u64 = thread_status.iter().map(|ts| ts.bytes_read).sum();
    let read_percentage = (total_read as f64 / file_size as f64) * 100.0;
    let throughput_mib = (total_read as f64 / elapsed) / (1024.0 * 1024.0);
    
    let bar_width = 80 - 20;
    let mut bar = vec![' '; bar_width];
    
    // Place thread numbers on progress bar
    for (i, ts) in thread_status.iter().enumerate() {
        let percentage = (ts.file_pos as f64 / file_size as f64) * 100.0;
        let bar_pos = ((percentage / 100.0) * bar_width as f64) as usize;
        if bar_pos < bar_width {
            bar[bar_pos] = char::from_digit((i + 1) as u32, 10).unwrap_or('*');
        }
    }
    
    let bar_str: String = bar.iter().collect();
    
    eprint!("\r[{}] {:>5.2}% {:>6.1} MiB/s", bar_str, read_percentage, throughput_mib);
}