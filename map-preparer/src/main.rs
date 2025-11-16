mod instrumented_reader;
mod osm_pbf;
mod osm_stream;
mod partially_compressed;
mod protobuf_helpers;

use clap::Parser;
use osm_stream::{BlobInfo, OsmNode, OsmStream};
use rayon::prelude::*;
use std::io::{Read, Seek, Write};
use std::sync::{Arc, Mutex};
use tracing::info;
use tracing_subscriber::FmtSubscriber;

const CHUNK_SIZE: usize = 8192;
const GRID_SIZE: usize = 1024;

#[derive(clap::Parser)]
struct ClArgs {
    /// Path to the OSM file
    #[arg()]
    osm_file: std::path::PathBuf,

    /// Just sequentially read the file without any processing to get a baseline speed measurement
    #[arg(short, long)]
    baseline: bool,

    /// Buffer chunk size for the blob scan
    #[arg(long = "blob-scan-chunk-size", default_value_t = 128)]
    blob_scan_chunk_size: usize,

    /// Buffer chunk size in bytes
    #[arg(short = 'c', long = "chunk-size", default_value_t = 16384)]
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

    /// Generate a density grid from nodes
    #[arg(long)]
    density_grid: bool,

    /// Output binary grid file path
    #[arg(short = 'o', long = "output")]
    output: Option<std::path::PathBuf>,

    /// Number of parallel threads for entity processing (0 = sequential)
    #[arg(short = 'j', long = "threads", default_value_t = 8)]
    threads: usize,

    /// Verbosity
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbosity: u8,
}

#[derive(Default, Clone)]
struct ThreadStatus {
    bytes_read: u64,
    start_pos: u64,
    file_pos: u64,
}

fn main() -> std::io::Result<()> {
    let args = ClArgs::parse();

    let tracing_subscriber = FmtSubscriber::builder()
        .with_max_level(match args.verbosity {
            0 => tracing::Level::WARN,
            1 => tracing::Level::INFO,
            2 => tracing::Level::DEBUG,
            _ => tracing::Level::TRACE,
        })
        .finish();
    tracing::subscriber::set_global_default(tracing_subscriber)
        .expect("setting default tracing subscriber failed");

    let filename = std::path::Path::new(&args.osm_file);

    if args.baseline {
        let file = std::fs::File::open(filename)?;
        let metadata = std::fs::metadata(filename).unwrap();
        let file_size = metadata.len();
        let start_time = std::time::Instant::now();
        let mut counting_reader =
            instrumented_reader::InstrumentedReader::with_callback(file, |bytes_read, file_pos| {
                let percentage = (file_pos as f64 / file_size as f64) * 100.0;
                print_progress(percentage, bytes_read, file_pos, start_time);
            });
        let stream = std::io::BufReader::with_capacity(args.chunk_size, &mut counting_reader);
        baseline(stream);
        return Ok(());
    }

    if args.count_nodes {
        let node_count = if args.threads == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Thread count cannot be zero",
            ));
        } else if args.threads == 1 {
            process_nodes_sequential(
                filename,
                args.chunk_size,
                args.skip_blobs,
                |_node| 1,
                0,
                |acc, v| acc + v,
            )
        } else {
            process_nodes_parallel(
                filename,
                args.chunk_size,
                args.blob_scan_chunk_size,
                args.skip_blobs,
                args.threads,
                |_node| 1,
                0,
                |acc, v| acc + v,
                |acc1, acc2| acc1 + acc2,
            )
        };

        println!("\nTotal entities: {}", node_count);
    }

    if args.density_grid {
        let output_path = args.output.as_ref().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Output path (-o/--output) is required for grid generation",
            )
        })?;

        let map_fn = |node: OsmNode| {
            let (x, y) = latlon_to_web_mercator(node.lat, node.lon);
            let grid_x = ((x * GRID_SIZE as f64).floor() as usize).min(GRID_SIZE - 1);
            let grid_y = ((y * GRID_SIZE as f64).floor() as usize).min(GRID_SIZE - 1);
            (grid_x, grid_y)
        };

        let fold_fn = |mut grid: Box<[[u32; GRID_SIZE]; GRID_SIZE]>, (x, y): (usize, usize)| {
            grid[y][x] = grid[y][x].saturating_add(1);
            grid
        };

        let merge_fn = |mut grid1: Box<[[u32; GRID_SIZE]; GRID_SIZE]>, 
                        grid2: Box<[[u32; GRID_SIZE]; GRID_SIZE]>| {
            for y in 0..GRID_SIZE {
                for x in 0..GRID_SIZE {
                    grid1[y][x] = grid1[y][x].saturating_add(grid2[y][x]);
                }
            }
            grid1
        };

        let grid = if args.threads == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Thread count cannot be zero",
            ));
        } else if args.threads == 1 {
            process_nodes_sequential(
                filename,
                args.chunk_size,
                args.skip_blobs,
                map_fn,
                Box::new([[0u32; GRID_SIZE]; GRID_SIZE]),
                fold_fn,
            )
        } else {
            process_nodes_parallel(
                filename,
                args.chunk_size,
                args.blob_scan_chunk_size,
                args.skip_blobs,
                args.threads,
                map_fn,
                Box::new([[0u32; GRID_SIZE]; GRID_SIZE]),
                fold_fn,
                merge_fn,
            )
        };

        println!("\nSaving density grid to binary file...");
        save_grid_to_file(&grid, output_path)?;
        println!("Grid saved to: {}", output_path.display());
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

fn process_nodes_sequential<T1, T2, F1: Fn(OsmNode) -> T1, F2: Fn(T2, T1) -> T2>(
    filename: &std::path::Path,
    chunk_size: usize,
    skip_blobs: usize,
    map_callback: F1,
    fold_initial: T2,
    fold_callback: F2,
) -> T2 {
    let file = std::fs::File::open(filename).unwrap();
    let metadata = std::fs::metadata(filename).unwrap();
    let file_size = metadata.len();
    let start_time = std::time::Instant::now();
    let counting_reader =
        instrumented_reader::InstrumentedReader::with_callback(file, |bytes_read, file_pos| {
            let percentage = (file_pos as f64 / file_size as f64) * 100.0;
            print_progress(percentage, bytes_read, file_pos, start_time);
        });
    let buf_reader = std::io::BufReader::with_capacity(chunk_size, counting_reader);
    let mut osmstream = OsmStream::new(buf_reader);

    let mut blob_iterator = osmstream.blobs();
    for _ in 0..skip_blobs {
        match blob_iterator.next() {
            Some(Ok(_)) => {}
            Some(Err(e)) => panic!("Error skipping blob: {}", e),
            None => break,
        }
    }
    osmstream
        .nodes()
        .map(map_callback)
        .fold(fold_initial, fold_callback)
}

fn process_nodes_parallel<
    T1: Sync + Send,
    T2: Sync + Send + Clone,
    F1: Fn(OsmNode) -> T1 + Sync + Send + Copy,
    F2: Fn(T2, T1) -> T2 + Sync + Send + Copy,
    F3: Fn(T2, T2) -> T2 + Sync + Send + Copy,
>(
    filename: &std::path::Path,
    chunk_size: usize,
    blob_scan_chunk_size: usize,
    skip_blobs: usize,
    n_threads: usize,
    map_callback: F1,
    fold_initial: T2,
    fold_callback: F2,
    merge_callback: F3,
) -> T2 {
    // Parallel processing
    // Collect blob positions (skip headers and apply skip_blobs)
    let blob_infos = scan_blobs(filename, blob_scan_chunk_size, skip_blobs);

    let num_threads = n_threads.min(blob_infos.len());
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .unwrap();

    // Partition blobs into contiguous chunks, one per thread
    let blobs_per_thread = (blob_infos.len() + num_threads - 1) / num_threads;
    let mut blob_chunks: Vec<Vec<_>> = Vec::with_capacity(num_threads);
    for i in 0..num_threads {
        let start = i * blobs_per_thread;
        let end = ((i + 1) * blobs_per_thread).min(blob_infos.len());
        if start < blob_infos.len() {
            blob_chunks.push(blob_infos[start..end].to_vec());
        }
    }

    info!(
        "Using {} threads for parallel processing of {} blobs ({} blobs/thread)",
        num_threads,
        blob_infos.len(),
        blobs_per_thread
    );

    // Thread positions for progress bar
    let thread_status = Arc::new(Mutex::new(vec![ThreadStatus::default(); num_threads]));
    let running = Arc::new(std::sync::atomic::AtomicBool::new(true));
    let running_clone = running.clone();

    // Spawn progress thread
    let metadata = std::fs::metadata(filename).unwrap();
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

    // Process blob chunks in parallel - each thread gets a contiguous slice
    let result = blob_chunks
        .par_iter()
        .enumerate()
        .filter_map(|(thread_idx, chunk)| {
            if chunk.is_empty() {
                return None;
            }

            let thread_status = thread_status.clone();
            let file = std::fs::File::open(&filename).unwrap();

            // Calculate the contiguous range for this chunk
            let start_pos = chunk[0].position;
            let end_blob = &chunk[chunk.len() - 1];
            let total_size = (end_blob.position + end_blob.size) - start_pos;

            // Set the starting position for this thread
            {
                let mut status = thread_status.lock().unwrap();
                status[thread_idx].start_pos = start_pos;
                status[thread_idx].file_pos = start_pos;
            }

            // Wrap file in InstrumentedReader to track position
            let mut instrumented = instrumented_reader::InstrumentedReader::with_frequency(
                file,
                1024 * 1024, // Update every 1MB
                move |bytes_read, file_pos| {
                    let mut status = thread_status.lock().unwrap();
                    status[thread_idx].file_pos = file_pos;
                    status[thread_idx].bytes_read = bytes_read;
                },
            );

            // Seek to start of this chunk and create a Take reader for the entire range
            instrumented
                .seek(std::io::SeekFrom::Start(start_pos))
                .unwrap();
            let limited = instrumented.take(total_size);

            let reader = std::io::BufReader::with_capacity(chunk_size, limited);
            let mut stream = OsmStream::new(reader);

            // Count all nodes in this contiguous chunk
            Some(
                stream
                    .nodes()
                    .map(map_callback)
                    .fold(fold_initial.clone(), fold_callback),
            )
        })
        .reduce(|| fold_initial.clone(), merge_callback);

    running.store(false, std::sync::atomic::Ordering::Relaxed);
    progress_handle.join().unwrap();
    result
}

fn scan_blobs(filename: &std::path::Path, chunk_size: usize, skip_blobs: usize) -> Vec<BlobInfo> {
    let file = std::fs::File::open(filename).unwrap();
    let metadata = std::fs::metadata(filename).unwrap();
    let file_size = metadata.len();
    let start_time = std::time::Instant::now();
    let instrumented = instrumented_reader::InstrumentedReader::with_frequency(
        file,
        chunk_size as u64,
        |bytes_read, file_pos| {
            let percentage = (file_pos as f64 / file_size as f64) * 100.0;
            print_progress(percentage, bytes_read, file_pos, start_time);
        },
    );
    let stream = std::io::BufReader::with_capacity(chunk_size, instrumented);
    let mut osmstream = OsmStream::new(stream);
    osmstream
        .blobs()
        .filter_map(|r| r.ok())
        .filter(|b| b.blob_type == "OSMData")
        .skip(skip_blobs)
        .collect()
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

    eprint!(
        "\r[{}] {:>5.2}% {:>6.1} MiB/s",
        bar, percentage, throughput_mib
    );
}

fn print_progress_parallel(
    thread_status: &[ThreadStatus],
    file_size: u64,
    start_time: std::time::Instant,
) {
    let now = std::time::Instant::now();
    let elapsed = now.duration_since(start_time).as_secs_f64();

    // Calculate average progress
    let total_read: u64 = thread_status.iter().map(|ts| ts.bytes_read).sum();
    let read_percentage = (total_read as f64 / file_size as f64) * 100.0;
    let throughput_mib = (total_read as f64 / elapsed) / (1024.0 * 1024.0);

    let bar_width = 80 - 20;
    let mut bar = vec![' '; bar_width];

    // Draw trails for each thread showing processed portion
    for (i, ts) in thread_status.iter().enumerate() {
        // Calculate this thread's actual starting position in the bar
        let start_pct = (ts.start_pos as f64 / file_size as f64);
        let start_bar_pos = (start_pct * bar_width as f64) as usize;

        // Calculate current position of this thread
        let current_pct = (ts.file_pos as f64 / file_size as f64);
        let current_bar_pos = (current_pct * bar_width as f64) as usize;

        // Draw trail from actual start position to current position
        let thread_char = char::from_digit((i + 1) as u32, 10).unwrap_or('*');
        for pos in start_bar_pos..current_bar_pos.min(bar_width) {
            bar[pos] = thread_char;
        }

        // Place thread number at current position
        if current_bar_pos < bar_width {
            bar[current_bar_pos] = thread_char;
        }
    }

    let bar_str: String = bar.iter().collect();

    eprint!(
        "\r[{}] {:>5.2}% {:>6.1} MiB/s",
        bar_str, read_percentage, throughput_mib
    );
}

/// Convert latitude/longitude to Web Mercator projection coordinates (0.0 to 1.0 range)
/// Uses the standard Web Mercator projection (EPSG:3857)
fn latlon_to_web_mercator(lat: f64, lon: f64) -> (f64, f64) {
    // Clamp latitude to avoid infinity at poles
    let lat = lat.clamp(-85.051129, 85.051129);
    
    // Longitude: -180 to 180 maps to 0 to 1
    let x = (lon + 180.0) / 360.0;
    
    // Latitude: use optimized Mercator projection formula
    // Original: y = (1.0 - ln(tan(lat_rad) + 1/cos(lat_rad)) / π) / 2.0
    // Simplified using: tan(x) + sec(x) = (1 + sin(x)) / cos(x)
    // Then: ln((1 + sin(x)) / cos(x)) = ln(1 + sin(x)) - ln(cos(x))
    let lat_rad = lat.to_radians();
    let sin_lat = lat_rad.sin();
    
    // Equivalent to the standard formula but avoids tan() and reduces operations
    // ln((1 + sin) / (1 - sin)) = ln(1 + sin) - ln(1 - sin)
    let y = 0.5 - (0.5 * ((1.0 + sin_lat) / (1.0 - sin_lat)).ln() / std::f64::consts::PI);
    
    (x, y)
}

/// Save a density grid to a binary file
/// Format: 1024x1024 grid of u32 values in little-endian format
fn save_grid_to_file(
    grid: &Box<[[u32; GRID_SIZE]; GRID_SIZE]>,
    output_path: &std::path::Path,
) -> std::io::Result<()> {
    let mut file = std::fs::File::create(output_path)?;
    
    // Write grid dimensions as header (for future compatibility)
    file.write_all(&(GRID_SIZE as u32).to_le_bytes())?;
    file.write_all(&(GRID_SIZE as u32).to_le_bytes())?;
    
    // Write all grid values as little-endian u32
    for row in grid.iter() {
        for &value in row.iter() {
            file.write_all(&value.to_le_bytes())?;
        }
    }
    
    Ok(())
}

/// Load a density grid from a binary file
pub fn load_grid_from_file(
    input_path: &std::path::Path,
) -> std::io::Result<Box<[[u32; GRID_SIZE]; GRID_SIZE]>> {
    use std::io::Read;
    
    let mut file = std::fs::File::open(input_path)?;
    
    // Read header
    let mut width_bytes = [0u8; 4];
    let mut height_bytes = [0u8; 4];
    file.read_exact(&mut width_bytes)?;
    file.read_exact(&mut height_bytes)?;
    
    let width = u32::from_le_bytes(width_bytes) as usize;
    let height = u32::from_le_bytes(height_bytes) as usize;
    
    if width != GRID_SIZE || height != GRID_SIZE {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Grid dimensions {}x{} do not match expected {}x{}", width, height, GRID_SIZE, GRID_SIZE),
        ));
    }
    
    // Read grid values
    let mut grid = Box::new([[0u32; GRID_SIZE]; GRID_SIZE]);
    for row in grid.iter_mut() {
        for value in row.iter_mut() {
            let mut bytes = [0u8; 4];
            file.read_exact(&mut bytes)?;
            *value = u32::from_le_bytes(bytes);
        }
    }
    
    Ok(grid)
}
