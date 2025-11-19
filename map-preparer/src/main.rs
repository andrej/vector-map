mod instrumented_reader;
mod osm_pbf;
mod osm_stream;
mod partially_compressed;
mod protobuf_helpers;

use clap::Parser;
use osm_stream::{BlobInfo, OsmNode, OsmStream};
use rayon::prelude::*;
use std::io::{Read, Seek, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tracing::{info, trace};
use tracing_subscriber::FmtSubscriber;

const CHUNK_SIZE: usize = 8192;
// Web Mercator practical latitude clamp to avoid infinite projection values at the poles.
// This cutoff is chosen such that the resulting map is square.
const MAX_MERCATOR_LAT: f64 = 85.051128;

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
    #[arg(short = 'c', long = "chunk-size", default_value_t = 1_048_576)]
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

    /// Number of chunks per thread for load balancing
    #[arg(long = "chunks-per-thread", default_value_t = 4)]
    chunks_per_thread: usize,

    /// Verbosity
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbosity: u8,

    /// Optional explicit Mercator scale (R). If omitted, a scale is chosen so max dimension ~= 1024.
    #[arg(long = "scale")]
    scale: Option<f64>,
}

struct ThreadStatus {
    total_bytes_completed: AtomicU64,  // Bytes read from already completed chunks
    current_chunk_start: AtomicU64,
    current_chunk_pos: AtomicU64,
}

impl ThreadStatus {
    fn new() -> Self {
        Self {
            total_bytes_completed: AtomicU64::new(0),
            current_chunk_start: AtomicU64::new(0),
            current_chunk_pos: AtomicU64::new(0),
        }
    }
}

/// Dynamic grid storing counts in row-major order.
#[derive(Debug, Clone)]
struct DynamicGrid {
    width: usize,
    height: usize,
    data: Vec<u32>,
}

impl DynamicGrid {
    fn new(width: usize, height: usize) -> Self {
        Self { width, height, data: vec![0u32; width * height] }
    }
    #[inline]
    fn inc(&mut self, x: usize, y: usize) {
        let idx = y * self.width + x;
        self.data[idx] = self.data[idx].saturating_add(1);
    }
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
                args.chunks_per_thread,
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

        // Determine scale: user-specified or derived to meet target max dimension.
        const TARGET_MAX: f64 = 1024.0;
        let scale_r = args.scale.unwrap_or_else(|| compute_scale_for_target_max(TARGET_MAX));
        let (width, height) = mercator_size(scale_r);
        let grid_width = width.ceil() as usize;
        let grid_height = height.ceil() as usize;
        info!(
            "Scale R {:.4} | width {:.2} height {:.2} -> grid {}x{}{}",
            scale_r, width, height, grid_width, grid_height,
            if args.scale.is_none() { " (auto)" } else { " (user)" }
        );
        println!("Generating {}x{} density grid (scale {:.4})...", grid_width, grid_height, scale_r);

        // Precompute latitude -> y-bin lookup to avoid per-node trig/log
        let lat_samples_per_degree = compute_lat_samples_per_degree(scale_r).max(1);
        let lat_lut = build_lat_to_ybin_lut(scale_r, lat_samples_per_degree);
        trace!("Latitude to Y-bin LUT: samples_per_degree {}, entries {}", lat_samples_per_degree, lat_lut.len());
        let lon_scale_per_degree = grid_width as f64 / 360.0;

        // Closure maps a node's lat/lon into grid indices using lookup and simple scaling.
        let map_fn = move |node: OsmNode| {
            latlon_to_web_mercator_bins(
                node.lat,
                node.lon,
                lon_scale_per_degree,
                lat_samples_per_degree,
                &lat_lut,
            )
        };

        let fold_fn = |mut grid: Box<DynamicGrid>, maybe_coords: Option<(usize, usize)>| {
            if let Some((x, y)) = maybe_coords {
                grid.inc(x, y);
            }
            grid
        };

        let merge_fn = |mut grid1: Box<DynamicGrid>, grid2: Box<DynamicGrid>| {
            // Both grids have identical dimensions
            for i in 0..grid1.data.len() {
                grid1.data[i] = grid1.data[i].saturating_add(grid2.data[i]);
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
                Box::new(DynamicGrid::new(grid_width, grid_height)),
                fold_fn,
            )
        } else {
            process_nodes_parallel(
                filename,
                args.chunk_size,
                args.blob_scan_chunk_size,
                args.skip_blobs,
                args.threads,
                args.chunks_per_thread,
                map_fn,
                Box::new(DynamicGrid::new(grid_width, grid_height)),
                fold_fn,
                merge_fn,
            )
        };

        println!("\nSaving density grid to binary file...");
        save_dynamic_grid_to_file(&grid, output_path)?;
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
    F1: Fn(OsmNode) -> T1 + Sync + Send,
    F2: Fn(T2, T1) -> T2 + Sync + Send,
    F3: Fn(T2, T2) -> T2 + Sync + Send,
>(
    filename: &std::path::Path,
    chunk_size: usize,
    blob_scan_chunk_size: usize,
    skip_blobs: usize,
    n_threads: usize,
    chunks_per_thread: usize,
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
    // Each thread processes multiple chunks to improve load balancing (chunks aren't the same size, so when a thread finishes early it can pick up another)
    let blobs_per_chunk = (blob_infos.len() + num_threads - 1) / num_threads / chunks_per_thread;
    let mut blob_chunks: Vec<Vec<_>> = Vec::with_capacity(num_threads * chunks_per_thread);
    for i in 0..(num_threads * chunks_per_thread) {
        let start = i * blobs_per_chunk;
        let end = ((i + 1) * blobs_per_chunk).min(blob_infos.len());
        if start < blob_infos.len() {
            blob_chunks.push(blob_infos[start..end].to_vec());
        }
    }

    info!(
        "Using {} threads for parallel processing of {} blobs",
        num_threads,
        blob_infos.len()
    );

    // Thread positions for progress bar
    let metadata = std::fs::metadata(filename).unwrap();
    let file_size = metadata.len();
    let thread_status: Arc<Vec<ThreadStatus>> = Arc::new(
        (0..num_threads).map(|_| ThreadStatus::new()).collect()
    );
    let running = Arc::new(std::sync::atomic::AtomicBool::new(true));
    let running_clone = running.clone();

    // Spawn progress thread
    let start_time = std::time::Instant::now();
    let thread_status_clone = thread_status.clone();
    let progress_handle = std::thread::spawn(move || {
        let bar_width = 80 - 20;
        let mut bar = vec![' '; bar_width];
        while running_clone.load(std::sync::atomic::Ordering::Relaxed) {
            std::thread::sleep(std::time::Duration::from_millis(200));  // 5 times per second
            print_progress_parallel(&thread_status_clone, file_size, start_time, &mut bar);
        }
    });

    let start_time = std::time::Instant::now();

    // Process blob chunks in parallel - each thread gets a contiguous slice
    let (n_nodes, result) = blob_chunks
        .par_iter()
        .filter_map(|chunk| {
            if chunk.is_empty() {
                return None;
            }

            let thread_idx = rayon::current_thread_index().unwrap();
            let file = std::fs::File::open(&filename).unwrap();

            // Calculate the contiguous range for this chunk
            let start_pos = chunk[0].position;
            let end_blob = &chunk[chunk.len() - 1];
            let total_size = (end_blob.position + end_blob.size) - start_pos;

            // Set the current chunk boundaries for this thread
            let thread_status = thread_status.clone();
            thread_status[thread_idx].current_chunk_start.store(start_pos, Ordering::Relaxed);
            thread_status[thread_idx].current_chunk_pos.store(start_pos, Ordering::Relaxed);

            // Wrap file in InstrumentedReader to track position
            let thread_status_clone = thread_status.clone();
            let mut instrumented = instrumented_reader::InstrumentedReader::with_frequency(
                file,
                10 * 1024 * 1024,  // FIXME: For some reason, updating more frequently seems to speed things up?! Not sure why.
                move |_bytes_read, file_pos| {
                    thread_status_clone[thread_idx].current_chunk_pos.store(file_pos, Ordering::Relaxed);
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
            let result = stream
                .nodes()
                .map(&map_callback)
                .fold((0, fold_initial.clone()), |acc, item| (acc.0 + 1, fold_callback(acc.1, item)));
            
            // After completing this chunk, add its size to total completed
            // and reset current chunk tracking to avoid double-counting
            thread_status[thread_idx].total_bytes_completed.fetch_add(total_size, Ordering::Relaxed);
            thread_status[thread_idx].current_chunk_start.store(0, Ordering::Relaxed);
            thread_status[thread_idx].current_chunk_pos.store(0, Ordering::Relaxed);
            
            Some(result)
        })
        .reduce(|| (0, fold_initial.clone()), |acc, item| (acc.0 + item.0, merge_callback(acc.1, item.1)));

    running.store(false, std::sync::atomic::Ordering::Relaxed);
    progress_handle.join().unwrap();
    
    println!(
        "\nProcessed {} nodes in {:.3} s.",
        n_nodes,
        start_time.elapsed().as_secs_f64()
    );
    result
}

fn scan_blobs(filename: &std::path::Path, chunk_size: usize, skip_blobs: usize) -> Vec<BlobInfo> {
    let file = std::fs::File::open(filename).unwrap();
    let metadata = std::fs::metadata(filename).unwrap();
    let file_size = metadata.len();
    let start_time = std::time::Instant::now();
    let instrumented = instrumented_reader::InstrumentedReader::with_frequency(
        file,
        32 * 1024,
        |bytes_read, file_pos| {
            let percentage = (file_pos as f64 / file_size as f64) * 100.0;
            print_progress(percentage, bytes_read, file_pos, start_time);
        },
    );
    let stream = std::io::BufReader::with_capacity(chunk_size, instrumented);
    let mut osmstream = OsmStream::new(stream);

    let start_time = std::time::Instant::now();
    let result: Vec<_>  = osmstream
        .blobs()
        .filter_map(|r| r.ok())
        .filter(|b| b.blob_type == "OSMData")
        .skip(skip_blobs)
        .collect();
    println!("\nScanned {} blobs in {:.3} s.", result.len(), start_time.elapsed().as_secs_f64());

    result
}

fn print_progress(percentage: f64, bytes_read: u64, _file_pos: u64, start_time: std::time::Instant) {
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
    bar: &mut Vec<char>
) {
    let now = std::time::Instant::now();
    let elapsed = now.duration_since(start_time).as_secs_f64();

    // Calculate total progress: completed chunks + current chunk progress for each thread
    let total_processed: u64 = thread_status.iter().map(|ts| {
        let completed = ts.total_bytes_completed.load(Ordering::Relaxed);
        let chunk_start = ts.current_chunk_start.load(Ordering::Relaxed);
        let chunk_pos = ts.current_chunk_pos.load(Ordering::Relaxed);
        let in_progress = chunk_pos.saturating_sub(chunk_start);
        completed + in_progress
    }).sum();
    
    let read_percentage = (total_processed as f64 / file_size as f64) * 100.0;
    let throughput_mib = (total_processed as f64 / elapsed) / (1024.0 * 1024.0);

    let bar_width = 80 - 20;

    // Draw over/onto the bar with the new thread positions and their trails from their start
    for (i, ts) in thread_status.iter().enumerate() {
        // Calculate this thread's current chunk starting position in the bar
        let chunk_start = ts.current_chunk_start.load(Ordering::Relaxed);
        let start_pct = chunk_start as f64 / file_size as f64;
        let start_bar_pos = (start_pct * bar_width as f64) as usize;

        // Calculate current position of this thread within the current chunk
        let chunk_pos = ts.current_chunk_pos.load(Ordering::Relaxed);
        let current_pct = chunk_pos as f64 / file_size as f64;
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

/// Convert latitude/longitude to Web Mercator projection coordinates in meters for a given scale (earth radius R).
/// Uses the standard spherical Mercator (EPSG:3857) equations:
///   x = R * λ
///   y = R * ln(tan(π/4 + φ/2))
/// Latitude is clamped to the Web Mercator practical limits (≈85.051129°) to avoid infinite values.
fn latlon_to_web_mercator(lat: f64, lon: f64, r: f64) -> (f64, f64) {
    let lat = lat.clamp(-MAX_MERCATOR_LAT, MAX_MERCATOR_LAT);
    let lon_rad = lon.to_radians();
    let lat_rad = lat.to_radians();
    let x = r * lon_rad;
    let y = r * ( (std::f64::consts::PI / 4.0 + lat_rad / 2.0).tan().ln() );
    (x, y)
}

/// Return the Mercator projection bounds (min_x, max_x), (min_y, max_y) for a given scale (earth radius R).
/// X spans [-πR, πR]. Y spans symmetric range based on clamped latitude.
fn mercator_bounds(r: f64) -> ((f64, f64), (f64, f64)) {
    let max_x = std::f64::consts::PI * r;
    let min_x = -max_x;
    let max_lat_rad = MAX_MERCATOR_LAT.to_radians();
    let max_y = r * ( (std::f64::consts::PI / 4.0 + max_lat_rad / 2.0).tan().ln() );
    let min_y = -max_y;
    ((min_x, max_x), (min_y, max_y))
}

/// Return the (width, height) of the full Mercator world for scale (earth radius R).
/// Width = 2πR, Height = 2 * R * ln(tan(π/4 + φ_max/2)).
fn mercator_size(r: f64) -> (f64, f64) {
    let ((min_x, max_x), (min_y, max_y)) = mercator_bounds(r);
    (max_x - min_x, max_y - min_y)
}

/// Compute scale R so that max(world_width, world_height) == target_max (approximately).
/// world_width = 2πR, world_height = 2R * ln(tan(π/4 + φ_max/2)). Width is larger, so we just use width.
fn compute_scale_for_target_max(target_max: f64) -> f64 {
    // Since width coefficient (2π) > height coefficient, using width achieves target on larger dimension.
    target_max / (2.0 * std::f64::consts::PI)
}

/// Compute minimum samples-per-degree for latitude lookup to ensure each table step moves at most one y-bin (== pixel == projected whole integer value) across the entire latitude range (worst case at MAX_MERCATOR_LAT).
/// The Mercator projection stretches y values near the poles, so more samples are needed there
/// (smaller change in latitude corresponds to a larger change in y).
/// Mercator latitude projection formula:
///  y = R * ln(tan(pi/4 + lat/2))
fn compute_lat_samples_per_degree(r: f64) -> usize {
    let sec_max = 1.0 / MAX_MERCATOR_LAT.to_radians().cos();
    // Derivative of y w.r.t. latitude (degrees):
    // dy/dlat = R * sec(lat) * (π/180)
    let dy_per_degree_max = r * sec_max * (std::f64::consts::PI / 180.0);
    // Require dy per table step <= 1 bin => steps_per_degree >= dy_per_degree_max
    dy_per_degree_max.ceil() as usize
}

/// Build a lookup table mapping latitude (indexed by scaled degrees) to y-bin index.
/// Table index i corresponds to lat_deg = i / samples_per_degree - MAX_MERCATOR_LAT.
/// Y-axis increases downward (0 at top, grid_height-1 at bottom).
/// Only covers the valid Mercator range [-MAX_MERCATOR_LAT, MAX_MERCATOR_LAT].
fn build_lat_to_ybin_lut(
    r: f64,
    samples_per_degree: usize,
) -> Vec<usize> {
    let lat_range = 2.0 * MAX_MERCATOR_LAT;
    let total_indices = (lat_range * samples_per_degree as f64).ceil() as usize + 1;
    
    // Calculate grid dimensions from scale
    let max_lat_rad = MAX_MERCATOR_LAT.to_radians();
    let max_y = r * ((std::f64::consts::PI / 4.0 + max_lat_rad / 2.0).tan().ln());
    let grid_height = (2.0 * max_y).ceil() as usize;
    
    let mut lut = Vec::with_capacity(total_indices);
    for i in 0..total_indices {
        let lat_deg = (i as f64) / (samples_per_degree as f64) - MAX_MERCATOR_LAT;
        let lat_rad = lat_deg.to_radians();
        let y = r * ((std::f64::consts::PI / 4.0 + lat_rad / 2.0).tan().ln());
        
        // Flip Y axis: max_y (north) -> 0, -max_y (south) -> grid_height-1
        let rel_y = max_y - y;
        let mut y_bin = rel_y.trunc() as isize;
        y_bin = y_bin.clamp(0, (grid_height - 1) as isize);
        lut.push(y_bin as usize);
    }
    lut
}

/// Fast mapping from latitude/longitude (degrees) to grid bins using precomputed LUT for latitude
/// and linear degree-based scaling for longitude.
/// Clamps input coordinates to valid Mercator range.
fn latlon_to_web_mercator_bins(
    lat_deg: f64,
    lon_deg: f64,
    lon_scale_per_degree: f64,
    lat_samples_per_degree: usize,
    lat_lut: &[usize],
) -> Option<(usize, usize)> {
    // Clamp to valid Mercator range
    if lat_deg.abs() > MAX_MERCATOR_LAT {
        return None
    }

    let lon_wrapped = (lon_deg + 180.0) % 360.0;
    let x_bin = (lon_wrapped * lon_scale_per_degree) as usize;

    let y_lut_idx = ((lat_deg + MAX_MERCATOR_LAT) * lat_samples_per_degree as f64) as usize;
    let y_bin = lat_lut[y_lut_idx];

    Some((x_bin as usize, y_bin))
}

/// Save a dynamic density grid to a binary file
/// Format: u32 width, u32 height, followed by width*height little-endian u32 counts row-major
fn save_dynamic_grid_to_file(
    grid: &DynamicGrid,
    output_path: &std::path::Path,
) -> std::io::Result<()> {
    let mut file = std::fs::File::create(output_path)?;
    file.write_all(&(grid.width as u32).to_le_bytes())?;
    file.write_all(&(grid.height as u32).to_le_bytes())?;
    for &value in grid.data.iter() {
        file.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}
