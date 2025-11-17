use clap::Parser;
use plotters::prelude::*;
use std::path::PathBuf;

/// Dynamic grid storing counts in row-major order.
#[derive(Debug)]
struct DynamicGrid {
    width: usize,
    height: usize,
    data: Vec<u32>,
}

#[derive(clap::Parser)]
struct ClArgs {
    /// Input binary grid file
    #[arg()]
    input_grid: PathBuf,

    /// Output PNG file path
    #[arg(short = 'o', long = "output")]
    output: PathBuf,

    /// Minimum color (hex format, e.g., 0000ff for blue)
    #[arg(long, default_value = "0000ff")]
    min_color: String,

    /// Maximum color (hex format, e.g., ffff00 for yellow)
    #[arg(long, default_value = "ffff00")]
    max_color: String,

    /// Verbosity
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbosity: u8,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = ClArgs::parse();

    println!("Loading grid from: {}", args.input_grid.display());
    let grid = load_grid_from_file(&args.input_grid)?;
    println!("Loaded grid: {}x{} cells", grid.width, grid.height);

    println!("Generating heatmap...");
    let min_color = parse_hex_color(&args.min_color)?;
    let max_color = parse_hex_color(&args.max_color)?;
    
    save_heatmap_to_png(&grid, &args.output, min_color, max_color)?;
    
    println!("Heatmap saved to: {}", args.output.display());
    Ok(())
}

/// Load a density grid from a binary file
fn load_grid_from_file(
    input_path: &std::path::Path,
) -> std::io::Result<DynamicGrid> {
    use std::io::Read;
    
    let mut file = std::fs::File::open(input_path)?;
    
    // Read header
    let mut width_bytes = [0u8; 4];
    let mut height_bytes = [0u8; 4];
    file.read_exact(&mut width_bytes)?;
    file.read_exact(&mut height_bytes)?;
    
    let width = u32::from_le_bytes(width_bytes) as usize;
    let height = u32::from_le_bytes(height_bytes) as usize;
    
    // Read grid values
    let mut data = vec![0u32; width * height];
    for value in data.iter_mut() {
        let mut bytes = [0u8; 4];
        file.read_exact(&mut bytes)?;
        *value = u32::from_le_bytes(bytes);
    }
    
    Ok(DynamicGrid { width, height, data })
}

/// Parse a hex color string (e.g., "ff0000" for red) into RGB values
fn parse_hex_color(hex: &str) -> Result<(u8, u8, u8), Box<dyn std::error::Error>> {
    let hex = hex.trim_start_matches('#');
    if hex.len() != 6 {
        return Err(format!("Invalid color format: {}, expected 6 hex digits", hex).into());
    }
    
    let r = u8::from_str_radix(&hex[0..2], 16)?;
    let g = u8::from_str_radix(&hex[2..4], 16)?;
    let b = u8::from_str_radix(&hex[4..6], 16)?;
    
    Ok((r, g, b))
}

/// Paint a heatmap from a 2D grid of counts and save to PNG
fn save_heatmap_to_png(
    grid: &DynamicGrid,
    output_path: &std::path::Path,
    min_color: (u8, u8, u8),
    max_color: (u8, u8, u8),
) -> Result<(), Box<dyn std::error::Error>> {
    // Find max value for normalization
    let max_value = grid
        .data
        .iter()
        .copied()
        .max()
        .unwrap_or(1);
    
    println!("Max value in grid: {}", max_value);
    
    let root = BitMapBackend::new(output_path, (grid.width as u32, grid.height as u32))
        .into_drawing_area();
    root.fill(&BLACK)?;
    
    for y in 0..grid.height {
        for x in 0..grid.width {
            let idx = y * grid.width + x;
            let count = grid.data[idx];
            if count > 0 {
                // Normalize to 0.0-1.0 using a logarithmic scale to reduce the effect of outliers
                // intensity = ln(1 + count) / ln(1 + max_value)
                let denom = (max_value as f64).ln_1p();
                // Guard against degenerate cases (shouldn't happen because max_value >= 1), but be safe
                let intensity = if denom > 0.0 {
                    ((count as f64).ln_1p() / denom).clamp(0.0, 1.0)
                } else {
                    0.0
                };

                // Interpolate between min_color and max_color
                let r = (min_color.0 as f64 + intensity * (max_color.0 as f64 - min_color.0 as f64)) as u8;
                let g = (min_color.1 as f64 + intensity * (max_color.1 as f64 - min_color.1 as f64)) as u8;
                let b = (min_color.2 as f64 + intensity * (max_color.2 as f64 - min_color.2 as f64)) as u8;

                let color = RGBColor(r, g, b);
                root.draw_pixel((x as i32, y as i32), &color)?;
            }
        }
    }
    
    root.present()?;
    Ok(())
}
