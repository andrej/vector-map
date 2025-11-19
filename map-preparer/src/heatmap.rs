use clap::Parser;
use plotters::prelude::*;
use std::path::PathBuf;

/// Available colormaps for heatmap visualization
#[derive(Debug, Clone, Copy)]
enum Colormap {
    Jet,       // Blue -> Cyan -> Green -> Yellow -> Red
    Viridis,   // Purple -> Blue -> Green -> Yellow
    Turbo,     // Blue -> Cyan -> Green -> Yellow -> Orange -> Red
}

impl Colormap {
    /// Map a normalized value [0.0, 1.0] to an RGB color
    fn map(&self, value: f64) -> (u8, u8, u8) {
        let v = value.clamp(0.0, 1.0);
        
        match self {
            Colormap::Jet => jet_colormap(v),
            Colormap::Viridis => viridis_colormap(v),
            Colormap::Turbo => turbo_colormap(v),
        }
    }
}

/// Jet colormap: Blue -> Cyan -> Green -> Yellow -> Red
fn jet_colormap(v: f64) -> (u8, u8, u8) {
    let r = if v < 0.375 {
        0.0
    } else if v < 0.625 {
        (v - 0.375) / 0.25
    } else if v < 0.875 {
        1.0
    } else {
        1.0 - (v - 0.875) / 0.125 * 0.5
    };
    
    let g = if v < 0.125 {
        0.0
    } else if v < 0.375 {
        (v - 0.125) / 0.25
    } else if v < 0.625 {
        1.0
    } else if v < 0.875 {
        1.0 - (v - 0.625) / 0.25
    } else {
        0.0
    };
    
    let b = if v < 0.125 {
        0.5 + v / 0.125 * 0.5
    } else if v < 0.375 {
        1.0
    } else if v < 0.625 {
        1.0 - (v - 0.375) / 0.25
    } else {
        0.0
    };
    
    ((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

/// Viridis colormap: perceptually uniform, colorblind-friendly
fn viridis_colormap(v: f64) -> (u8, u8, u8) {
    // Simplified 5-point interpolation based on viridis control points
    let points = [
        (0.267004, 0.004874, 0.329415), // Dark purple
        (0.282623, 0.140926, 0.457517), // Purple-blue
        (0.163625, 0.471133, 0.558148), // Blue-green
        (0.477504, 0.821444, 0.318195), // Yellow-green
        (0.993248, 0.906157, 0.143936), // Yellow
    ];
    
    let idx = v * (points.len() - 1) as f64;
    let i = idx.floor() as usize;
    let t = idx - i as f64;
    
    if i >= points.len() - 1 {
        let p = points[points.len() - 1];
        return ((p.0 * 255.0) as u8, (p.1 * 255.0) as u8, (p.2 * 255.0) as u8);
    }
    
    let (r0, g0, b0) = points[i];
    let (r1, g1, b1) = points[i + 1];
    
    let r = r0 + t * (r1 - r0);
    let g = g0 + t * (g1 - g0);
    let b = b0 + t * (b1 - b0);
    
    ((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

/// Turbo colormap: improved rainbow, more perceptually uniform than jet
fn turbo_colormap(v: f64) -> (u8, u8, u8) {
    // Simplified 6-point turbo approximation
    let points = [
        (0.18995, 0.07176, 0.23217), // Dark blue
        (0.11770, 0.56700, 0.75088), // Cyan
        (0.17205, 0.88797, 0.54362), // Green
        (0.89567, 0.99343, 0.29685), // Yellow
        (0.97809, 0.55414, 0.10540), // Orange
        (0.78801, 0.08080, 0.06051), // Red
    ];
    
    let idx = v * (points.len() - 1) as f64;
    let i = idx.floor() as usize;
    let t = idx - i as f64;
    
    if i >= points.len() - 1 {
        let p = points[points.len() - 1];
        return ((p.0 * 255.0) as u8, (p.1 * 255.0) as u8, (p.2 * 255.0) as u8);
    }
    
    let (r0, g0, b0) = points[i];
    let (r1, g1, b1) = points[i + 1];
    
    let r = r0 + t * (r1 - r0);
    let g = g0 + t * (g1 - g0);
    let b = b0 + t * (b1 - b0);
    
    ((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

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

    /// Color scheme to use (jet, viridis, turbo)
    #[arg(long, default_value = "jet")]
    colormap: String,

    /// Verbosity
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbosity: u8,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = ClArgs::parse();

    println!("Loading grid from: {}", args.input_grid.display());
    let grid = load_grid_from_file(&args.input_grid)?;
    println!("Loaded grid: {}x{} cells", grid.width, grid.height);

    println!("Generating heatmap with {} colormap...", args.colormap);
    let colormap = match args.colormap.to_lowercase().as_str() {
        "jet" => Colormap::Jet,
        "viridis" => Colormap::Viridis,
        "turbo" => Colormap::Turbo,
        _ => {
            eprintln!("Unknown colormap '{}', using 'jet'", args.colormap);
            Colormap::Jet
        }
    };
    
    save_heatmap_to_png(&grid, &args.output, colormap)?;
    
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

/// Paint a heatmap from a 2D grid of counts and save to PNG
fn save_heatmap_to_png(
    grid: &DynamicGrid,
    output_path: &std::path::Path,
    colormap: Colormap,
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

                // Map intensity to color using the selected colormap
                let (r, g, b) = colormap.map(intensity);

                let color = RGBColor(r, g, b);
                root.draw_pixel((x as i32, y as i32), &color)?;
            }
        }
    }
    
    root.present()?;
    Ok(())
}
