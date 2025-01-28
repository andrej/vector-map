/*
TODO:
[ ] Fix culling issues (lines connecting visible and culled points do not get
    drawn, which at times leads to bad shapes)
[x] Fix off-by-one error on lat/lon lines
[x] Properly keep track of elapsed time between frames instead of advancing by
    a fixed amount
[ ] Figure out why I'm having to do negative latitudes to get a right-side up
    (North pole up) globe currently. Probably a mistake in the projection
[ ] Add zoom/pan capabilities to projection
[ ] Add user interaction
[ ] Be smarter about shape simplification (currently only looking at +/- 1 deg
    difference)
*/
mod utils;
mod geometry;
mod drawing;

use wasm_bindgen::prelude::*;
use std::rc::Rc;
use futures::lock::Mutex;

use geometry::*;
use drawing::*;

const BOUNDARIES_SHP: &[u8; 161661560] = include_bytes!("geoBoundariesCGAZ_ADM0/geoBoundariesCGAZ_ADM0.shp");

// Disable req_animation_frame and update_state loop for debugging
const ANIMATE: bool = true;

enum BounceDirection {
    BounceUp(f64),
    BounceDown(f64)
}

fn duplicate_iter<'a, IterType, ElemType>(iter: IterType) -> (IterType, IterType)
where
    IterType: Iterator<Item=ElemType> + Default + Extend<ElemType>,
    ElemType: Clone
{
    iter.map(|x| { (x.clone(), x) }).unzip()
}

#[derive(Clone)]
struct GeoLine<'a, It>
where It: Iterator<Item=CoordGeo>
{
    points: It,
    stroke_style: Option<&'a String>,
    fill_style: Option<&'a String>
}

/// This is a small struct that contains all the necessary context for a to-be-
/// drawn frame. Values like the projection parameters and the stroke/fill
/// styles need to live all the way until the actual drawing takes place, since
/// values aren't calculated until the last moment. We will pass this struct
/// along with the actual iterator of drawing operations to keep these necessary
/// contextual bits of information alive.
struct World {
    yaw: f64,
    pitch: f64,
    country_outlines: Vec<Vec<CoordGeo>>,
    cur_bounce_direction: BounceDirection,
    proj_3d: SphereProjection,
    proj_2d: OrthogonalProjection,
    culler: OrthogonalSphereCulling,
    latlon_stroke_style: &'static str,
    country_outlines_stroke_style: &'static str,
    country_outlines_fill_style: &'static str,
}

impl World {
    fn new() -> Self {
        let yaw = f64::to_radians(230.0);
        let pitch = f64::to_radians(5.0);
        Self {
            yaw: yaw,
            pitch: pitch,
            cur_bounce_direction: BounceDirection::BounceUp(0.5),
            proj_3d: SphereProjection,
            proj_2d: OrthogonalProjection::new_from_angles(pitch, yaw),
            culler: OrthogonalSphereCulling::new(CoordGeo { latitude: pitch, longitude: yaw }),
            latlon_stroke_style: "#ccc",
            country_outlines_stroke_style: "#fff",
            country_outlines_fill_style: "#039",
            country_outlines: gen_country_outlines()
        }
    }
}

impl CanvasRenderLoopState for World
{
    fn update(&mut self, t_diff: f64) -> () {
        let t_diff_s = t_diff/1e3;
        let yaw_speed = 10.0; // [deg/s]
        let pitch_speed = 3.0; // [deg/s]
        let yaw_inc = yaw_speed * t_diff_s;
        self.yaw = f64::to_radians((f64::to_degrees(self.yaw) - yaw_inc) % 360.0);
        match self.cur_bounce_direction {
            BounceDirection::BounceUp(x) => {
                let pitch_inc = x * t_diff_s;
                self.pitch = f64::to_radians((f64::to_degrees(self.pitch) + pitch_inc));
                if self.pitch > f64::to_radians(5.0) {
                    self.cur_bounce_direction = BounceDirection::BounceDown(-pitch_speed);
                }
            }
            BounceDirection::BounceDown(x) => {
                let pitch_inc = x * t_diff_s;
                self.pitch = f64::to_radians((f64::to_degrees(self.pitch) + pitch_inc));
                if self.pitch < f64::to_radians(-0.0) {
                    self.cur_bounce_direction = BounceDirection::BounceUp(pitch_speed);
                }
            }
        }
        //self.cur_pitch = f64::to_radians((f64::to_degrees(self.cur_pitch) + 1.0) % 10.0);
        self.proj_3d = SphereProjection;
        self.proj_2d = OrthogonalProjection::new_from_angles(self.pitch, self.yaw);
        self.culler = OrthogonalSphereCulling::new(CoordGeo { latitude: self.pitch, longitude: self.yaw })
    }

    fn frame<'a, 'b>(&'a self) -> Option<Box<dyn Iterator<Item=DrawOp<Coord2D>> + 'b>>
    where 'a: 'b
    {
        let draw_ops = gen_frame_draw_ops(&self);
        if let Some(draw_ops) = draw_ops {
            Option::Some(Box::new(draw_ops))
        } else {
            Option::None
        }
    }
}


/// Return two iterators, one of latitude lines, and one of longitude lines
fn gen_lat_lon_lines(n_lines: i32, resolution: i32) -> (impl Iterator<Item=impl Iterator<Item=CoordGeo>>, impl Iterator<Item=impl Iterator<Item=CoordGeo>>) {
    let lat_lines = 
        (0..n_lines).map(move |lat_i| { 
            let lat: f64 = -90.0 + (lat_i as f64) / (n_lines as f64) * 180.0;
            (0..(resolution+1)).map(move |lon_i| { 
                let lon: f64 = -180.0 + (lon_i as f64) / (resolution as f64) * 360.0;
                //let lon: f64 = -90.0 + (lon_i as f64) / (resolution as f64) * 180.0;
                CoordGeo { 
                    latitude: f64::to_radians(lat), 
                    longitude: f64::to_radians(lon)
                } 
            })
        });
    // I'm assuming `move` in the closure here probably works because n_lines
    // implements Copy
    let lon_lines =
        (0..n_lines).map(move |lon_i| { 
            let lon: f64 = -90.0 + (lon_i as f64) / (n_lines as f64) * 180.0;
            (0..(resolution+1)).map(move |lat_i| { 
                let lat: f64 = -180.0 + (lat_i as f64) / (resolution as f64) * 360.0;
                CoordGeo { 
                    latitude: f64::to_radians(lat), 
                    longitude: f64::to_radians(lon)
                } 
            })
        });
    (lat_lines, lon_lines)
}

/// Create a vector of country outlines once; we can reuse this vector to
/// recreate iterators from it each time we need to draw them
fn gen_country_outlines() -> Vec<Vec<CoordGeo>> {
    let mut country_outlines: Vec<Vec<CoordGeo>> = Vec::new();
    let mut boundaries_shp_curs = std::io::Cursor::new(&BOUNDARIES_SHP[..]);
    let mut shp = shapefile::ShapeReader::new(boundaries_shp_curs).expect("unable to read shapefile");

    let res = f64::to_radians(1.0); // degrees latitude/longitude difference to be included TODO: proper shape simplification

    for maybe_shp in shp.iter_shapes() {
        if let Ok(shp) = maybe_shp {
            if let shapefile::record::Shape::Polygon(polyshp) = shp {
                for ring in polyshp.rings() {
                    if let shapefile::record::polygon::PolygonRing::Outer(line) = ring {
                        let mut out_line = Vec::<CoordGeo>::with_capacity(line.len());
                        let mut last_p = CoordGeo { 
                            latitude: f64::to_radians(line[0].y), 
                            longitude: f64::to_radians(line[0].x)
                        };
                        out_line.push(last_p.clone());
                        for point in &line[1..] {
                            let this_p = CoordGeo { 
                                latitude: f64::to_radians(point.y), 
                                longitude: f64::to_radians(point.x)
                            };
                            if f64::abs(last_p.latitude - this_p.latitude) > res || f64::abs(last_p.longitude - this_p.longitude) > res {
                                last_p = this_p;
                                out_line.push(this_p.clone());
                            }
                        }
                        country_outlines.push(out_line)
                    }
                }
            }
        } else {
            break;
        }
    }

    country_outlines
}

/// Helper function that takes an iterator of a line and projects its
/// coordinates
fn project_lines<'a>(
    lines: impl Iterator<Item=impl Iterator<Item=CoordGeo> + 'a>, 
    proj_3d: &'a impl Projection<CoordGeo, To=Coord3D>, 
    proj_2d: &'a impl Projection<Coord3D, To=Coord2D>,
) 
    -> impl Iterator<Item=impl Iterator<Item=Coord2D> + 'a>
{
    //let scale_fac = f64::min(self.width*0.8/2.0, self.height*0.8/2.0);
    //let scale = Scale2D { x: scale_fac, y: scale_fac };
    //let translate = Translate2D { x: self.width/2.0, y: self.height/2.0 };
    let scale_fac = f64::min(600.0*0.8/2.0, 400.0*0.8/2.0);

    lines.filter_map(move |line| {
        let mut projected = proj_2d.project(proj_3d.project(line));
        let first_point = projected.next();
        if let Some(first_point) = first_point {
            Option::Some(projected.chain(std::iter::once(first_point)).map(move |coord| {
                // TODO: move below transforms into World struct
                let scale = Scale2D { x: scale_fac, y: scale_fac };
                let translate = Translate2D { x: 600.0/2.0, y: 400.0/2.0 };
                translate.transform(&scale.transform(&coord))
            }))
        } else {
            Option::None
        }
    })
}

/// Create an iterator of drawing operations for each frame. The idea is that
/// we store as little as possible in memory -- only things that do not need
/// to be recomputed between frames are stored in memory. This means that the
/// latitude/longitude lines never get stored in memory; the iterator creates
/// them on the fly. The country outlines are stored in memory as absolute 
/// coordinates, but their projected coordinates change between each frame;
/// therefore, we map those vectors using projection iterators.
fn gen_frame_draw_ops<'a>(context: &'a World) -> Option<impl Iterator<Item=DrawOp<Coord2D>> + 'a> {
    // Latitude/longitude line iterator
    let n_latlon_lines = 18;
    let latlon_resolution = 36;
    let (lat_lines, lon_lines) = gen_lat_lon_lines(n_latlon_lines, latlon_resolution);

    // Country outline iterator
    let country_outlines = context.country_outlines.iter().map(|outline| { outline.iter().map(|point| { *point }) });
    let country_outlines = country_outlines;

    // Project all lines
    let lat_lines = project_lines(lat_lines, &context.proj_3d, &context.proj_2d);
    let lon_lines = project_lines(lon_lines, &context.proj_3d, &context.proj_2d);
    let country_outlines = project_lines(country_outlines, &context.proj_3d, &context.proj_2d);

    Some(std::iter::once(DrawOp::BeginPath)
        .chain(lat_lines.map(coord_iter_into_draw_ops).map(move |ops| { ops.chain(std::iter::once(DrawOp::Stroke(context.latlon_stroke_style.to_string()))) }).flatten())
        .chain(lon_lines.map(coord_iter_into_draw_ops).map(move |ops| { ops.chain(std::iter::once(DrawOp::Stroke(context.latlon_stroke_style.to_string()))) }).flatten())
        .chain(country_outlines
            .map(coord_iter_into_draw_ops)
            .map(move |ops| { 
                ops
                    .chain(std::iter::once(DrawOp::Stroke(context.country_outlines_stroke_style.to_string())))
                    .chain(std::iter::once(DrawOp::Fill(context.country_outlines_fill_style.to_string()))) })
            .flatten())
    )

}

#[wasm_bindgen]
pub fn main() {
    utils::set_panic_hook();

    //let shp = Shapefile::from_bytes(BOUNDARIES_SHP);

    // need to borrow IMMUTABLY here, because each closure below will capture this borrow, and we cannot have multiple mutable borrows
    // leak should be fine because these styles need to be live for the rest of the program, but FIXME: find more elegant solution
    // (i.e. pass ownership to the World struct and let it drop this value when it goes out of scope)

    let world = World::new();
    let draw_loop = CanvasRenderLoop::new(web_sys::window().unwrap(), "canvas", world).wrap();
    CanvasRenderLoop::<World>::init(&draw_loop);
    CanvasRenderLoop::<World>::run(&draw_loop);

}
