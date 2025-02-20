/*
TODO:
[x] Fix culling issues (lines connecting visible and culled points do not get
    drawn, which at times leads to bad shapes)
[x] Fix off-by-one error on lat/lon lines
[x] Properly keep track of elapsed time between frames instead of advancing by
    a fixed amount
[x] Figure out why I'm having to do negative latitudes to get a right-side up
    (North pole up) globe currently. Probably a mistake in the projection
    -> Canvas Y axis grows downwards
    -> Canvas X axis increases right, but positive longitudes mean going left
[ ] Add zoom capabilities to projection, including clipping to viewport
[x] Add user interaction
[ ] Improve panning
[ ] Be smarter about shape simplification (currently only looking at +/- 1 deg
    difference)
[ ] Fix projection issues when latitude > 45 deg
[ ] Fix hard-coded arc center/radius
*/
mod drawing;
mod geometry;
mod utils;

use futures::lock::Mutex;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

use drawing::*;
use geometry::*;

const BOUNDARIES_SHP: &[u8; 161661560] =
    include_bytes!("geoBoundariesCGAZ_ADM0/geoBoundariesCGAZ_ADM0.shp");

// Disable req_animation_frame and update_state loop for debugging
use std::f64::consts::PI;
const ANIMATE: bool = true;
const DEBUG_POINTS: [CoordGeo; 1] = [CoordGeo {
    latitude: 0.0,
    longitude: 0.0,
}];
const DEBUG_SHAPES: [[CoordGeo; 5]; 1] = [[
    CoordGeo {
        latitude: -0.3 * PI,
        longitude: 0.0,
    },
    CoordGeo {
        latitude: 0.3 * PI,
        longitude: 0.0,
    },
    CoordGeo {
        latitude: 0.3 * PI,
        longitude: 0.5 * PI,
    },
    CoordGeo {
        latitude: -0.3 * PI,
        longitude: 0.5 * PI,
    },
    CoordGeo {
        latitude: -0.3 * PI,
        longitude: 0.0,
    },
]];
const START_LON: f64 = 160.0 / 360.0 * (2.0 * PI); // 88.9
const START_LAT: f64 = 0.0 / 360.0 * (2.0 * PI);

enum BounceDirection {
    BounceUp(f64),
    BounceDown(f64),
}

fn duplicate_iter<'a, IterType, ElemType>(iter: IterType) -> (IterType, IterType)
where
    IterType: Iterator<Item = ElemType> + Default + Extend<ElemType>,
    ElemType: Clone,
{
    iter.map(|x| (x.clone(), x)).unzip()
}

#[derive(Clone)]
struct GeoLine<'a, It>
where
    It: Iterator<Item = CoordGeo>,
{
    points: It,
    stroke_style: Option<&'a String>,
    fill_style: Option<&'a String>,
}

enum MouseState {
    MouseDown,
    MouseUp,
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
    zoom: f64,
    country_outlines: Vec<Vec<CoordGeo>>,
    cur_bounce_direction: BounceDirection,
    proj_3d: SphereProjection,
    proj_2d: OrthogonalProjection,
    latlon_stroke_style: &'static str,
    country_outlines_stroke_style: &'static str,
    country_outlines_fill_style: &'static str,
    latlon_str: String,
    mouse_state: MouseState,
    yaw_speed: f64,
    pitch_speed: f64,
    needs_update: bool,
}

impl World {
    fn new() -> Self {
        let yaw = START_LON;
        let pitch = START_LAT;
        Self {
            yaw: yaw,
            pitch: pitch,
            zoom: 1.0,
            cur_bounce_direction: BounceDirection::BounceUp(0.5),
            proj_3d: SphereProjection,
            proj_2d: OrthogonalProjection::new_from_angles(pitch, yaw),
            latlon_stroke_style: "#ccc",
            country_outlines_stroke_style: "#fff",
            country_outlines_fill_style: "#039",
            country_outlines: gen_country_outlines(),
            latlon_str: String::new(),
            mouse_state: MouseState::MouseUp,
            yaw_speed: 0.1,
            pitch_speed: 3.0,
            needs_update: false,
        }
    }
}

impl CanvasRenderLoopState for World {
    fn update(&mut self, t_diff: f64) -> bool {
        let t_diff_s = t_diff / 1e3;
        let yaw_speed = self.yaw_speed; // [deg/s]
        let pitch_speed = self.pitch_speed; // [deg/s]
        if (yaw_speed == 0.0 || pitch_speed == 0.0) && !self.needs_update {
            return false;
        }

        if !self.needs_update {
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
        }

        self.proj_3d = SphereProjection;
        self.proj_2d = OrthogonalProjection::new_from_angles(self.pitch, self.yaw);

        self.latlon_str = String::from(format!(
            "lat: {:3.3} lon: {:3.3}",
            f64::to_degrees(self.pitch),
            f64::to_degrees(self.yaw)
        ));

        self.needs_update = false;

        return true;
    }

    fn frame<'a, 'b>(
        &'a self,
        canvas_width: f64,
        canvas_height: f64,
    ) -> Option<Box<dyn Iterator<Item = DrawOp<Coord2D>> + 'b>>
    where
        'a: 'b,
    {
        let draw_ops = gen_frame_draw_ops(&self, canvas_width, canvas_height);
        if let Some(draw_ops) = draw_ops {
            Option::Some(Box::new(draw_ops))
        } else {
            Option::None
        }
    }
}

/// Return two iterators, one of latitude lines, and one of longitude lines
fn gen_lat_lon_lines(
    n_lines: i32,
    resolution: i32,
) -> (
    impl Iterator<Item = impl Iterator<Item = CoordGeo>>,
    impl Iterator<Item = impl Iterator<Item = CoordGeo>>,
) {
    let lat_lines = (0..n_lines).map(move |lat_i| {
        let lat: f64 = -90.0 + (lat_i as f64) / (n_lines as f64) * 180.0;
        (0..=resolution).map(move |lon_i| {
            let lon: f64 = -180.0 + (lon_i as f64) / (resolution as f64) * 360.0;
            //let lon: f64 = -90.0 + (lon_i as f64) / (resolution as f64) * 180.0;
            CoordGeo {
                latitude: f64::to_radians(lat),
                longitude: f64::to_radians(lon),
            }
        })
    });
    // I'm assuming `move` in the closure here probably works because n_lines
    // implements Copy
    let lon_lines = (0..n_lines).map(move |lon_i| {
        let lon: f64 = -90.0 + (lon_i as f64) / (n_lines as f64) * 180.0;
        (0..=resolution).map(move |lat_i| {
            let lat: f64 = -180.0 + (lat_i as f64) / (resolution as f64) * 360.0;
            CoordGeo {
                latitude: f64::to_radians(lat),
                longitude: f64::to_radians(lon),
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
    let mut shp =
        shapefile::ShapeReader::new(boundaries_shp_curs).expect("unable to read shapefile");

    let res = f64::to_radians(1.0); // degrees latitude/longitude difference to be included TODO: proper shape simplification

    for maybe_shp in shp.iter_shapes() {
        if let Ok(shp) = maybe_shp {
            if let shapefile::record::Shape::Polygon(polyshp) = shp {
                for ring in polyshp.rings() {
                    if let shapefile::record::polygon::PolygonRing::Outer(line) = ring {
                        let mut out_line = Vec::<CoordGeo>::with_capacity(line.len());
                        let first_p = CoordGeo {
                            latitude: f64::to_radians(line[0].y),
                            longitude: f64::to_radians(line[0].x),
                        };
                        out_line.push(first_p.clone());
                        let mut last_p = first_p.clone();
                        for point in &line[1..] {
                            let this_p = CoordGeo {
                                latitude: f64::to_radians(point.y),
                                longitude: f64::to_radians(point.x),
                            };
                            if f64::abs(last_p.latitude - this_p.latitude) > res
                                || f64::abs(last_p.longitude - this_p.longitude) > res
                            {
                                last_p = this_p;
                                out_line.push(this_p.clone());
                            }
                        }
                        // close the path by pushing a copy of the first point
                        // at the end; this simplifies our culling logic
                        out_line.push(first_p.clone());
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
    lines: impl Iterator<Item = impl Iterator<Item = CoordGeo> + 'a> + 'a,
    context: &'a World,
    start_op: DrawOp<'a, Coord2D>,
    end_op: DrawOp<'a, Coord2D>,
    draw_arc: bool,
    canvas_width: f64,
    canvas_height: f64,
) -> impl Iterator<Item = DrawOp<'a, Coord2D>> + 'a {
    // TODO: move below transforms into World struct
    let scale_fac = context.zoom * f64::min(canvas_width * 0.8 / 2.0, canvas_height * 0.8 / 2.0);
    let scale = Scale2D {
        x: scale_fac,
        y: scale_fac,
    };
    let translate = Translate2D {
        x: canvas_width / 2.0,
        y: canvas_height / 2.0,
    };

    lines
        .filter_map(move |line| {
            let projected = into_clamped_iter(
                line.map(move |point| context.proj_2d.project(&context.proj_3d.project(&point))),
            );
            // TODO: move arc_center, arc_radius to be applied when we apply the scale and translate transforms below;
            // the clampedArcIterator really should just return arcs with center at 0.0 and radius 1 or something like that
            let mut draw_op_gen: Box<dyn Iterator<Item = DrawOp<Coord2D>>> = if context.zoom < 2.0 {
                Box::new(ClampedArcIterator::new(
                    ClampedIterator::new(
                        projected,
                        |p| p.x <= 0.0,
                        geometry::get_viewport_intersection_point,
                    ),
                    draw_arc,
                    Coord2D {
                        x: canvas_width / 2.0,
                        y: canvas_height / 2.0,
                    },
                    scale_fac,
                ))
            } else {
                let limit = 1.0 / context.zoom;
                Box::new(ClampedLineIterator::new(
                    ClampedIterator::new(
                        projected,
                        move |p| {
                            p.x <= 0.0 && f64::abs(p.y) < limit && f64::abs(p.z) < limit
                        },
                        move |visible, invisible| {
                            let dx = invisible.x - visible.x;
                            let dy = invisible.y - visible.y;
                            let dz = invisible.z - visible.z;
                            // TODO: handle case where dy or dz are zero to avoid division by zero
                            let dist = if f64::abs(invisible.y) >= limit {
                                (f64::signum(invisible.y) * limit - visible.y) / dy
                            } else {
                                (f64::signum(invisible.z) * limit - visible.z) / dz
                            };
                            Coord3D {
                                x: visible.x + dist * dx,
                                y: visible.y + dist * dy,
                                z: visible.z + dist * dz
                            }
                        },
                    )
                ))
            };
            let first_point = draw_op_gen.next();
            if let Some(mut first_point) = first_point {
                let first_coord = first_point.get_coord();
                if let Some(&first_coord) = first_coord {
                    first_point = DrawOp::MoveTo(first_coord);
                }
                let ret = std::iter::once(first_point)
                    .chain(draw_op_gen)
                    .map(move |mut op| {
                        let maybe_coord = op.get_coord();
                        if let Some(coord) = maybe_coord {
                            op.set_coord(translate.transform(&scale.transform(coord)));
                        }
                        op
                    });
                Option::Some(ret)
            } else {
                Option::None
            }
        })
        .map(move |line_ops| {
            std::iter::once(start_op.clone())
                .chain(line_ops)
                .chain(std::iter::once(end_op.clone()))
        })
        .flatten()
}

/// Create an iterator of drawing operations for each frame. The idea is that
/// we store as little as possible in memory -- only things that do not need
/// to be recomputed between frames are stored in memory. This means that the
/// latitude/longitude lines never get stored in memory; the iterator creates
/// them on the fly. The country outlines are stored in memory as absolute
/// coordinates, but their projected coordinates change between each frame;
/// therefore, we map those vectors using projection iterators.
fn gen_frame_draw_ops<'a>(
    context: &'a World,
    canvas_width: f64,
    canvas_height: f64,
) -> Option<impl Iterator<Item = DrawOp<Coord2D>> + 'a> {
    // Latitude/longitude line iterator
    let n_latlon_lines = 18;
    let latlon_resolution = 72;
    let (lat_lines, lon_lines) = gen_lat_lon_lines(n_latlon_lines, latlon_resolution);

    // Country outline iterator
    let country_outlines = context
        .country_outlines
        .iter()
        .map(|outline| outline.iter().map(|point| *point));
    let country_outlines = country_outlines;

    // Debug points
    //let debug_points = project_lines(std::iter::once(DEBUG_POINTS.iter().cloned()), &context.proj_3d, &context.proj_2d, DrawOp::BeginPath, DrawOp::Stroke(context.latlon_stroke_style.to_string()), false, canvas_width, canvas_height);
    let debug_shapes = project_lines(
        DEBUG_SHAPES.iter().map(|s| s.iter().cloned()),
        context,
        DrawOp::BeginPath,
        DrawOp::Fill(context.country_outlines_fill_style.to_string()),
        true,
        canvas_width,
        canvas_height,
    );

    // Project all lines
    let lat_lines = project_lines(
        lat_lines,
        context,
        DrawOp::BeginPath,
        DrawOp::Stroke(context.latlon_stroke_style.to_string()),
        false,
        canvas_width,
        canvas_height,
    );
    let lon_lines = project_lines(
        lon_lines,
        context,
        DrawOp::BeginPath,
        DrawOp::Stroke(context.latlon_stroke_style.to_string()),
        false,
        canvas_width,
        canvas_height,
    );
    let country_outlines = project_lines(
        country_outlines,
        context,
        DrawOp::BeginPath,
        DrawOp::Fill(context.country_outlines_fill_style.to_string()),
        true,
        canvas_width,
        canvas_height,
    );

    Some(
        std::iter::once(DrawOp::BeginPath)
            .chain(lat_lines)
            .chain(lon_lines)
            .chain(country_outlines)
            //.chain(debug_points.filter_map(|op| if let Some(&coord) = op.get_coord() { Option::Some(DrawOp::BigRedCircle(coord)) } else { Option::None }))
            //.chain(debug_shapes)
            .chain(std::iter::once(DrawOp::Text(
                Coord2D { x: 10.0, y: 350.0 },
                &context.latlon_str,
            ))),
    )
}

fn add_leaky_event_listener(event: &str, cb: impl Fn(web_sys::Event) -> () + 'static) {
    let cb = Box::new(wasm_bindgen::closure::Closure::<dyn Fn(web_sys::Event) -> ()>::new(cb));
    web_sys::window()
        .unwrap()
        .add_event_listener_with_callback(event, Box::leak(cb).as_ref().unchecked_ref());
}

fn add_leaky_event_listener_with_state(
    state: &Rc<Mutex<CanvasRenderLoop<World>>>,
    event: &str,
    cb: impl Fn(&mut World, web_sys::Event) -> () + Clone + 'static,
) {
    let wrapped_state = Box::leak(Box::new(state.clone()));
    let wrapped_cb = Box::new(
        wasm_bindgen::closure::Closure::<dyn Fn(web_sys::Event) -> ()>::new(
            move |e: web_sys::Event| {
                let my_state = Rc::clone(wrapped_state);
                let cb = cb.clone();
                wasm_bindgen_futures::spawn_local(async move {
                    let mut my_state_unlocked = my_state.lock().await;
                    cb(&mut my_state_unlocked.state, e);
                })
            },
        ),
    );
    web_sys::window()
        .unwrap()
        .add_event_listener_with_callback(event, Box::leak(wrapped_cb).as_ref().unchecked_ref());
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

    add_leaky_event_listener_with_state(
        &draw_loop,
        "mousedown",
        move |w: &mut World, e: web_sys::Event| {
            w.mouse_state = MouseState::MouseDown;
            w.yaw_speed = 0.0;
            w.pitch_speed = 0.0;
        },
    );
    add_leaky_event_listener_with_state(
        &draw_loop,
        "mousemove",
        move |w: &mut World, e: web_sys::Event| {
            if let MouseState::MouseDown = w.mouse_state {
                let me: web_sys::MouseEvent = e.unchecked_into();
                let dx = me.movement_x();
                let dy = me.movement_y();
                w.yaw -= (-dx as f64) * PI / 400.0;
                w.pitch -= (dy as f64) * PI / 400.0;
                w.needs_update = true;
            }
        },
    );
    add_leaky_event_listener_with_state(
        &draw_loop,
        "mouseup",
        move |w: &mut World, e: web_sys::Event| {
            w.mouse_state = MouseState::MouseUp;
        },
    );
    add_leaky_event_listener_with_state(
        &draw_loop,
        "wheel",
        move |w: &mut World, e: web_sys::Event| {
            let we: web_sys::WheelEvent = e.unchecked_into();
            w.zoom += f64::signum(we.delta_y()) * 0.1;
            w.needs_update = true;
        },
    );
}
