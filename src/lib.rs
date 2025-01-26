/*
TODO:
[ ] Fix culling issues (lines connecting visible and culled points do not get
    drawn, which at times leads to bad shapes)
[ ] Fix off-by-one error on lat/lon lines
[x] Properly keep track of elapsed time between frames instead of advancing by
    a fixed amount
[ ] Figure out why I'm having to do negative latitudes to get a right-side up
    (North pole up) globe currently. Probably a mistake in the projection
[ ] Add zoom/pan capabilities to projection
[ ] Add user interaction
[ ] Be smarter about shape simplification (currently only looking at +/- 1 deg
    difference)
*/
mod copy;
mod utils;
mod geometry;
mod drawing;

use wasm_bindgen::prelude::*;
use std::rc::Rc;
use std::cell::RefCell;
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

struct World
{
    window: web_sys::Window,
    performance: web_sys::Performance,
    context: web_sys::CanvasRenderingContext2d,
    width: f64,
    height: f64,
    render_closure: Option<Closure<dyn Fn(f64)>>,
    cur_yaw: f64,
    cur_pitch: f64,
    cur_bounce_direction: BounceDirection,
    last_state_update_t: f64,
    state: Option<FrameDrawOpsContext>
}

impl World
{
    fn new(window: web_sys::Window, canvas_id: &str, state: Option<FrameDrawOpsContext>) -> Self {
        let document = window.document().unwrap();
        let canvas_elem = document.get_element_by_id(canvas_id).unwrap();
        let canvas: &web_sys::HtmlCanvasElement = canvas_elem
            .dyn_ref::<web_sys::HtmlCanvasElement>()
            .expect("element with ID #canvas should be a <canvas> in index.html");
        let context = canvas
            .get_context("2d")
            .expect("browser should provide a 2d context")
            .unwrap()
            .dyn_into::<web_sys::CanvasRenderingContext2d>()
            .expect("should be a CanvasRenderingContext2D object");

        let get_u32_attr_with_default = |el: &web_sys::Element, attr: &str, default: u32| -> u32 {
            el.get_attribute(attr).and_then(|x| x.parse::<u32>().ok()).unwrap_or(default)
        };
        let canvas_width = get_u32_attr_with_default(&canvas_elem, "width", 400) as f64;
        let canvas_height = get_u32_attr_with_default(&canvas_elem, "height", 400) as f64;
        Self {
            window: web_sys::window().unwrap(),
            performance: window.performance().expect("performance should be available"),
            render_closure: None,
            context: context,
            width: canvas_width,
            height: canvas_height,
            cur_yaw: f64::to_radians(230.0),
            cur_pitch: f64::to_radians(5.0),
            cur_bounce_direction: BounceDirection::BounceUp(0.5),
            last_state_update_t: 0.0,
            state: state
        }
    }

    // Using a RefCell and Rc is almost unavoidable here:
    // The request_animation_frame callback needs to borrow World so it can see
    // what it should render. It needs access to World.
    // However, that callback may get called at any time in the future by the
    // browser; hence the requirement of web_sys for the callback to have a 
    // 'static lifetime.

    // At the same time, if we want to be able to modify the world to render
    // at all, we are going to need to take mutable references to it at some
    // point. Thus, to enable this, we must use dynamic borrow checking 
    // implemented in RefCell. Rc allows us to hold multiple references to this
    // RefCell using reference counting.

    // An alternative approach would be to create a new closure for each
    // rendering frame, then call any functionality needed to update the state
    // of the world from within the rendering frame closure. The last thing the 
    // closure would do is move `self` into a new, next closure for the next 
    // rendering frame. This would preserve borrow semantics, since no borrows
    // would overlap (the state updating function would require a mutable ref
    // to self, but it would return and then let the rendering function inspect
    // the state of the world after it is done).
    fn wrap(self) -> Rc<Mutex<Self>> {
        let this = Rc::new(Mutex::new(self));
        this
    }

    fn init(this: &Rc<Mutex<Self>>) -> () {
        let clos_this = Rc::clone(&this);
        let clos: Closure<dyn Fn(f64)> = Closure::new(move |time: f64| { 
            let fut_this = Rc::clone(&clos_this);
            wasm_bindgen_futures::spawn_local(async move {
                let mut t = fut_this.lock().await;
                t.frame(time).await;
            });
        });
        //(*this).as_ref().borrow_mut().render_closure = Some(clos);

        let another_this = Rc::clone(&this);
        wasm_bindgen_futures::spawn_local(async move {
            another_this.lock().await.render_closure = Some(clos);
        });
    }

    fn run(this: &Rc<Mutex<Self>>) -> () {
        // Kick off rendering loop; this is the first request_animation frame,
        // then subsequent calls to the same function are made form within 
        // frame() to keep it going. 
        // Since frame is async we can preempt it with state updates
        let this1 = Rc::clone(&this);
        wasm_bindgen_futures::spawn_local(async move {
            loop {
                if this1.lock().await.render_closure.is_some() {
                    break;
                }
                utils::sleep(10).await;
            }
            this1.lock().await.req_animation_frame();
        });
        let this2 = Rc::clone(&this);
        if ANIMATE {
            wasm_bindgen_futures::spawn_local(async move {
                loop {
                    World::update_state(&this2).await;
                }
            });
        }
    }

    async fn update_state(this: &Rc<Mutex<Self>>) -> () {
        let mut cur_yaw: f64;
        {
            let this = &mut *this.lock().await;
            let t_cur = this.performance.now();
            let t_diff = t_cur - this.last_state_update_t;
            this.last_state_update_t = t_cur;
            let t_diff_s = t_diff/1e3;
            let yaw_speed = 10.0; // [deg/s]
            let pitch_speed = 3.0; // [deg/s]
            let yaw_inc = yaw_speed * t_diff_s;
            this.cur_yaw = f64::to_radians((f64::to_degrees(this.cur_yaw) - yaw_inc) % 360.0);
            match this.cur_bounce_direction {
                BounceDirection::BounceUp(x) => {
                    let pitch_inc = x * t_diff_s;
                    this.cur_pitch = f64::to_radians((f64::to_degrees(this.cur_pitch) + pitch_inc));
                    if this.cur_pitch > f64::to_radians(5.0) {
                        this.cur_bounce_direction = BounceDirection::BounceDown(-pitch_speed);
                    }
                }
                BounceDirection::BounceDown(x) => {
                    let pitch_inc = x * t_diff_s;
                    this.cur_pitch = f64::to_radians((f64::to_degrees(this.cur_pitch) + pitch_inc));
                    if this.cur_pitch < f64::to_radians(-0.0) {
                        this.cur_bounce_direction = BounceDirection::BounceUp(pitch_speed);
                    }
                }
            }
            //this.cur_pitch = f64::to_radians((f64::to_degrees(this.cur_pitch) + 1.0) % 10.0);
            cur_yaw = this.cur_yaw;
            update_state(&mut this.state, this.cur_yaw, this.cur_pitch);
        }
        // It is critical this sleep is outside of the above block; otherwise
        // the lock on `this` is apparently held for the entire time we are
        // sleeping
        utils::sleep(20).await;
    }

    fn req_animation_frame(&self) {
        let clos = self.render_closure.as_ref().unwrap();
        self.window.request_animation_frame(clos.as_ref().unchecked_ref()).expect("Unable to set requestAnimationFrame");
    }

    async fn frame(&mut self, t: f64) {
        let tstart = self.performance.now();

        self.context.clear_rect(0.0, 0.0, self.width, self.height);
        let draw_ops = gen_frame_draw_ops(&self.state);
        if let Some(draw_ops) = draw_ops {
            draw_ops.for_each(|op| { 
                op.draw(&self.context);
            });
        }

        let tend = self.performance.now();
        let tdiff = tend-tstart;
        self.context.fill_text(&format!("{:3.0} ms to render this frame", tdiff), 20.0, 20.0);
        self.context.fill_text(&format!("{:3.0} FPS", 1.0/(tdiff/1e3)), 20.0, 40.0);

        if ANIMATE {
            self.req_animation_frame();
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
                CoordGeo { latitude: lat, longitude: lon } 
            })
        });
    // I'm assuming `move` in the closure here probably works because n_lines
    // implements Copy
    let lon_lines =
        (0..n_lines).map(move |lon_i| { 
            let lon: f64 = -90.0 + (lon_i as f64) / (n_lines as f64) * 180.0;
            (0..(resolution+1)).map(move |lat_i| { 
                let lat: f64 = -180.0 + (lat_i as f64) / (resolution as f64) * 360.0;
                CoordGeo { latitude: lat, longitude: lon } 
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

    let res = 1.0; // degrees latitude/longitude difference to be included TODO: proper shape simplification

    for maybe_shp in shp.iter_shapes() {
        if let Ok(shp) = maybe_shp {
            if let shapefile::record::Shape::Polygon(polyshp) = shp {
                for ring in polyshp.rings() {
                    if let shapefile::record::polygon::PolygonRing::Outer(line) = ring {
                        let mut out_line = Vec::<CoordGeo>::with_capacity(line.len());
                        let mut last_p = CoordGeo { latitude: line[0].y, longitude: line[0].x };
                        out_line.push(last_p.clone());
                        for point in &line[1..] {
                            let this_p = CoordGeo{latitude: point.y, longitude: point.x};
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
    proj_2d: &'a impl Projection<Coord3D, To=Coord2D>
) 
    -> impl Iterator<Item=impl Iterator<Item=Coord2D> + 'a>
{
    //let scale_fac = f64::min(self.width*0.8/2.0, self.height*0.8/2.0);
    //let scale = Scale2D { x: scale_fac, y: scale_fac };
    //let translate = Translate2D { x: self.width/2.0, y: self.height/2.0 };
    let scale_fac = f64::min(600.0*0.8/2.0, 400.0*0.8/2.0);
    let scale = Scale2D { x: scale_fac, y: scale_fac };
    let translate = Translate2D { x: 600.0/2.0, y: 400.0/2.0 };

    lines.map(move |line| {
        line.map(move |coord| {
            let projected = proj_2d.project(&proj_3d.project(&coord));
            let translated = translate.transform(&scale.transform(&projected));
            translated
        })
    })
}

/// This is a small struct that contains all the necessary context for a to-be-
/// drawn frame. Values like the projection parameters and the stroke/fill
/// styles need to live all the way until the actual drawing takes place, since
/// values aren't calculated until the last moment. We will pass this struct
/// along with the actual iterator of drawing operations to keep these necessary
/// contextual bits of information alive.
struct FrameDrawOpsContext {
    yaw: f64,
    pitch: f64,
    proj_3d: SphereProjection,
    proj_2d: OrthogonalProjection,
    latlon_stroke_style: &'static str,
    country_outlines_stroke_style: &'static str,
    country_outlines_fill_style: &'static str,
    country_outlines: Vec<Vec<CoordGeo>>
}

fn update_state(context: &mut Option<FrameDrawOpsContext>, yaw: f64, pitch: f64) -> () {
    if let Some(ctx) = context {
        ctx.yaw = yaw;
        ctx.pitch = pitch;
        ctx.proj_3d = SphereProjection;
        ctx.proj_2d = OrthogonalProjection::new_from_angles(yaw, pitch);
    } else {
        *context = Some(FrameDrawOpsContext {
            yaw: yaw,
            pitch: pitch,
            proj_3d: SphereProjection,
            proj_2d: OrthogonalProjection::new_from_angles(yaw, pitch),
            latlon_stroke_style: "#ccc",
            country_outlines_stroke_style: "#fff",
            country_outlines_fill_style: "#039",
            country_outlines: gen_country_outlines()
        });
    }
}

/// Create an iterator of drawing operations for each frame. The idea is that
/// we store as little as possible in memory -- only things that do not need
/// to be recomputed between frames are stored in memory. This means that the
/// latitude/longitude lines never get stored in memory; the iterator creates
/// them on the fly. The country outlines are stored in memory as absolute 
/// coordinates, but their projected coordinates change between each frame;
/// therefore, we map those vectors using projection iterators.
fn gen_frame_draw_ops<'a>(context: &'a Option<FrameDrawOpsContext>) -> Option<impl Iterator<Item=DrawOp<Coord2D>> + 'a> {
    if context.is_none() {
        return Option::None
    }

    let context = context.as_ref().unwrap();

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

    let moved_world = World::new(
        web_sys::window().expect("should have a window"),
        &"canvas",
        Option::<FrameDrawOpsContext>::None,
    ).wrap();
    World::init(&moved_world);
    World::run(&moved_world);

}
