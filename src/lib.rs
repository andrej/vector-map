/*
TODO:
[ ] Fix culling issues (lines connecting visible and culled points do not get
    drawn, which at times leads to bad shapes)
[ ] Fix off-by-one error on lat/lon lines
[ ] Properly keep track of elapsed time between frames instead of advancing by
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
const ANIMATE: bool = false;

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

struct World<'a, F, T1, T2>
where
    F: FnMut() -> T1,
    T1: Iterator<Item=GeoLine<'a, T2>>,
    T2: Iterator<Item=CoordGeo>
{
    window: web_sys::Window,
    context: web_sys::CanvasRenderingContext2d,
    width: f64,
    height: f64,
    render_closure: Option<Closure<dyn Fn()>>,
    cur_yaw: f64,
    cur_pitch: f64,
    cur_bounce_direction: BounceDirection,
    get_lines_it: F,
}

impl<F, T1, T2> World<'static, F, T1, T2>
where 
    F: (FnMut() -> T1) + 'static,
    T1: Iterator<Item=GeoLine<'static, T2>> + 'static,
    T2: Iterator<Item=CoordGeo> + 'static
{
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
        let clos: Closure<dyn Fn()> = Closure::new(move || { 
            let fut_this = Rc::clone(&clos_this);
            wasm_bindgen_futures::spawn_local(async move {
                let mut t = fut_this.lock().await;
                t.frame().await;
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
                sleep(10).await;
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
            let t = &mut *this.lock().await;
            t.cur_yaw = f64::to_radians((f64::to_degrees(t.cur_yaw) - 0.5) % 360.0);
            match t.cur_bounce_direction {
                BounceDirection::BounceUp(x) => {
                    t.cur_pitch = f64::to_radians((f64::to_degrees(t.cur_pitch) + x));
                    if t.cur_pitch > f64::to_radians(5.0) {
                        t.cur_bounce_direction = BounceDirection::BounceDown(-0.2);
                    }
                }
                BounceDirection::BounceDown(x) => {
                    t.cur_pitch = f64::to_radians((f64::to_degrees(t.cur_pitch) + x));
                    if t.cur_pitch < f64::to_radians(-0.0) {
                        t.cur_bounce_direction = BounceDirection::BounceUp(0.3);
                    }
                }
            }
            //t.cur_pitch = f64::to_radians((f64::to_degrees(t.cur_pitch) + 1.0) % 10.0);
            cur_yaw = t.cur_yaw;
        }
        // It is critical this sleep is outside of the above block; otherwise
        // the lock on `this` is apparently held for the entire time we are
        // sleeping
        //web_sys::console::log_1(&cur_yaw.to_string().into());
        sleep(20).await;
    }

    fn req_animation_frame(&self) {
        let clos = self.render_closure.as_ref().unwrap();
        self.window.request_animation_frame(clos.as_ref().unchecked_ref()).expect("Unable to set requestAnimationFrame");
    }

    fn map_and_filter_lines<'b>(
        &self, 
        proj_3d: &'b (impl Projection<CoordGeo, Coord3D>), 
        proj_2d: &'b OrthogonalProjection, 
        inp_lines: T1)
    -> impl Iterator<Item=Line<impl Iterator<Item=DrawOp>>> + 'b {

        let scale_fac = f64::min(self.width*0.8/2.0, self.height*0.8/2.0);
        let scale = Scale2D { x: scale_fac, y: scale_fac };
        let translate = Translate2D { x: self.width/2.0, y: self.height/2.0 };

        let cull = OrthogonalSphereCulling::new(
            CoordGeo { 
                latitude: f64::to_degrees(self.cur_pitch), 
                longitude: f64::to_degrees(self.cur_yaw)
            }
        );

        let lines = 
            inp_lines.into_iter()
            .map(move |x| { 
                (x.stroke_style, x.fill_style, 
                    project(proj_3d, x.points.into_iter()
                    .filter(
                        |y|{!cull.cull(y)})
                    .collect::<Vec<CoordGeo>>()
                    .into_iter()
                    )
                )
            })
            .map(move |(ss, fs, line)| { 
                //let (it1, it2): (impl Iterator<Item = Coord3D>, impl Iterator<Item = Coord3D>) = line.map(|x| {(x, x)} ).unzip();
                //it2.take();
                //let (it1, it2) : (impl Iterator<Item=Coord3D>, impl Iterator<Item=Coord3D>) = duplicate_iter(line);

                let mut culled_last = true;
                let mut draw_ops = Vec::<DrawOp>::new();
                for p_3d in line {
                    let cull_this = false; //proj_2d.cull(&p_3d, 0.0);
                    let p_2d = translate.transform(&scale.transform(&proj_2d.project(&p_3d)));
                    if !cull_this {
                        draw_ops.push(if culled_last {
                            DrawOp::MoveTo(p_2d)
                        } else {
                            DrawOp::LineTo(p_2d)
                        });
                    }
                    culled_last = cull_this
                }
                Line { 
                    points: draw_ops.into_iter(),
                    stroke_style: ss,
                    fill_style: fs
                }
            });

        lines
    }

    async fn frame(&mut self) {

        let proj_3d = SphereProjection;
        let proj_2d = proj_from_angles(self.cur_yaw, self.cur_pitch);

        self.context.clear_rect(0.0, 0.0, self.width, self.height);

        let lines = (self.get_lines_it)();

        let lines = self.map_and_filter_lines(&proj_3d, &proj_2d, lines);

        lines.enumerate().for_each(|(i, mut l)| { 
            self.context.set_line_width(1.0);
            self.context.set_stroke_style_str(format!("hsl({}deg 100% 50%", ((i as f64)/(18.0 as f64)*360.0).to_string()).as_str());
            draw_line(&self.context, &mut l) 
        } );

        if ANIMATE {
            self.req_animation_frame();
        }
    }
}

// credit: anon80458984
// https://users.rust-lang.org/t/async-sleep-in-rust-wasm32/78218/5
pub async fn sleep(delay: i32) {
    let mut cb = 
        | resolve: wasm_bindgen_futures::js_sys::Function,
          reject: wasm_bindgen_futures::js_sys::Function | 
        {
            web_sys::window()
                .unwrap()
                .set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, delay)
                .expect("unable to use JS setTimeout");
        };

    let p = wasm_bindgen_futures::js_sys::Promise::new(&mut cb);

    wasm_bindgen_futures::JsFuture::from(p).await.unwrap();
}

fn gen_lat_lon_lines(n_lines: i32, resolution: i32) -> (impl Iterator<Item=impl Iterator<Item=CoordGeo>>, impl Iterator<Item=impl Iterator<Item=CoordGeo>>) {
    let lat_lines = 
        (0..n_lines).map(move |lat_i| { 
            let lat: f64 = -90.0 + (lat_i as f64) / (n_lines as f64) * 180.0;
            (0..resolution).map(move |lon_i| { 
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
            (0..resolution).map(move |lat_i| { 
                let lat: f64 = -180.0 + (lat_i as f64) / (resolution as f64) * 360.0;
                CoordGeo { latitude: lat, longitude: lon } 
            })
        });
    (lat_lines, lon_lines)
}

fn proj_from_angles(yaw: f64, pitch: f64) -> OrthogonalProjection {
    // First, draw a unit length vector going into space from the origin, at
    // an angle of `pitch` measured from the XY plane. To get the Z component,
    // consider the vector the hypothenuse of a triangle, with the Z axis being
    // the adjacent side of a 90-pitch angle. cos=adj./hyp., so cos(180-pitch)
    // is the Z height. Since cos(90-pitch) = sin(pitch) we use that.
    // Now, to figure out the X and Y components, forget about the first 
    // triangle. Instead, draw a triangle where the hypothenuse lies on the 
    // XY plane, the unit vector is the adjacent side, and the opposite side
    // is prependicular to the unit vector, i.e. a right angle floating in
    // space, pointing down at the XY plane.
    // The length of the hyptohenuse will not be one (unless pitch is zero).
    // Draw two more triangles using the hypthenuse of the previous triangle as
    // its hypothenuse, and the adjacent sides being the X and Y axes, 
    // respectively, for each triangle.
    // The yaw angle is the angle between Y axis and the hypothenuse, so
    // the Y component is cos*hyp = ajd/hyp*hyp = adj. and the X component is
    // sin*hyp = opp/hyp*hyp = opp.
    // To get the value of hyp, use cos(yaw).
    let normal = Coord3D {
        x: f64::sin(yaw)*f64::cos(pitch),
        y: f64::cos(yaw)*f64::cos(pitch),
        z: f64::sin(pitch)
    };
    OrthogonalProjection::new_from_normal(normal)
}


#[wasm_bindgen]
pub fn main() {

    utils::set_panic_hook();

    //let shp = Shapefile::from_bytes(BOUNDARIES_SHP);
    let mut boundaries_shp_curs = std::io::Cursor::new(&BOUNDARIES_SHP[..]);
    let mut shp = shapefile::ShapeReader::new(boundaries_shp_curs).expect("unable to read shapefile");

    let document = web_sys::window().unwrap().document().unwrap();
    let canvas_elem = document.get_element_by_id("canvas").unwrap();
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

    let mut lines: Vec<GeoLine<<Vec<CoordGeo> as IntoIterator>::IntoIter>> = Vec::new();

    let n_latlon_lines = 18;
    let latlon_resolution = 36;
    let (lat_lines, lon_lines) = gen_lat_lon_lines(n_latlon_lines, latlon_resolution);
    let latlon_stroke_style = Box::new(String::from("#ccc"));
    let latlon_stroke_style: &String = Box::leak(latlon_stroke_style); // need to borrow IMMUTABLY here, because each closure below will capture this borrow, and we cannot have multiple mutable borrows
    lines.extend(
        lat_lines.map(|x| { GeoLine { points: x.collect::<Vec<CoordGeo>>().into_iter(), stroke_style: Option::Some(latlon_stroke_style), fill_style: Option::None } })
    );
    lines.extend(
        lon_lines.map(|x| { GeoLine { points: x.collect::<Vec<CoordGeo>>().into_iter(), stroke_style: Option::Some(latlon_stroke_style), fill_style: Option::None } })
    );

    // The following leaks are probably fine because we are essentially doing
    // the same thing in World::wrap for all other variables. These will all
    // have to live until the end of program execution
    let country_fill_style = Box::new(String::from("#019"));
    let country_fill_style = Box::leak(country_fill_style);
    let country_stroke_style = Box::new(String::from("#000"));
    let country_stroke_style = Box::leak(country_stroke_style);

    let res = 2.0; // degrees latitude/longitude difference to be included TODO: proper shape simplification
    for maybe_shp in shp.iter_shapes() {
        if let Ok(shp) = maybe_shp {
            if let shapefile::record::Shape::Polygon(polyshp) = shp {
                for ring in polyshp.rings() {
                    if let shapefile::record::polygon::PolygonRing::Outer(line) = ring {
                        let mut out_line = Vec::<CoordGeo>::new();
                        let mut last_p = CoordGeo { latitude: line[0].y, longitude: line[0].x };
                        out_line.push(last_p.clone());
                        for point in &line[1..] {
                            let this_p = CoordGeo{latitude: point.y, longitude: point.x};
                            if f64::abs(last_p.latitude - this_p.latitude) > res || f64::abs(last_p.longitude - this_p.longitude) > res {
                                last_p = this_p;
                                out_line.push(this_p.clone());
                            }
                        }
                        lines.push(GeoLine {
                            points: out_line.into_iter(),
                            stroke_style: Option::Some(country_stroke_style),
                            fill_style: Option::Some(country_fill_style)
                        })
                    }
                }
            }
        } else {
            break;
        }
    }

    let wrld = World { 
        window: web_sys::window().unwrap(),
        render_closure: None,
        context: context,
        width: canvas_width,
        height: canvas_height,
        cur_yaw: f64::to_radians(230.0),
        cur_pitch: f64::to_radians(5.0),
        cur_bounce_direction: BounceDirection::BounceUp(0.5),
        get_lines_it: move || { 
            lines.clone().into_iter() //.map(|x|{x.into_iter()})
        }
    };
    let moved_world = wrld.wrap();
    World::init(&moved_world);
    World::run(&moved_world);

    web_sys::console::log_2(&canvas_width.to_string().into(), &canvas_height.to_string().into());

}
