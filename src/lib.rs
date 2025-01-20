mod utils;

use wasm_bindgen::prelude::*;
use std::fmt;
use std::iter::Zip;
use std::ops;

use std::rc::Rc;
use std::cell::RefCell;
use futures::lock::Mutex;

const BOUNDARIES_SHP: &[u8; 161661560] = include_bytes!("geoBoundariesCGAZ_ADM0/geoBoundariesCGAZ_ADM0.shp");

//struct Shapefile;
//
//impl Shapefile {
//    fn from_bytes(bytes: &[u8]) -> Self {
//        //web_sys::console::log_3(&bytes[0].to_string().into(), &bytes[1].to_string().into(), &bytes[3].to_string().into());
//        web_sys::console::log_1(&format!("{0:x} {1:x} {2:x} {3:x}", bytes[0], bytes[1], bytes[2], bytes[3]).into());
//        //web_sys::console::log_3(&bytes[0].to_string().into(), &bytes[1].to_string().into(), &bytes[3].to_string().into());
//        assert!(bytes[0] == 0x00);
//        assert!(bytes[1] == 0x00);
//        assert!(bytes[2] == 0x27);
//        assert!(bytes[3] == 0x0a);
//        Shapefile {}
//    }
//}
//


#[derive(Copy, Clone)]
struct CoordGeo {
    latitude: f64,
    longitude: f64
}

impl fmt::Display for CoordGeo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{ latitude: {}, longitude: {} }}", self.latitude, self.longitude)
    }
}

#[derive(Copy, Clone)]
struct Coord3D {
    x: f64,
    y: f64,
    z: f64
}

impl ops::Sub<&Coord3D> for &Coord3D {
    type Output = Coord3D;

    fn sub(self, rhs: &Coord3D) -> Coord3D {
        Coord3D {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z
        }
    }
}

impl ops::Mul<f64> for &Coord3D {
    type Output = Coord3D;

    fn mul(self, rhs: f64) -> Coord3D {
        Coord3D {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs
        }
    }
}

impl fmt::Display for Coord3D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{ x: {}, y: {}, z: {} }}", self.x, self.y, self.z)
    }
}

#[derive(Copy, Clone)]
struct Coord2D {
    x: f64,
    y: f64
}

impl fmt::Display for Coord2D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{ x: {}, y: {}}}", self.x, self.y)
    }
}

trait Projection<From, To> {
    fn project(&self, input: &From) -> To;
}

struct SphereProjection;

impl Projection<CoordGeo, Coord3D> for SphereProjection {
    fn project(&self, input: &CoordGeo) -> Coord3D {
        let r = 1.0;
        let [lat, lon] = [input.latitude, input.longitude].map(f64::to_radians);
        Coord3D { x: r * f64::cos(-lat) * f64::cos(lon),
                  y: r * f64::cos(-lat) * f64::sin(lon),
                  z: r * f64::sin(-lat) }
    }
}

fn dot_product(a: &Coord3D, b: &Coord3D) -> f64 {
    a.x*b.x + a.y*b.y + a.z*b.z
}

fn norm(a: &Coord3D) -> f64 {
    f64::sqrt(a.x*a.x + a.y*a.y + a.z*a.z)
}

fn normalize(a: &Coord3D) -> Coord3D {
    let norm = norm(a);
    Coord3D { x: a.x/norm, y: a.y/norm, z: a.z/norm }
}

fn cross_product(a: &Coord3D, b: &Coord3D) -> Coord3D {
    Coord3D {
        x: a.y * b.z - a.z * b.y,
        y: a.z * b.x - a.x * b.z,
        z: a.x * b.y - a.y * b.x
    }
}

fn project_onto_plane(plane_normal: &Coord3D, vector: &Coord3D) -> Coord3D {
    // Intuition: dot product `vector`*`plane_normal` gives the component of 
    // `vector` in the `plane_normal` direction (i.e. if they are parallel,
    // this gives all of `vector`).
    // By subtracting everything in this direction, which is the normal of the
    // plane, we subtract the direction that would take a point "off" the plane.
    let plane_normal = normalize(plane_normal);
    vector - &(&plane_normal * dot_product(vector, &plane_normal))
}
struct OrthogonalProjection {
    x_axis: Coord3D,
    y_axis: Coord3D
}

impl OrthogonalProjection {
    fn new(x_axis: Coord3D, y_axis: Coord3D) -> OrthogonalProjection {
        OrthogonalProjection {
            x_axis: normalize(&x_axis),
            y_axis: normalize(&y_axis)
        }
    }

    fn new_from_normal(normal: Coord3D) -> OrthogonalProjection {
        // Choose a vector that is not parallel to normal. We will project it
        // onto the plane next.
        // Our goal is to make sure the projection plane is "level" w.r.t. the
        // XY plane. That is, one axis of the projection plane should be
        // parallel to the XY plane, and the other perpendicular to that. This
        // way, the projection plane is not askew w.r.t. the XY plane.
        // This works in all but one case: If we are looking straight down at
        // the scene (i.e. normal vector is equal to the Z basis vector), the
        // projection of the Z basis vector onto the plane will be the zero
        // vector. This also makes sense in that it is unclear what "askew"
        // w.r.t. to the XZ plane would mean if our projection plane is exactly
        // the XZ plane. Another way of thinking of it: If we look straight down
        // at the globe form the North pole, there is no right or wrong
        // rotation.
        // We arbitrarily choose the Y basis vector in that special case.
        let v = if normal.z != 0.0 && normal.x == 0.0 && normal.y == 0.0 {
            // Special case: Looking straight down at XZ plane.
            Coord3D {
                x: 0.0,
                y: 1.0,
                z: 0.0
            }
        } else {
            Coord3D {
                x: 0.0,
                y: 0.0,
                z: 1.0
            }
        };
        // Project it onto the plane normal to `normal`
        let y_axis = normalize(&project_onto_plane(&normal, &v));
        // Find a vector orthogonal to both x_axis and `normal`.
        // By making it orthogonal to `normal` it is guaranteed to lie in the plane.
        let x_axis = normalize(&cross_product(&normal, &y_axis));
        OrthogonalProjection {
            x_axis: x_axis,
            y_axis: y_axis
        }
    }
}

impl Projection<Coord3D, Coord2D> for OrthogonalProjection {
    fn project(&self, input: &Coord3D) -> Coord2D {
        Coord2D { x:  dot_product(input, &self.x_axis), y: dot_product(input, &self.y_axis) }
    }
}

impl OrthogonalProjection {
    fn cull(&self, input: &Coord3D, distance: f64) -> bool {
        let normal = cross_product(&self.x_axis, &self.y_axis); 
            // TODO: optimize this so we don't recaculate normal for every. single. coordinate.
        //web_sys::console::log_1(&dot_product(input, &normal).to_string().into());
        dot_product(input, &normal) < distance 
    }
}

// TODO: Refactor transforms into matrices; allows for multiplying matrices
// then applying single transform all at once

trait Transform<CoordType> {
    fn transform(&self, input: &CoordType) -> CoordType;
}

#[derive(Copy, Clone)]
struct Translate2D {
    x: f64,
    y: f64   
}

impl Transform<Coord2D> for Translate2D {
    fn transform(&self, input: &Coord2D) -> Coord2D {
        Coord2D { x: input.x + self.x, y: input.y + self.y }
    }
}

#[derive(Copy, Clone)]
struct Scale2D {
    x: f64,
    y: f64
}

impl Transform<Coord2D> for Scale2D {
    fn transform(&self, input: &Coord2D) -> Coord2D {
        Coord2D { x: input.x * self.x, y: input.y * self.y }
    }
}


fn project<'a, A, B>(proj: &'a impl Projection<A, B>, line: impl Iterator<Item=A> + 'a) -> impl Iterator<Item=B> + 'a { 
    line.map(move |point | { 
        proj.project(&point)
    })
}

// The reason using a generic (e.g. I_out where I_out: Iterator<Item=Coord2d>)
// for the return type does not work here is that the type that the function
// returns cannot be named, because it contains closures. Closures have a type
// that cannot be named. When using generics, the caller would have to specify
// the concrete return type at the call site; something that is not possible
// with types containing closuers. Using `impl` allows the compiler to infer 
// that type. `impl` does not mean dynamic dispatch; the compiler will still
// monomorphize the code at each call site, that is, specialzie the function
// for each type that it returns, just like it would for a generic.
// The reason generics work for the *input* parameters is that the caller is
// responsible for naming all types: The caller knows the (opaque) types for
// the iterators it passes in, becaues it has created the closures. However,
// this function then creates a closure for the returned iterator as well; the
// caller cannot know or name the type of this closure, since it is created
// in this function, not the caller.
fn project_generic<'a, A, B, I, P>(proj: &'a P, line: I) -> impl Iterator<Item=B> + 'a
where 
    I: Iterator<Item=A> + 'a,
    P: Projection<A, B> + 'a
{ 
    line.map(move |point| { proj.project(&point) })
}

enum DrawOp {
    MoveTo(Coord2D),
    LineTo(Coord2D)
}

fn draw_line(context: &web_sys::CanvasRenderingContext2d, line: &mut impl Iterator<Item=DrawOp>) {
    context.begin_path();
    let mut i = 0;
    for op in line {
        match op {
            DrawOp::MoveTo(Coord2D { x, y }) => context.move_to(x, y),
            DrawOp::LineTo(Coord2D { x, y }) => context.line_to(x, y)
        }
    }
    context.stroke();
    context.fill();
}

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


struct World<F, T1, T2>
where 
    F: FnMut() -> T1,
    T1: Iterator<Item=T2>,
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

impl<'a, F, T1, T2> World<F, T1, T2>
where 
    F: (FnMut() -> T1) + 'static,
    T1: Iterator<Item=T2> + 'static,
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
        wasm_bindgen_futures::spawn_local(async move {
            loop {
                World::update_state(&this2).await;
            }
        });
    }

    async fn update_state(this: &Rc<Mutex<Self>>) -> () {
        let mut cur_yaw: f64;
        {
            let t = &mut *this.lock().await;
            t.cur_yaw = f64::to_radians((f64::to_degrees(t.cur_yaw) - 1.0) % 360.0);
            match t.cur_bounce_direction {
                BounceDirection::BounceUp(x) => {
                    t.cur_pitch = f64::to_radians((f64::to_degrees(t.cur_pitch) + x));
                    if t.cur_pitch > f64::to_radians(30.0) {
                        t.cur_bounce_direction = BounceDirection::BounceDown(-0.2);
                    }
                }
                BounceDirection::BounceDown(x) => {
                    t.cur_pitch = f64::to_radians((f64::to_degrees(t.cur_pitch) + x));
                    if t.cur_pitch < f64::to_radians(-30.0) {
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
        inp_lines: impl Iterator<Item=impl Iterator<Item=CoordGeo> + 'b> + 'b)
    -> impl Iterator<Item=impl Iterator<Item=DrawOp> + 'b> + 'b {

        let scale_fac = f64::min(self.width*0.8/2.0, self.height*0.8/2.0);
        let scale = Scale2D { x: scale_fac, y: scale_fac };
        let translate = Translate2D { x: self.width/2.0, y: self.height/2.0 };

        let lines = 
            inp_lines.into_iter()
            .map(move |x| { project(proj_3d, x.into_iter()) })
            .map(move |line| { 
                //let (it1, it2): (impl Iterator<Item = Coord3D>, impl Iterator<Item = Coord3D>) = line.map(|x| {(x, x)} ).unzip();
                //it2.take();
                //let (it1, it2) : (impl Iterator<Item=Coord3D>, impl Iterator<Item=Coord3D>) = duplicate_iter(line);

                let mut culled_last = true;
                let mut draw_ops = Vec::<DrawOp>::new();
                for p_3d in line {
                    let cull_this = proj_2d.cull(&p_3d, 0.0);
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
                draw_ops.into_iter()
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

        self.req_animation_frame();
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

    //let shp = Shapefile::from_bytes(BOUNDARIES_SHP);
    let mut boundaries_shp_curs = std::io::Cursor::new(&BOUNDARIES_SHP[..]);
    let mut shp = shapefile::ShapeReader::new(boundaries_shp_curs).expect("unable to read shapefile");

    //let counts = 
    //    shp.iter_shapes().fold([0,0,0,0,0,0,0,0,0,0,0,0,0,0], |acc, el| {
    //        let mut acc = acc.clone();
    //        if let Ok(s) = el {
    //            match s {
    //                shapefile::record::Shape::NullShape                                => acc[0] += 1,
    //                shapefile::record::Shape::Point(p)                           => acc[1] += 1,
    //                shapefile::record::Shape::PointM(p)                         => acc[2] += 1,
    //                shapefile::record::Shape::PointZ(p)                         => acc[3] += 1,
    //                shapefile::record::Shape::Polyline(p)       => acc[4] += 1,
    //                shapefile::record::Shape::PolylineM(p)     => acc[5] += 1,
    //                shapefile::record::Shape::PolylineZ(p)     => acc[6] += 1,
    //                shapefile::record::Shape::Polygon(p)         => acc[7] += 1,
    //                shapefile::record::Shape::PolygonM(p)       => acc[8] += 1,
    //                shapefile::record::Shape::PolygonZ(p)       => acc[9] += 1,
    //                shapefile::record::Shape::Multipoint(p)   => acc[10] += 1,
    //                shapefile::record::Shape::MultipointM(p) => acc[11] += 1,
    //                shapefile::record::Shape::MultipointZ(p) => acc[12] += 1,
    //                shapefile::record::Shape::Multipatch(p)                 => acc[13] += 1
    //            }
    //        }
    //        acc
    //    });
    
    //web_sys::console::log_1(&counts.map(|x|{x.to_string()}).join(", ").into());

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

    let res = 1.0; // degrees latitude/longitude difference to be included TODO: proper shape simplification
    let mut lines: Vec<Vec<CoordGeo>> = Vec::new();
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
                        lines.push(out_line)
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
        cur_yaw: 0.0,
        cur_pitch: f64::to_radians(0.0),
        cur_bounce_direction: BounceDirection::BounceUp(0.5),
        get_lines_it: move || { 
            lines.clone().into_iter().map(|x|{x.into_iter()})

            // Lat lon line rendering
            //let n_lines = 18;
            //let resolution = 36; 
            //let (lat_lines, lon_lines) = gen_lat_lon_lines(n_lines, resolution);
            //lon_lines
            //lat_lines.chain(lon_lines)

            // Problem with the following: shp.iter_shapes() can only be consumed once;
            // after that it appears to yield infinite Error()s, leading to infinite
            // looping
            // shp.iter_shapes().filter_map(|maybe_shp| {
            //     match maybe_shp {
            //         Err(_) => Option::None,
            //         Ok(shp) => {
            //             match shp {
            //                 shapefile::record::Shape::Polygon(polyshp) => Option::Some (
            //                     polyshp.rings().iter().fold(Vec::<CoordGeo>::new(), |mut acc, el| {
            //                         match el {
            //                             shapefile::record::polygon::PolygonRing::Inner(p) => acc,
            //                             shapefile::record::polygon::PolygonRing::Outer(p) => {
            //                                 acc.extend(p.into_iter().map(|x|{
            //                                     let out = CoordGeo{latitude: x.x, longitude: x.y};
            //                                     web_sys::console::log_1(&out.to_string().into());
            //                                     out
            //                                 }));
            //                                 acc
            //                             }
            //                         }
            //                     })
            //                 ),
            //                 otherwise => Option::None
            //             }
            //         }
            //     }
            // }).collect::<Vec<Vec<CoordGeo>>>().into_iter().map(|x|{x.into_iter()})
        }
    };
    let moved_world = wrld.wrap();
    World::init(&moved_world);
    World::run(&moved_world);

    // Less idiomatically:
    // let canvasWidth = canvasElem
    //     .get_attribute("width")
    //     .map_or(None, (|x| x.parse::<u32>().map_or(None, |y| Some(y))))
    //     .unwrap_or(400);

        //.m
        //.map(|x| x.parse<u32>().unwrap_or())
        //unwrap_or("400").parse::<u32>().unwrap_or(400);

    web_sys::console::log_2(&canvas_width.to_string().into(), &canvas_height.to_string().into());

    //context.begin_path();
    //context.move_to(0.05*canvas_width, 0.05*canvas_height);
    //context.line_to(0.95*canvas_width, 0.05*canvas_height);
    //context.line_to(0.95*canvas_width, 0.95*canvas_height);
    //context.line_to(0.05*canvas_width, 0.95*canvas_height);
    //context.close_path();
    //context.set_stroke_style_str("black");
    //context.stroke();

}
