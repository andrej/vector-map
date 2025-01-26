use crate::geometry::Coord2D;
use crate::console_log;
use crate::*;
use wasm_bindgen::prelude::*;
use std::rc::Rc;
use futures::lock::Mutex;

/// Drawing operation; this enum is not meant to be stored in memory, as it
/// would be quite inefficient to e.g. store every point of a line inside of
/// a "LineTo" operation. For such a use case, a vector of coordinates would 
/// make more sense; this could then be mapped on-the-fly to this enum.
pub enum DrawOp<CoordT> {
    BeginPath,
    MoveTo(CoordT),
    LineTo(CoordT),
    ClosePath,
    Stroke(String),
    Fill(String),
}

impl DrawOp<Coord2D> {
    pub fn draw(&self, context: &web_sys::CanvasRenderingContext2d) {
        match self {
            DrawOp::BeginPath =>  {
                //console_log!("begin path");
                context.begin_path()
            },
            DrawOp::MoveTo(Coord2D { x, y }) => {
                //console_log!("move to {} {}", x, y);
                context.move_to(*x, *y)
            },
            DrawOp::LineTo(Coord2D { x, y }) => {
                //console_log!("line to {} {}", x, y);
                context.line_to(*x, *y)
            },
            DrawOp::ClosePath => {
                //console_log!("close path");
                context.close_path()
            },
            DrawOp::Stroke(style) => {
                //console_log!("set stroke style {}", style);
                context.set_stroke_style_str(style);
                context.stroke();
            },
            DrawOp::Fill(style) => {
                //console_log!("set fill style {}", style);
                context.set_fill_style_str(style);
                context.fill();
            }
        }
    }
}

/// Consumes the line iterator and turns it into an iterator of drawing
/// operations. The lifetime annotation is for the references contained within
/// the DrawOp. We do not create any of the enum values that contain references
/// in this function, so the lifetime can be anything you need at the call site.
pub fn coord_iter_into_draw_ops<CoordT>(mut iter: impl Iterator<Item=CoordT>) -> impl Iterator<Item=DrawOp<CoordT>> {
    std::iter::once(DrawOp::BeginPath)
        .chain(iter.next().map(|x| { DrawOp::MoveTo(x) }).into_iter())
        .chain(iter.map(|x| { DrawOp::LineTo(x) }))
        .chain(std::iter::once(DrawOp::ClosePath))
}


/// Manages the main render and state update loop by calling into an
/// impl `CanvasRenderLoopState`
pub struct CanvasRenderLoop<CanvasRenderLoopStateT, DrawOpIterT>
where CanvasRenderLoopStateT: CanvasRenderLoopState<DrawOpIterT>,
DrawOpIterT: Iterator<Item=DrawOp<Coord2D>>
{
    window: web_sys::Window,
    performance: web_sys::Performance,
    context: web_sys::CanvasRenderingContext2d,
    width: f64,
    height: f64,
    render_closure: Option<Closure<dyn Fn(f64)>>,
    state: CanvasRenderLoopStateT,
    last_state_update_t: f64
}

const ANIMATE: bool = true;
impl<CanvasRenderLoopStateT, DrawOpIterT> CanvasRenderLoop<CanvasRenderLoopStateT, DrawOpIterT>
where CanvasRenderLoopStateT: CanvasRenderLoopState<DrawOpIterT> + 'static,
DrawOpIterT: Iterator<Item=DrawOp<Coord2D>>
{
    pub fn new(window: web_sys::Window, canvas_id: &str, state: CanvasRenderLoopStateT) -> Self {
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
            state: state,
            last_state_update_t: 0.0
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
    pub fn wrap(self) -> Rc<Mutex<Self>> {
        let this = Rc::new(Mutex::new(self));
        this
    }

    pub fn init(this: &Rc<Mutex<Self>>) -> () {
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

    pub fn run(this: &Rc<Mutex<Self>>) -> () {
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
                    CanvasRenderLoop::update(&this2).await;
                }
            });
        }
    }

    async fn update(this: &Rc<Mutex<Self>>) -> () {
        let mut cur_yaw: f64;
        {
            let this = &mut *this.lock().await;
            let t_cur = this.performance.now();
            let t_diff = t_cur - this.last_state_update_t;
            this.state.update(t_diff);
            this.last_state_update_t = t_cur;
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
        let draw_ops = self.state.frame();
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

pub trait CanvasRenderLoopState<DrawOpsIterT>
where DrawOpsIterT: Iterator<Item=DrawOp<Coord2D>>
{
    fn frame(&self) -> Option<DrawOpsIterT>;
    fn update(&mut self, t_diff: f64);
}