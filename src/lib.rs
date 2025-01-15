mod utils;

use wasm_bindgen::prelude::*;
use std::fmt;

struct CoordGeo {
    latitude: f64,
    longitude: f64
}

struct Coord3D {
    x: f64,
    y: f64,
    z: f64
}

struct Coord2D {
    x: f64,
    y: f64
}

trait Projection<From, To> {
    fn project(&self, input: &From) -> To;
}

struct SphereProjection;

impl Projection<CoordGeo, Coord3D> for SphereProjection {
    fn project(&self, input: &CoordGeo) -> Coord3D {
        let r = 1.0;
        let [lat, lon] = [input.latitude, input.longitude].map(f64::to_radians);
        Coord3D { x: r * f64::cos(lat) * f64::cos(lon),
                  y: r * f64::cos(lat) * f64::sin(lon),
                  z: r * f64::sin(lat) }
    }
}

struct YZPlaneProjection;

impl Projection<Coord3D, Coord2D> for YZPlaneProjection {
    fn project(&self, input: &Coord3D) -> Coord2D {
        Coord2D { x: input.y, y: input.z }
    }
}

// TODO: Refactor transforms into matrices; allows for multiplying matrices
// then applying single transform all at once

trait Transform<CoordType> {
    fn transform(&self, input: &CoordType) -> CoordType;
}

struct Translate2D {
    x: f64,
    y: f64   
}

impl Transform<Coord2D> for Translate2D {
    fn transform(&self, input: &Coord2D) -> Coord2D {
        Coord2D { x: input.x + self.x, y: input.y + self.y }
    }
}

struct Scale2D {
    x: f64,
    y: f64
}

impl Transform<Coord2D> for Scale2D {
    fn transform(&self, input: &Coord2D) -> Coord2D {
        Coord2D { x: input.x * self.x, y: input.y * self.y }
    }
}


fn project(line: impl Iterator<Item=CoordGeo>) -> impl Iterator<Item=Coord2D> { 
    let proj_3d = SphereProjection;
    let proj_2d = YZPlaneProjection;
    line.map(move |point| { 
        let Coord3D { x, y, z} = proj_3d.project(&point);
        proj_2d.project(&Coord3D { x, y, z})
    })
}

fn project_generic<I1, I2>(line: I1) -> impl Iterator<Item=Coord2D>
where 
    I1: Iterator<Item=CoordGeo>,
{ 
    let proj_3d = SphereProjection;
    let proj_2d = YZPlaneProjection;
    line.map(move |point| { proj_2d.project(&proj_3d.project(&point)) })
}

fn draw_line(context: &web_sys::CanvasRenderingContext2d, line: &mut impl Iterator<Item=Coord2D>) {
    context.begin_path();
    let first_point = line.nth(0).expect("every line should have at least one point");
    let Coord2D {x, y} = first_point;
    context.move_to(x, y);
    for Coord2D {x, y} in line {
        context.line_to(x, y);
    }
    context.stroke();
}

#[wasm_bindgen]
pub fn main() {
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

    // Less idiomatically:
    // let canvasWidth = canvasElem
    //     .get_attribute("width")
    //     .map_or(None, (|x| x.parse::<u32>().map_or(None, |y| Some(y))))
    //     .unwrap_or(400);

        //.m
        //.map(|x| x.parse<u32>().unwrap_or())
        //unwrap_or("400").parse::<u32>().unwrap_or(400);

    web_sys::console::log_2(&canvas_width.to_string().into(), &canvas_height.to_string().into());

    context.begin_path();
    context.move_to(0.05*canvas_width, 0.05*canvas_height);
    context.line_to(0.95*canvas_width, 0.05*canvas_height);
    context.line_to(0.95*canvas_width, 0.95*canvas_height);
    context.line_to(0.05*canvas_width, 0.95*canvas_height);
    context.close_path();
    context.set_stroke_style_str("black");
    context.stroke();

    // Generate some latitude and longitude points
    let n_lines = 18;
    let resolution = 36; 
    let lat_lines = 
        (0..n_lines).map(|lat_i| { 
            let lat: f64 = -90.0 + (lat_i as f64) / (n_lines as f64) * 180.0;
            (0..resolution).map(move |lon_i| { 
                let lon: f64 = -180.0 + (lon_i as f64) / (resolution as f64) * 360.0;
                //let lon: f64 = -90.0 + (lon_i as f64) / (resolution as f64) * 180.0;
                CoordGeo { latitude: lat, longitude: lon } 
            })
        });
    let lon_lines =
        (0..n_lines).map(|lon_i| { 
            let lon: f64 = -90.0 + (lon_i as f64) / (n_lines as f64) * 180.0;
            (0..resolution).map(move |lat_i| { 
                let lat: f64 = -180.0 + (lat_i as f64) / (resolution as f64) * 360.0;
                CoordGeo { latitude: lat, longitude: lon } 
            })
        });
    
    let lat_lines = lat_lines.map(project);
    let lon_lines = lon_lines.map(project);

    let scale_fac = f64::min(canvas_width*0.8/2.0, canvas_height*0.8/2.0);
    let scale = Scale2D { x: scale_fac, y: scale_fac };
    let translate = Translate2D { x: canvas_width/2.0, y: canvas_height/2.0 };

    let lat_lines = lat_lines.map(|x| { x.map(|x| { translate.transform(&scale.transform(&x)) })});
    let lon_lines = lon_lines.map(|x| { x.map(|x| { translate.transform(&scale.transform(&x)) })});

    lat_lines.for_each(|mut l| { draw_line(&context, &mut l) } );
    lon_lines.for_each(|mut l| { draw_line(&context, &mut l) } );

}
