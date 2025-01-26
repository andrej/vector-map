use crate::geometry::Coord2D;
use crate::console_log;

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