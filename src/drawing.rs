use crate::geometry::Coord2D;

pub struct Line<'a, It> 
where It: Iterator<Item=DrawOp>
{
    pub points: It,
    pub stroke_style: Option<&'a String>,
    pub fill_style: Option<&'a String>
}

pub enum DrawOp {
    MoveTo(Coord2D),
    LineTo(Coord2D)
}

pub fn draw_line<It>
(
    context: &web_sys::CanvasRenderingContext2d, 
    line: &mut Line<It>
) 
where It: Iterator<Item=DrawOp>
{
    context.begin_path();
    for op in &mut line.points {
        match op {
            DrawOp::MoveTo(Coord2D { x, y }) => context.move_to(x, y),
            DrawOp::LineTo(Coord2D { x, y }) => context.line_to(x, y)
        }
    }
    if let Some(ss) = line.stroke_style {
        context.set_stroke_style_str(&ss);
        context.stroke();
    }
    if let Some(fs) = line.fill_style {
        context.set_fill_style_str(fs);
        context.fill();
    }
}