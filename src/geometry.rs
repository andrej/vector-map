use std::fmt;
use std::ops;

use crate::console_log;
use crate::DrawOp;

// --------------------------------------------------------------------------
// CoordGeo

#[derive(Copy, Clone, Debug)]
pub struct CoordGeo {
    pub latitude: f64,
    pub longitude: f64
}

impl fmt::Display for CoordGeo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{ latitude: {}, longitude: {} }}", self.latitude, self.longitude)
    }
}


// --------------------------------------------------------------------------
// Coord3D

#[derive(Copy, Clone, Debug)]
pub struct Coord3D {
    pub x: f64,
    pub y: f64,
    pub z: f64
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

// TODO: Implement the following as methods on the struct?

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


// --------------------------------------------------------------------------
// Coord2D

#[derive(Copy, Clone, Debug)]
pub struct Coord2D {
    pub x: f64,
    pub y: f64
}

impl fmt::Display for Coord2D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{ x: {}, y: {}}}", self.x, self.y)
    }
}


// --------------------------------------------------------------------------
// Projection

pub trait Projection<From> {
    type To;
    fn project(&self, input: &From) -> Self::To;
}

pub struct SphereProjection;

impl Projection<CoordGeo> for SphereProjection {
    type To = Coord3D;
    fn project(&self, &CoordGeo { latitude: lat, longitude: lon}: &CoordGeo) -> Coord3D {
        let r = 1.0;
        // Positive longitudes are EAST of meridian
        // Positive latitudes are NORTH of the equator
        Coord3D { 
            x: r * f64::cos(-lon) * f64::cos(lat),
            y: r * f64::sin(-lon) * f64::cos(lat),
            z: r * f64::sin(lat) 
        }
    }
}

impl Projection<Coord3D> for SphereProjection {
    type To = CoordGeo;
    fn project(&self, Coord3D{ x, y, z}: &Coord3D) -> CoordGeo {
        let r = 1.0;
        //let longitude = f64::signum(*y) * (f64::atan(f64::abs(*y)/f64::abs(*x)) + if *x < 0.0 { f64::to_radians(90.0) } else { 0.0 } );
        CoordGeo {
            latitude: f64::asin(z / r),
            longitude: -f64::atan2(*y, *x),
        }
    }
}

pub struct OrthogonalProjection {
    x_axis: Coord3D,
    y_axis: Coord3D,
    z_axis: Coord3D
}

impl OrthogonalProjection {
    pub fn new(x_axis: Coord3D, y_axis: Coord3D) -> OrthogonalProjection {
        //let x_axis = normalize(&x_axis);
        //let y_axis = normalize(&y_axis);
        //OrthogonalProjection {
        //    x_axis: x_axis,
        //    y_axis: y_axis,
        //    z_axis: cross_product(&x_axis, &y_axis)
        //}
        OrthogonalProjection {
            x_axis: Coord3D { x: 0.0, y: 1.0, z: 0.0 },
            y_axis: Coord3D { x: 0.0, y: 0.0, z: 1.0 },
            z_axis: Coord3D { x: 1.0, y: 0.0, z: 0.0 },
        }
    }

    pub fn new_from_normal(normal: Coord3D) -> OrthogonalProjection {
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
        let y_axis = project_onto_plane(&normal, &v);
        // Find a vector orthogonal to both x_axis and `normal`.
        // By making it orthogonal to `normal` it is guaranteed to lie in the plane.
        let x_axis = cross_product(&normal, &y_axis);
        OrthogonalProjection::new(x_axis, y_axis)
    }
    

    pub fn new_from_angles(latitude: f64, longitude: f64) -> OrthogonalProjection {
        let x_axis = Coord3D {
            x: f64::cos(longitude) * f64::cos(latitude),
            y: f64::sin(longitude) * f64::cos(latitude),
            z: f64::sin(latitude)
        };
        let y_axis = Coord3D {
            x: f64::cos(longitude + f64::to_radians(90.0)) * f64::cos(latitude),
            y: f64::sin(longitude + f64::to_radians(90.0)) * f64::cos(latitude),
            z: f64::sin(latitude)
        };
        let z_axis = cross_product(&x_axis, &y_axis);
        OrthogonalProjection {
            x_axis: x_axis,
            y_axis: y_axis,
            z_axis: z_axis
        }
    }
}


impl Projection<Coord3D> for OrthogonalProjection {
    type To = Coord3D;

    fn project(&self, input: &Coord3D) -> Coord3D {
        Coord3D { 
            x: dot_product(&input, &self.x_axis), 
            y: dot_product(&input, &self.y_axis),
            z: dot_product(&input, &self.z_axis)
        }
    }
}

/// Takes two points, one inside the viewport, the other outside of it, and
/// returns the intersection of the line between them and the edge of the
/// visible area.
pub fn get_viewport_intersection_point(inside: Coord3D, outside: Coord3D) -> Coord3D {
    let proj = SphereProjection;
    let mut outside_rev = proj.project(&outside);
    let mut inside_rev = proj.project(&inside);
    let mut lon_diff = outside_rev.longitude - inside_rev.longitude;
    // sgn needs to be -1 if we are in the left hemisphere, +1 if in the right
    // (i.e. we will clamp the longitude to -90 deg or +90 deg)
    // if I have to increase the longitude to get from inside_rev to outside_rev
    // on the shortest path, it's in the left hemisphere; if I have to decrease 
    // the longitude, it's in the right
    if lon_diff > std::f64::consts::PI || lon_diff < -std::f64::consts::PI {
        lon_diff = -(lon_diff - std::f64::consts::PI);
    }
    let sgn = f64::signum(-lon_diff);
    let lat_slope = (outside_rev.latitude - inside_rev.latitude) / (outside_rev.longitude - inside_rev.longitude);
    let lon = sgn * f64::to_radians(90.0);
    let edge_rev = CoordGeo {
        longitude: lon,
        latitude: (inside_rev.latitude + lat_slope * (lon - inside_rev.longitude))
    };
    let sphere_proj = SphereProjection;
    let mut edge = sphere_proj.project(&edge_rev);
    // due to floating point inaccuracies, x might be a very small positive 
    // number at this point, even though we just chose a point that should be
    // the last visible point. To make sure it doesn't get culled, set it to
    // 0.0, because that is the accurate solution.
    edge.x = 0.0;
    edge
}

#[derive(Clone,Debug)]
pub enum ClampedIteratorPoint {
    /// This point is visible, and the points before and after it (if any) are
    /// also visible.
    Visible(Coord3D), 
    /// This point is the last visible point before the shape leaves the
    /// viewport. It is likely the result of clamping the intersection of a
    /// visible point in the input iterator and its following invisible point.
    LastVisible(Coord3D),
    /// This point is the first visible point after previous points of the
    /// shape were outside the viewport.
    FirstVisible(Coord3D)
    // Note that if we have a viewport that is alrger than zero (which we 
    // assume) we cannot have a point that is both FirstVisible *and* 
    // LastVisible, as those points would be clamped to the respective edges of
    // the viewport. We also assume that we don't have lines that go 
    // from point A to point B to point A, where A is outside of the viewport
    // and B is inside, and we assume no point lies exactly on the boundary of
    // the viewport.
}

impl ClampedIteratorPoint {
    fn get_coord(&self) -> &Coord3D {
        match self {
            Self::Visible(x) | Self::LastVisible(x) | Self::FirstVisible(x) => { x }
        }
    }
}

pub fn into_clamped_iter(iter: impl Iterator<Item=Coord3D>) -> impl Iterator<Item=ClampedIteratorPoint> {
    iter.map(|point| ClampedIteratorPoint::Visible(point))
}

pub struct ClampedIterator<InputIter, IsVisibleFnT, ClampFnT>
where InputIter: Iterator<Item=ClampedIteratorPoint>,
IsVisibleFnT: Fn(Coord3D) -> bool,
ClampFnT: Fn(Coord3D, Coord3D) -> Coord3D
{
    iter: InputIter,
    next: Option<ClampedIteratorPoint>,
    after_next: Option<ClampedIteratorPoint>,
    is_visible_fn: IsVisibleFnT,
    clamp_fn: ClampFnT
}

impl<InputIter, IsVisibleFnT, ClampFnT> ClampedIterator<InputIter, IsVisibleFnT, ClampFnT>
where InputIter: Iterator<Item=ClampedIteratorPoint>,
IsVisibleFnT: Fn(Coord3D) -> bool,
ClampFnT: Fn(Coord3D, Coord3D) -> Coord3D
{
    pub fn new(mut iter: InputIter, is_visible_fn: IsVisibleFnT, clamp_fn: ClampFnT) -> Self {
        Self {
            iter: iter,
            next: Option::None,
            after_next: Option::None,
            is_visible_fn: is_visible_fn,
            clamp_fn: clamp_fn
        }
    }
}

impl<InputIter, IsVisibleFnT, ClampFnT> Iterator for ClampedIterator<InputIter, IsVisibleFnT, ClampFnT>
where InputIter: Iterator<Item=ClampedIteratorPoint>,
IsVisibleFnT: Fn(Coord3D) -> bool,
ClampFnT: Fn(Coord3D, Coord3D) -> Coord3D {
    type Item = ClampedIteratorPoint;
    fn next(&mut self) -> Option<ClampedIteratorPoint> {

        use ClampedIteratorPoint::*;
        let current = match self.next.clone() {
            Some(LastVisible(p)) => {
                // return LastVisibles immediately; we start fresh after this
                self.next = self.after_next.clone();
                self.after_next = None;
                return Some(LastVisible(p))
            },
            Some(Visible(p)) | Some(FirstVisible(p)) => {
                self.next = self.after_next.clone();
                self.after_next = None;
                p
            },
            None => match self.iter.next() {
                None => return None,
                // FIXME below code is literally copy pasted ...
                Some(LastVisible(p)) => {
                    // return LastVisibles immediately; we start fresh after this
                    self.next = self.after_next.clone();
                    self.after_next = None;
                    return Some(LastVisible(p))
                },
                Some(Visible(p)) | Some(FirstVisible(p)) => {
                    self.next = self.after_next.clone();
                    self.after_next = None;
                    p
                },
            }
        };
        
        if (self.is_visible_fn)(current) {
            // current is visible
            let maybe_next = self.iter.next();
            if let Some(next) = maybe_next { 
                let next = next.get_coord();
                if !(self.is_visible_fn)(*next) {
                    // next is invisible; clamp and enqueue it as LastVisible
                    let next_clamped = (self.clamp_fn)(current, *next);
                    self.next = Some(LastVisible(next_clamped));
                    // The intersection between next and the point after next
                    // might not be at LastVisible ...
                    // Visible here is a lie, but we handle it correctly.
                    self.after_next = Some(Visible(*next));
                } else {
                    // next is visible
                    self.next = Some(Visible(*next));
                }
            }
            // Return this point as regular Visible
            return Some(Visible(current));

        } else {
            // current invisible
            // find next visible
            let mut maybe_before_next_visible = Option::Some(current);
            let mut maybe_next_visible = Option::None;
            while let Some(next_visible) = self.iter.next() {
                let next_visible = next_visible.get_coord();
                if (self.is_visible_fn)(*next_visible) {
                    // found visible point
                    maybe_next_visible = Some(*next_visible);
                    break;
                }
                maybe_before_next_visible = Some(*next_visible);
            }
            assert!(maybe_next_visible.is_some() && maybe_before_next_visible.is_some() || !maybe_next_visible.is_some());
            if let (Some(before_next_visible), Some(next_visible)) = (maybe_before_next_visible, maybe_next_visible) {
                let next_clamped = (self.clamp_fn)(next_visible, before_next_visible);
                self.next = Some(Visible(next_visible));
                return Some(FirstVisible(next_clamped));
            }
        }

        // input iterator exhausted
        return Option::None
    }
}

pub struct ClampedArcIterator<'a, InputIter>
where InputIter: Iterator<Item=ClampedIteratorPoint> + 'a {
    iter: InputIter,
    a: Option<ClampedIteratorPoint>,
    b: Option<ClampedIteratorPoint>,
    first: Option<ClampedIteratorPoint>,
    draw_arc: bool,
    arc_center: Coord2D,
    arc_radius: f64,
    _phantom: std::marker::PhantomData<&'a InputIter>
}

impl<'a, InputIter> ClampedArcIterator<'a, InputIter>
where InputIter: Iterator<Item=ClampedIteratorPoint> {
    pub fn new(mut iter: InputIter, draw_arc: bool, arc_center: Coord2D, arc_radius: f64) -> Self {
        let first = iter.next();
        Self {
            iter: iter,
            a: Option::None,
            b: first.clone(),
            first: first,
            draw_arc: draw_arc,
            arc_center: arc_center,
            arc_radius: arc_radius,
            _phantom: std::marker::PhantomData
        }
    }
}

impl<'a, InputIter> Iterator for ClampedArcIterator<'a, InputIter>
where InputIter: Iterator<Item=ClampedIteratorPoint> + 'a {
    type Item = DrawOp<'a, Coord2D>;
    fn next(&mut self) -> Option<DrawOp<'a, Coord2D>> {
        use ClampedIteratorPoint::*;
        let next = match self.iter.next() {
            p@Some(_) => p,
            None => { let p = self.first.clone(); self.first = None; p }
        };
        let a = self.a.clone();
        let b = self.b.clone();
        self.a = b.clone();
        self.b = next;

        match (a, b) {
            (Some(LastVisible(a)), Some(FirstVisible(b))) => {
                if self.draw_arc {
                    let (ax, ay) = (a.y, -a.z);
                    let (bx, by) = (b.y, -b.z);
                    use std::f64::consts::PI;
                    let angle_a = f64::atan2(ay, ax);
                    let angle_b = f64::atan2(by, bx);
                    let angle_diff = (angle_b - angle_a + 2.0*PI) % (2.0*PI);
                    Some(DrawOp::Arc(self.arc_center, self.arc_radius, angle_a, angle_b, angle_diff>PI))
                } else {
                    Some(DrawOp::MoveTo(Coord2D { x: b.y, y: -b.z }))
                }
            },
            (_, Some(FirstVisible(p))) 
                => Some(DrawOp::LineTo(Coord2D { x: p.y, y: -p.z })),
            (_, Some(Visible(p)))  |
            (_, Some(LastVisible(p))) 
                => Some(DrawOp::LineTo(Coord2D { x: p.y, y: -p.z })),
            (_, None) => None
        }
    }
}

// --------------------------------------------------------------------------
// Transform

// TODO: Refactor transforms into matrices; allows for multiplying matrices
// then applying single transform all at once

pub trait Transform<CoordType> {
    fn transform(&self, input: &CoordType) -> CoordType;
}

#[derive(Copy, Clone)]
pub struct Translate2D {
    pub x: f64,
    pub y: f64   
}

impl Transform<Coord2D> for Translate2D {
    fn transform(&self, input: &Coord2D) -> Coord2D {
        Coord2D { x: input.x + self.x, y: input.y + self.y }
    }
}

#[derive(Copy, Clone)]
pub struct Scale2D {
    pub x: f64,
    pub y: f64
}

impl Transform<Coord2D> for Scale2D {
    fn transform(&self, input: &Coord2D) -> Coord2D {
        Coord2D { x: input.x * self.x, y: input.y * self.y }
    }
}
