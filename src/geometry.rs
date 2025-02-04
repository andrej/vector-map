use std::fmt;
use std::ops;

use crate::console_log;
use crate::DrawOp;

// --------------------------------------------------------------------------
// CoordGeo

#[derive(Copy, Clone)]
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

#[derive(Copy, Clone)]
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

#[derive(Copy, Clone)]
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
            x: r * f64::cos(-lon) * f64::cos(-lat),
            y: r * f64::sin(-lon) * f64::cos(-lat),
            z: r * f64::sin(-lat) 
        }
    }
}

fn project_coord3d_to_coordgeo(Coord3D{ x, y, z}: &Coord3D) -> CoordGeo {
    let r = 1.0;
    //let longitude = f64::signum(*y) * (f64::atan(f64::abs(*y)/f64::abs(*x)) + if *x < 0.0 { f64::to_radians(90.0) } else { 0.0 } );
    CoordGeo {
        latitude: f64::asin(z / r),
        longitude: f64::atan2(*y, *x),
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
fn get_viewport_intersection_point(inside: Coord3D, outside: Coord3D) -> Coord3D {
    let outside_rev = project_coord3d_to_coordgeo(&outside);
    let inside_rev = project_coord3d_to_coordgeo(&inside);
    let lat_slope = (outside_rev.latitude - inside_rev.latitude) / (outside_rev.longitude - inside_rev.longitude);
    let lon = f64::signum(inside_rev.longitude) * f64::to_radians(90.0);
    let edge_rev = CoordGeo {
        longitude: -lon,
        latitude: -(inside_rev.latitude + lat_slope * (lon - inside_rev.longitude))
    };
    //console_log!("inside: {},  outside: {}, edge: {}", inside_rev, outside_rev, edge_rev);
    let sphere_proj = SphereProjection;
    let edge = sphere_proj.project(&edge_rev);
    edge
}

#[derive(Clone)]
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

pub struct ClampedIterator<InputIter>
where InputIter: Iterator<Item=Coord3D> {
    iter: InputIter,
    next: Option<ClampedIteratorPoint>,
}

impl<InputIter> ClampedIterator<InputIter>
where InputIter: Iterator<Item=Coord3D> {
    pub fn new(mut iter: InputIter) -> Self {
        Self {
            iter: iter,
            next: Option::None 
        }
    }

    fn next_visible(&mut self) -> (Option<Coord3D>, Option<Coord3D>) {
        let mut maybe_before_current = Option::None;
        let mut maybe_current = self.iter.next();
        while maybe_current.is_some() && maybe_current.unwrap().x < 0.0 {
            maybe_before_current = maybe_current;
            maybe_current = self.iter.next();
        }
        (maybe_before_current, maybe_current)
    }
}

impl<InputIter> Iterator for ClampedIterator<InputIter>
where InputIter: Iterator<Item=Coord3D> {
    type Item = ClampedIteratorPoint;
    fn next(&mut self) -> Option<ClampedIteratorPoint> {
        use ClampedIteratorPoint::*;
        let current = match self.next.clone() {
            Some(LastVisible(p)) => {
                // return LastVisibles immediately; we start fresh after this
                self.next = None;
                return Some(LastVisible(p))
            },
            Some(Visible(p)) | Some(FirstVisible(p)) => {
                self.next = None;
                p
            },
            None => self.iter.next()?
        };
        
        if current.x >= 0.0 {
            // current is visible
            let maybe_next = self.iter.next();
            if let Some(next) = maybe_next { 
                if next.x < 0.0 {
                    // next is invisible; clamp and enqueue it as LastVisible
                    let next_clamped = get_viewport_intersection_point(current, next);
                    self.next = Some(LastVisible(next_clamped));
                } else {
                    // next is visible
                    self.next = Some(Visible(next));
                }
            }
            // Return this point as regular Visible
            return Some(Visible(current));

        } else {
            // current invisible
            // find next visible
            let mut maybe_before_next_visible = Option::Some(current);
            let mut maybe_next_visible = Option::None;
            while let it@Some(next_visible) = self.iter.next() {
                if next_visible.x < 0.0 {
                    // found visible point
                    maybe_next_visible = it;
                }
                maybe_before_next_visible = it;
            }
            if let (Some(before_next_visible), Some(next_visible)) = (maybe_before_next_visible, maybe_next_visible) {
                let next_clamped = get_viewport_intersection_point(next_visible, before_next_visible);
                self.next = Some(Visible(next_visible));
                return Some(FirstVisible(next_clamped));
            }
        }

        // input iterator exhausted
        return Option::None
    }
}

pub struct ClampedArcIterator<InputIter>
where InputIter: Iterator<Item=ClampedIteratorPoint> {
    iter: InputIter,
    a: Option<ClampedIteratorPoint>,
    b: Option<ClampedIteratorPoint>
}

impl<InputIter> ClampedArcIterator<InputIter>
where InputIter: Iterator<Item=ClampedIteratorPoint> {
    pub fn new(mut iter: InputIter) -> Self {
        let next = iter.next();
        Self {
            iter: iter,
            a: Option::None,
            b: next
        }
    }
}

impl<InputIter> Iterator for ClampedArcIterator<InputIter>
where InputIter: Iterator<Item=ClampedIteratorPoint> {
    type Item = DrawOp<Coord2D>;
    fn next(&mut self) -> Option<DrawOp<Coord2D>> {
        use ClampedIteratorPoint::*;
        let a = self.a.clone();
        let b = self.b.clone();
        self.a = b.clone();
        self.b = self.iter.next();

        match (a, b) {
            (Some(FirstVisible(p)), _) 
                => Some(DrawOp::MoveTo(Coord2D { x: p.y, y: p.z })),
            (Some(Visible(p)), _)  |
            (Some(LastVisible(p)), _) 
                => Some(DrawOp::LineTo(Coord2D { x: p.y, y: p.z })),
            (None, _) => None
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
