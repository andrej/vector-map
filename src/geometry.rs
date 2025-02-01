use std::fmt;
use std::ops;

use crate::console_log;

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
    fn project(&self, input: impl Iterator<Item=From>) -> impl Iterator<Item=Self::To>;
}

pub struct SphereProjection;

impl Projection<CoordGeo> for SphereProjection {
    type To = Coord3D;
    fn project(&self, input: impl Iterator<Item=CoordGeo>) -> impl Iterator<Item=Coord3D> {
        let r = 1.0;
        // Positive longitudes are EAST of meridian
        // Positive latitudes are NORTH of the equator
        input.map(move |CoordGeo { latitude: lat, longitude: lon } | {
            Coord3D { 
                x: r * f64::cos(-lon) * f64::cos(-lat),
                y: r * f64::sin(-lon) * f64::cos(-lat),
                z: r * f64::sin(-lat) 
            }
        })
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
        //let normal = Coord3D {
        //    x: f64::cos(longitude) * f64::cos(latitude),
        //    y: f64::sin(longitude) * f64::cos(latitude),
        //    z: f64::sin(latitude)
        //};
        //OrthogonalProjection::new_from_normal(normal)
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

    fn project_single_with_depth(&self, input: Coord3D) -> Coord3D {
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
    let edge = sphere_proj.project(std::iter::once(edge_rev)).next().unwrap();
    edge
}

fn coord3d_to_coord2d_yz_plane(input: Coord3D) -> Coord2D {
    let ux = Coord3D { x: 0.0, y: 1.0, z: 0.0 };
    let uy = Coord3D { x: 0.0, y: 0.0, z: 1.0 };  // FIXME why -1 Z axis??
    Coord2D {
        x: dot_product(&input, &ux),
        y: dot_product(&input, &uy)
    }
}

impl Projection<Coord3D> for OrthogonalProjection {
    type To = Coord2D;

    fn project(&self, input: impl Iterator<Item=Coord3D>) -> impl Iterator<Item=Coord2D> {
        OrthogonalProjectionIterator::new(self, input)
    }
}

struct OrthogonalProjectionIterator<'a, InputIter>
where InputIter: Iterator<Item=Coord3D> {
    proj: &'a OrthogonalProjection,
    iter: InputIter,
    next: Option<Coord3D>
}

impl<'a, InputIter> OrthogonalProjectionIterator<'a, InputIter>
where InputIter: Iterator<Item=Coord3D> {
    fn new(proj: &'a OrthogonalProjection, mut iter: InputIter) -> Self {
        Self {
            proj: proj,
            iter: iter,
            next: Option::None
        }
    }
}

/*
last visible    this visible    - project this
last invisible  this visible    - project clamped last; queue this next
last visible    this invisible  - project clamped this
last invisible  this invisibile - move to next coord
*/


impl<'a, InputIter> Iterator for OrthogonalProjectionIterator<'a, InputIter>
where InputIter: Iterator<Item=Coord3D> {
    type Item = Coord2D;
    fn next(&mut self) -> Option<Coord2D> {
        let mut maybe_cur_projected = 
            if self.next.is_some() { self.next } 
            else { self.iter.next().map(|x| { self.proj.project_single_with_depth(x) }) };
        self.next = Option::None;
        let mut maybe_next_projected = 
            self.iter.next().map(|x| { self.proj.project_single_with_depth(x) });
        if let Some(cur) = maybe_cur_projected {
            if cur.x < 0.0 {
                let maybe_next_projected = loop {
                    // current is invisible; we need to advance to a point
                    // where "next" is visible
                    if let Some(next) = maybe_next_projected {
                        if next.x >= 0.0 {
                            // next visible point found
                            break maybe_next_projected;
                        }
                        maybe_cur_projected = maybe_next_projected;
                        maybe_next_projected = 
                            self.iter.next().map(|x| { self.proj.project_single_with_depth(x) });
                        continue
                    } else {
                        break Option::None
                    }
                };
                if let Some(next) = maybe_next_projected {
                    // Current is invisible but next is visible. Clamp 
                    // current to the last visible point on the line towards
                    // "next", return it, and enqueue "next" to be used as
                    // the next point (since we have already taken it out
                    // of the input iterator).
                    self.next = maybe_next_projected;
                    let cur_clamped = get_viewport_intersection_point(next, cur);
                    return Option::Some(Coord2D{ x: cur_clamped.y, y: cur_clamped.z });
                } else {
                    // Current is invisible and we have no next point to
                    // draw a line to; yield no more points. Note that we
                    // will have drawn the last visible point when we
                    // clamped "next" and enqueued it.
                    return Option::None
                }
            } else {
                // current is visible; we can project and return it as-is,
                // but we also need to take care of "next". If it is outside
                // the visible range, we need to clamp it to the 
                // intersection between current and next and enqueue it.
                if let Some(next) = maybe_next_projected {
                    if next.x < 0.0 {
                        // next is invisible; clamp it and enqueue the
                        // clamped version
                        let next_clamped = get_viewport_intersection_point(cur, next);
                        self.next = Option::Some(next_clamped);
                    } else {
                        // next is also visible; just enqueue it as-is
                        self.next = maybe_next_projected;
                    }
                }
                return Option::Some(Coord2D { x: cur.y, y: cur.z })
            }
        }
        return Option::None
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
