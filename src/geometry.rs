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

pub trait Projection<From, To> {
    fn project(&self, input: &From) -> To;
}

pub struct SphereProjection;

impl Projection<CoordGeo, Coord3D> for SphereProjection {
    fn project(&self, input: &CoordGeo) -> Coord3D {
        let r = 1.0;
        let [lat, lon] = [input.latitude, input.longitude].map(f64::to_radians);
        Coord3D { x: r * f64::cos(-lat) * f64::cos(lon),
                  y: r * f64::cos(-lat) * f64::sin(lon),
                  z: r * f64::sin(-lat) }
    }
}

impl Projection<Coord3D, CoordGeo> for SphereProjection {
    fn project(&self, Coord3D{ x, y, z}: &Coord3D) -> CoordGeo {
        let r = 1.0;
        CoordGeo {
            latitude: f64::asin(z / r),
            longitude: f64::atan2(*y, *x)
        }
    }
}


//fn matmul<const M: usize, const K: usize, const N: usize>(a: &[f64], b: &[f64]) -> [f64; M*N] {
//    let mut out = [0.0;M*N];
//    for row in 0..M {
//        for col in 0..N {
//            let mut acc = 0.0;
//            for k in 0..K {
//                acc += a[row*K+k] * b[k*K+col];
//            }
//            out[row*N+col] = acc;
//        }
//    }
//    out
//}

macro_rules! inline_matmul {
    ( $M:literal x $K:literal x $N:literal ( $a:expr, $b:expr ) ) => {
        {
            let a = $a;
            let b = $b;
            let mut c = [0.0;$M*$N];
            for row in 0..$M {
                for col in 0..$N {
                    let mut acc = 0.0;
                    for k in 0..$K {
                        acc += a[row*$K+k] * b[k*$N+col];
                    }
                    c[row*$N+col] = acc;
                }
            };
            c
        }
    }
}

fn matmul() {
    let a = [0.0; 9];
    let b = [0.0; 9];
    let x = inline_matmul!(3 x 3 x 1 (a, b));
}

//fn matmul_3x3x3(a: &[f64; 9], b: &[f64; 9]) -> &[f64; 9] {
//    [
//        
//    ]
//}

fn rotate(input: &Coord3D, &Coord3D { x: x, y: y, z: z }: &Coord3D, angle: f64) -> Coord3D {
    let sin = f64::sin(angle);
    let cos = f64::cos(angle);
    let [x, y, z] = inline_matmul!(3 x 3 x 1 ( [
        x*x*(1.0-cos)+cos,    x*y*(1.0-cos)-z*sin,  x*z*(1.0-cos)+y*sin,
        x*y*(1.0-cos)+z*sin,  y*y*(1.0-cos)+cos,    y*z*(1.0-cos)-x*sin,   
        x*z*(1.0-cos)-y*sin,  y*z*(1.0-cos)+x*sin,  z*z*(1.0-cos)+cos
    ], [input.x, input.y, input.z] ));
    Coord3D { x: x, y: y, z: z }
}

// This struct contains everything needed to cull points based on what would be
// invisible on an orthogonally projected sphere looking straight at a defined
// center point, i.e. equator and meridian. Everything outside of 180 deg 
// longitude of the center point in the new coordinate system centered around
// `center` will be culled (latitude <= 90 deg is assumed)
pub struct OrthogonalSphereCulling {
    new_x_axis: Coord3D,  // calculated based off rotation from (0,0) to center
    new_y_axis: Coord3D,
    new_z_axis: Coord3D
}

impl OrthogonalSphereCulling {
    pub fn new(center: CoordGeo) -> Self {
        let CoordGeo { latitude: lat, longitude: lon } = center;
        // New basis
        // There are an infinite number of basis vectors we could choose that
        // have `center` as the intersection of meridian and equator (they 
        // could be rotated around it arbitrarily). We choose one by starting
        // with the basis vectors x=(1,0,0), y=(0,1,0), z=(0,0,1) and rotating
        // them by `latitude` about the Y axis first, which places the equator 
        // going through that latitude, and then rotating everything about the
        // NEW Z axis by `longitude` so the meridian goes through that point.
        // It's critical that in the second step, we use the new (rotated) Z 
        // axis.
        let lat = f64::to_radians(lat);
        let lon = f64::to_radians(lon);
        let new_z_axis = 
            rotate(&Coord3D { x: 0.0, y: 0.0, z: 1.0 },
                   &Coord3D { x: 0.0, y: 1.0, z: 0.0 },
                   lat);
        let new_y_axis =
                rotate(
                    &Coord3D { x: 0.0, y: 1.0, z: 0.0},
                    &new_z_axis,
                    lon
                );
        let new_x_axis =
                rotate(
                    &rotate (
                        &Coord3D { x: 1.0, y: 0.0, z: 0.0},
                        &Coord3D { x: 0.0, y: 1.0, z: 0.0},
                        lat
                    ),
                    &new_z_axis,
                    lon
                );
                
        Self {
            new_x_axis: new_x_axis,
            new_y_axis: new_y_axis,
            new_z_axis: new_z_axis
        }
    }

    pub fn cull(&self, input: &CoordGeo) -> bool {
        let xyz_proj = SphereProjection;
        // Project into XYZ coordinates
        let inp_xyz = xyz_proj.project(input);
        //console_log!("{} is {} in XYZ", input, inp_xyz);
        // Change basis using `center` as the equator/meridian
        let change_of_basis = [
            self.new_x_axis.x,  self.new_y_axis.x,  self.new_z_axis.x,
            self.new_x_axis.y,  self.new_y_axis.y,  self.new_z_axis.y,
            self.new_x_axis.z,  self.new_y_axis.z,  self.new_z_axis.z,
        ];
        let new_xyz = inline_matmul! ( 3 x 3 x 1 (change_of_basis, [inp_xyz.x, inp_xyz.y, inp_xyz.z]));
        new_xyz[1] > 0.0
        // Now convert back to lat lon based off new basis
        //let longitude = f64::atan(new_xyz[1]/new_xyz[0]);
        //console_log!("orig lon: {} new lon: {}", input.longitude, f64::to_degrees(longitude));
        //longitude > 0.0 //f64::to_radians(45.0)
    }
}

pub struct OrthogonalProjection {
    x_axis: Coord3D,
    y_axis: Coord3D
}

impl OrthogonalProjection {
    pub fn new(x_axis: Coord3D, y_axis: Coord3D) -> OrthogonalProjection {
        OrthogonalProjection {
            x_axis: normalize(&x_axis),
            y_axis: normalize(&y_axis)
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
    pub fn cull(&self, input: &Coord3D, distance: f64) -> bool {
        let normal = cross_product(&self.x_axis, &self.y_axis); 
            // TODO: optimize this so we don't recaculate normal for every. single. coordinate.
        //web_sys::console::log_1(&dot_product(input, &normal).to_string().into());
        dot_product(input, &normal) < distance 
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


pub fn project<'a, A, B>(proj: &'a impl Projection<A, B>, line: impl Iterator<Item=A> + 'a) -> impl Iterator<Item=B> + 'a { 
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
pub fn project_generic<'a, A, B, I, P>(proj: &'a P, line: I) -> impl Iterator<Item=B> + 'a
where 
    I: Iterator<Item=A> + 'a,
    P: Projection<A, B> + 'a
{ 
    line.map(move |point| { proj.project(&point) })
}
