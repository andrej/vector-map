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
        input.map(move |CoordGeo { latitude: lat, longitude: lon } | {
            Coord3D { x: r * f64::cos(-lat) * f64::cos(lon),
                    y: r * f64::cos(-lat) * f64::sin(lon),
                    z: r * f64::sin(-lat) }
        })
    }
}

// impl Projection<Coord3D> for SphereProjection {
//     type To = CoordGeo;
//     fn project(&self, Coord3D{ x, y, z}: &Coord3D) -> CoordGeo {
//         let r = 1.0;
//         CoordGeo {
//             latitude: f64::asin(z / r),
//             longitude: f64::atan2(*y, *x)
//         }
//     }
// }

fn project_coord3d_to_coordgeo(Coord3D{ x, y, z}: &Coord3D) -> CoordGeo {
    let r = 1.0;
    //let longitude = f64::signum(*y) * (f64::atan(f64::abs(*y)/f64::abs(*x)) + if *x < 0.0 { f64::to_radians(90.0) } else { 0.0 } );
    CoordGeo {
        latitude: f64::asin(y / r),
        longitude: f64::atan2(*x, *z),
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
        let lat = -lat; // FIXME
        let lon = -lon; // FIXME
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
        let inp_xyz = xyz_proj.project(std::iter::once(*input)).next().unwrap();
        //console_log!("{} is {} in XYZ", input, inp_xyz);
        // Change basis using `center` as the equator/meridian
        let change_of_basis = [
            self.new_x_axis.x,  self.new_y_axis.x,  self.new_z_axis.x,
            self.new_x_axis.y,  self.new_y_axis.y,  self.new_z_axis.y,
            self.new_x_axis.z,  self.new_y_axis.z,  self.new_z_axis.z,
        ];
        let new_xyz = inline_matmul! ( 3 x 3 x 1 (change_of_basis, [inp_xyz.x, inp_xyz.y, inp_xyz.z]));
        //new_xyz[1] > 0.0
        // Now convert back to lat lon based off new basis
        let mut longitude = f64::signum(new_xyz[1]) * (f64::atan(f64::abs(new_xyz[1])/f64::abs(new_xyz[0])) + if new_xyz[0] < 0.0 { f64::to_radians(90.0) } else { 0.0 } );
        //console_log!("lon: {}", f64::to_degrees(longitude));
        if longitude > f64::to_radians(90.0) {
            longitude = f64::to_radians(90.0);
        } else if longitude < f64::to_radians(-90.0) {
            longitude = f64::to_radians(-90.0);
        }
        false
    }
}

pub struct OrthogonalProjection {
    x_axis: Coord3D,
    y_axis: Coord3D,
    z_axis: Coord3D
}

impl OrthogonalProjection {
    pub fn new(x_axis: Coord3D, y_axis: Coord3D) -> OrthogonalProjection {
        let x_axis = normalize(&x_axis);
        let y_axis = normalize(&y_axis);
        OrthogonalProjection {
            x_axis: x_axis,
            y_axis: y_axis,
            z_axis: cross_product(&x_axis, &y_axis)
        }
        //OrthogonalProjection {
        //    x_axis: Coord3D { x: 0.0, y: 1.0, z: 0.0 },
        //    y_axis: Coord3D { x: 0.0, y: 0.0, z: 1.0 },
        //    z_axis: Coord3D { x: 1.0, y: 0.0, z: 0.0 },
        //}
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
        let normal = Coord3D {
            x: f64::cos(longitude) * f64::cos(latitude),
            y: f64::sin(longitude) * f64::cos(latitude),
            z: f64::sin(latitude)
        };
        OrthogonalProjection::new_from_normal(normal)
    }

    fn project_single_with_depth(&self, input: Coord3D) -> Coord3D {
        Coord3D { 
            x: dot_product(&input, &self.x_axis), 
            y: dot_product(&input, &self.y_axis),
            z: dot_product(&input, &self.z_axis)
        }
    }
}

impl Projection<Coord3D> for OrthogonalProjection {
    type To = Coord2D;

    fn project(&self, input: impl Iterator<Item=Coord3D>) -> impl Iterator<Item=Coord2D> {
        let mut last = Option::<Coord3D>::None;
        input.filter_map(move |coord| {
            let projected = self.project_single_with_depth(coord); 
            let this_rev = project_coord3d_to_coordgeo(&projected);
            if projected.z >= 0.0 {
                last = Option::Some(projected);
                Option::Some(Coord2D { x: projected.x, y: projected.y })
            } else {
                if let Some(last_coord) = last {
                    // The previous coordinate was visible, but this one is invisible (back side of the globe);
                    // this means there is some point between the last point and this one that is at the edge
                    // of the globe. We should clamp this coordinate to that last visible position.
                    let this_rev = project_coord3d_to_coordgeo(&projected);
                    let last_rev = project_coord3d_to_coordgeo(&last_coord);
                    let lat_slope = (this_rev.latitude - last_rev.latitude) / (this_rev.longitude - last_rev.longitude);
                    let lon = f64::signum(last_rev.longitude) * f64::to_radians(90.0);
                    let edge_rev = CoordGeo {
                        longitude: lon,
                        latitude: last_rev.latitude + lat_slope * (lon - last_rev.longitude)
                    };
                    //console_log!("last: {},  this: {}, edge: {}", last_rev, this_rev, edge_rev);
                    let sphere_proj = SphereProjection;
                    let edge = sphere_proj.project(std::iter::once(edge_rev)).next().unwrap();
                    last = Option::None;
                    let ux = Coord3D { x: 0.0, y: 1.0, z: 0.0 };
                    let uy = Coord3D { x: 0.0, y: 0.0, z: -1.0 };  // FIXME why -1 Z axis??
                    Option::Some(Coord2D {
                        x: dot_product(&edge, &ux),
                        y: dot_product(&edge, &uy)
                    })
                } else {
                    Option::None
                }
            }
        })
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
