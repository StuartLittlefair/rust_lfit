use rayon::prelude::*;
use pyo3::prelude::*;
use std::f64::consts::{PI, TAU};
use roche::{Vec3, Star, RocheContext, Etype, Point, planck, set_earth_iangle};
use roche::errors::RocheError;

//use numpy::{PyReadonlyArray1, IntoPyArray, PyArray1};

pub fn eggleton(q: f32) -> f32 {
    let q13 = q.powf(1.0/3.0);
    0.49 * q13.powf(2.0) / (0.6 * q13.powf(2.0) + (1.0 + q13).ln())
}

#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
    /// Create a new Donor object.
    /// \param q: mass ratio M2/M1
    /// \param ulimb: limb darkening coefficient
    /// \param nlat: number of latitude points to use in the grid. If
    ///              nlat=0, the grid will not be calculated and the flux 
    ///              will be calculated using the approximation for the 
    ///              ellipsoidal variation amplitude found in Kopal 1959.
pub struct Donor {
    // Roche lobe filling donor star
    q: f32, // mass ratio M2/M1
    beta: f32, // beta = gravity darkening exponent
    ulimb: f32, // limb darkening coefficient
    nlat: usize, // number of latitude points
    grid: Vec<Point>, // grid points on the donor star
    gmin: f32, // minimum gravity on the donor star (at back face)
    normalisation: f32, // normalisation factor for the flux
    approx: bool, // whether to use the approximation for the ellipsoidal variation amplitude
}

#[pymethods]
impl Donor {
    #[new]
    /// Create a new Donor object.
    /// \param q: mass ratio M2/M1
    /// \param ulimb: limb darkening coefficient
    /// \param nlat: number of latitude points to use in the grid. If
    /// nlat=0, the grid will not be calculated and the flux will be calculated
    // using the approximation for the ellipsoidal variation amplitude found in Kopal 1959.
    pub fn new(q: f32, ulimb: f32, nlat: usize) -> Self {
        let mut approx: bool = false;
        let mut size = nlat;
        if nlat == 0{
            approx = true;
            size = 2; // dummy value, we won't actually use the grid if approx is true
        }
        let grid_size = Donor::calc_grid_size(size);
        let donor = Donor {
            q,
            beta: 0.08,
            ulimb,
            nlat: size,
            grid: Vec::with_capacity(grid_size),
            gmin: 0.0,
            normalisation: -1.0, // -1 indicates not calculated yet
            approx,
        };
        donor
    }

    pub fn kopal59(&self, phi: f64, incl: f64) -> f64 {
        // Kopal 1959, IV.2-37, for the ellipsoidal variation amplitude
        let r_a = eggleton(self.q);
        let q = 1.0/self.q ;
        let phi = phi as f32 + 0.5;
        let cosphi = (TAU as f32 * phi).cos();
        let cosphi2 = (2.0  * TAU as f32  * phi).cos();
        let cosphi3 = (3.0  * TAU as f32  * phi).cos();
        let cosphi4 = (4.0  * TAU as f32  * phi).cos();
        let sini = (incl as f32 * TAU as f32 / 360.0).sin();
        let u = self.ulimb as f32;
        let beta = self.beta as f32;
        let t1 = (15.0 + u) * (1.0 + beta) * r_a.powf(3.0) * (2.0+5.0*q) * (2.0-3.0*sini.powf(2.0)) / 60.0 / (3.0-u);
        let t2 = 9.0 * (1.0-u) * (3.0+beta) * r_a.powf(5.0) * q * (8.0 - 40.0*sini.powf(2.0) + 35.0*sini.powf(4.0)) / 256.0/ (3.0-u);
        let t3 = cosphi * (15.0 * u *(2.0+beta) * r_a.powf(4.0) * q * (4.0*sini - 5.0*sini.powf(3.0))/ 32.0 / (3.0-u));
        let t4 = cosphi2 * (-3.0*(15.0+u) * (1.0+beta) * r_a.powf(3.0) * q * sini.powf(2.0) / 20.0 / (3.0-u) 
                        - 15.0 * (1.0-u) * (3.0+beta) *r_a.powf(5.0) * q * (6.0*sini.powf(2.0) - 7.0 * sini.powf(4.0))/64.0/(3.0-u));
        let t5 = cosphi3 * (-25.0*u*(2.0+beta)*r_a.powf(4.0) * q * sini.powf(3.0) / 32.0 / (3.0-u));
        let t6 = cosphi4 * (
            105.0 * (1.0-u) * (3.0+beta) * r_a.powf(5.0) * q * sini.powf(4.0)/256.0/(3.0-u)
        );
        (1.0 + t1 + t2 + t3 + t4 + t5 + t6) as f64
    }

    pub fn tweak(&mut self, q: f32){
        if (q - self.q).abs() < 1e-6 {
            // no change
            return;
        }
        self.q = q;
        // we've changed the roche lobe, so...
        // clear the grid and reset normalisation
        self.grid.clear();
        self.normalisation = -1.0;
    }

    #[staticmethod]
    pub fn calc_grid_size(nlat: usize) -> usize {
        // Calculate the total number of grid points based on nlat
        let mut count = 0;
        let dtheta: f64 = PI / nlat as f64; // latitude step size
        for i in 0..nlat {
            let theta = PI * (i as f64 + 0.5) / nlat as f64; // latitude angle
            let dphi = dtheta / theta.sin(); // longitude step size to maintain approximately equal area
            let mut nface: usize = (2.0 * PI / dphi) as usize; // number of longitude points at this latitude
            // min of 16
            nface = nface.max(16);
            count += nface;
        }
        count
    }

    pub fn update_grid(&mut self) -> Result<(), RocheError> {
        // clear the grid!
        self.grid.clear();

        // assume synchronous rotation
        let rochectx = RocheContext::new(self.q as f64, Star::Secondary, 1.0)?;

        // use filling factor of 1
        let rref: f64;
        let pref: f64;
        (rref, pref) = rochectx.ref_sphere(1.0)?;

        // find the back of the secondary star
        let mut dirn: Vec3 = Vec3::new(1.0, 0.0, 0.0);
        let acc: f64 = 1.0e-6;
        // find position, norm vector, radius, and gravity at the back face
        let mut pos: Vec3;
        let mut norm: Vec3;
        let mut rad: f64;
        let mut grav: f64;
        (_, _, _, grav) = rochectx.face(dirn, rref, pref, acc)?;
        self.gmin = grav as f32;

        // OK, let's tile the surface.
        let dtheta: f64 = PI/self.nlat as f64; // latitude step size
        let eclipses: Etype = Etype::new();
        for i in 0..self.nlat {
            let theta = PI * (i as f64 + 0.5) / self.nlat as f64; // latitude angle
            let sint = theta.sin();
            let cost = theta.cos();

            // variable number of longitude points at each latitude
            let nphi: usize = (TAU * sint / dtheta).max(16.0) as usize;
            let dphi: f64 = TAU / nphi as f64; // longitude step size

            for j in 0..nphi {
                let phi = TAU * j as f64 / nphi as f64; // longitude angle
                let sinp = phi.sin();
                let cosp = phi.cos();

                // calculate the position of the point on the Roche lobe
                // dirn points to this tile.
                dirn.set(cost, sint * cosp, sint * sinp);
                (pos, norm, rad, grav) = rochectx.face(dirn, rref, pref, acc)?;

                // we also need the element area, which is the circumference
                // of the roche lobe at this point, divided by the number
                // of theta steps, and multiplied by delta_x/cos(alpha),
                // where alpha is angle between element and x-axis
                // area needs to be multiplied by sep**2.0 to become physical
                let area = rad * rad * sint * dphi * dtheta / dirn.dot(&norm);

                // now set temperature of element, scaling for limb and gravity darkening
                let temp: f32 = 3000.0 * (grav as f32 / self.gmin).powf(self.beta);

                // flux, not accounting for limb darkening
                let flux: f64 =  planck(6565.0, temp as f64);
                // add the point to the grid
                // clone eclipses for each point.
                let mut p = Point::new(pos, norm, area, grav, eclipses.clone());
                p.set_flux(flux as f32);
                self.grid.push(p);
            }
        }
        Ok(())
    }

    fn initialise(& mut self, incl: f64) -> Result<(), RocheError> {
        if self.grid.is_empty() && !self.approx {
            // called before grid is calculated
            println!("Warning: calculating flux before grid is calculated. Call update_grid() first.");
            self.update_grid()?;
        }
        // set normalisation if not set
        if self.normalisation < 0.0 {
            // maximum flux is at phi = 0.75
            let maxphi: f64 = 0.75;
            let earth: Vec3 = set_earth_iangle(incl, maxphi);
            // use rayon to parallelise the flux calculation across the grid points
            let sum: f32 = (0..self.grid.len()).into_par_iter().map(|i| {
                let mut flux: f32 = 0.0;
                let p: &Point = &self.grid[i];
                let mu: f32 = earth.dot(&p.direction) as f32;
                if mu > 0.0 && p.is_visible(maxphi) {
                    flux = p.flux * (1.0 - self.ulimb + mu * self.ulimb) as f32;
                    flux = flux * p.area * mu 
                }
                flux as f32
            }).sum();
            self.normalisation = sum;
        }
        Ok(())
    }

    fn calcflux_over_bin(&self, phi: f64, width: f64, incl: f64) -> f64 {
        // integrates over bin of finite phase width, width using trapezoidal integration
        let phi1: f64 = phi - width/2.0;
        let mut rflux: f64 = 0.0;
        let nphi: usize = 5; // number of points to sample across the bin
        for i in 0..nphi {
            let phi: f64 = phi1 + width * (i as f64) / (nphi-1) as f64;
            if i==0 || i==nphi-1 {
                rflux += self.calcflux_at_point(phi, incl) / 2.0;
            } else {
                rflux += self.calcflux_at_point(phi, incl);
            }
        }
        rflux / (nphi-1) as f64
    }

    fn calcflux_at_point(&self, phi: f64, incl: f64) -> f64 {
        if self.approx {
            return self.kopal59(phi, incl);
        }
        let earth: Vec3 = set_earth_iangle(incl, phi);
        // use rayon to parallelise the flux calculation across the grid points
        let sum: f64 = (0..self.grid.len()).into_par_iter().map(|i| {
            let mut flux: f32 = 0.0;
            let p: &Point = &self.grid[i];
            let mu: f32 = earth.dot(&p.direction) as f32;
            if (mu > 0.0) && p.is_visible(phi) {
                flux = p.flux * (1.0 - self.ulimb + mu * self.ulimb) as f32;
                flux = flux * p.area * mu 
            }
            flux as f64
        }).sum();
        sum / self.normalisation as f64
    }

    #[pyo3(signature=(_q, incl, phases, widths=None), text_signature="(q, incl, phases, widths=None)")]
    pub fn calcflux(&mut self, _q: f64, incl: f64, phases: Vec<f64>, widths: Option<Vec<f64>>) -> PyResult<Vec<f64>> {

        let n: usize = phases.len();

        // calculate flux at each phase in phases, for given q and incl
        let mut fluxes: Vec<f64> = vec![0.0; n];
        // painless if not needed, so do each time
        self.initialise(incl)?;

        match widths{
            Some(w) => {
                fluxes
                .par_iter_mut()
                 .enumerate()
                 .for_each(|(i, flux)| {
                     *flux = self.calcflux_over_bin(phases[i], w[i], incl);
                 });
            },
            None => {
                fluxes
                .par_iter_mut()
                 .enumerate()
                 .for_each(|(i, flux)| {
                     *flux = self.calcflux_at_point(phases[i], incl);
                 });
            }
        };
        Ok(fluxes)
    }
}