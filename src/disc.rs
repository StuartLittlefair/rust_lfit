use rayon::prelude::*;
use pyo3::prelude::*;
use std::f64::consts::TAU;
use rust_roche::{
    x_l1,
    Vec3,
    Point,
    Etype,
    ingress_egress,
    Star,
    set_earth_iangle,
};

#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct Disc {
    q: f64, // mass ratio M2/M1
    rwd: f64, // inner radius in units of xl1
    rout: f64, // outer radius in units xl1
    exp: f64, // exponent of the power law radial brightness distribution
    normalisation: f64, // normalisation factor for the flux
    size: usize, // number of grid points
    grid: Vec<Point>, // grid points on the disc
    xl1: f64, // separation in units of xl1
}

#[pymethods]
impl Disc {
   
    /// Constructor for the disc
    /// /param q: mass ratio M2/M1
    /// /param rwd: inner radius in units of separation (converted internally to xl1)
    /// /param rout: outer radius in units of separation (converted internally to xl1)
    /// /param exp: exponent of the power law radial brightness distribution
    /// /param size: number of grid points
     #[new]
    #[pyo3(signature=(q, rwd, rout, exp, size=None),
        text_signature="(q, rin, rout, exp, size=1000)")]
    pub fn new(q: f64, rwd: f64, rout: f64, exp: f64, size: Option<usize>) -> Self {
        let size = size.unwrap_or(1000);
        let grid = Vec::with_capacity(size);
        let mut disc = Disc {
            q,
            rwd,
            rout,
            exp,
            normalisation: -1.0, // -1 indicates not calculated yet
            size,
            grid,
            xl1: 0.0, // will be set by tweak
        };
        // sets radii in units of xl1, not separation
        disc.tweak(q, rwd, rout, exp);
        disc
    }

    pub fn params_changed(&self, rwd: f64, rout: f64, exp: f64) -> bool {
        // check if the parameters have changed since the last grid calculation
        let radius_changed = (self.rwd - rwd).abs() > 1e-6 || (self.rout - rout).abs() > 1e-6;
        let exp_changed = (self.exp - exp).abs() > 1e-6;
        radius_changed || exp_changed 
    }

    pub fn tweak(&mut self, q: f64, rwd: f64, rout: f64, exp: f64){
        let xl1 = x_l1(q);
        self.xl1 = xl1;
        let rwd_xl1 = rwd * xl1;
        let rout_xl1 = rout * xl1;  
        if !self.params_changed(rwd_xl1, rout_xl1, exp) {
            return;
        }
        // we have changed, update the parameters
        self.q = q;
        self.rwd = rwd_xl1;
        self.rout = rout_xl1;
        self.exp = exp;
        // clear the grid and reset normalisation
        self.grid.clear();
        self.normalisation = -1.0;
    }

    pub fn update_grid(&mut self, iangle: f64) {
        // let's create the grid
        // we need more elements at smaller radii, since that's where most flux comes
        // from, but total number of elements must equal size.
        // if ntheta = 2*nrad - i
        // total number of tiles = 0.5*nrad*(3*nrad+1)
        // nrad = (-1 + sqrt(1 + 24*ntiles) ) / 6
        let nrad = (-1.0 + (1.0 + 24.0 * self.size as f64).sqrt()) / 6.0;
        let nrad = nrad as usize;
        // calculate the grid points
        let delta_r = (self.rout - self.rwd) / nrad as f64;

        // max flux at phase 0.25
        let earth_at_max: Vec3 = set_earth_iangle(iangle, 0.25);
        let mut maxflux: f64 = 0.0;
        self.grid.clear();

        for i in 0..nrad {
            let r = self.rwd + (self.rout - self.rwd) * (i as f64 + 1.0) / nrad as f64;
            let ntheta = 2 * nrad - i;
            for j in 0..ntheta {
                let theta = TAU * j as f64 / ntheta as f64;
                let x = r * theta.cos();
                let y = r * theta.sin();
                let posn: Vec3 = Vec3::new(x, y, 0.0);
                let dirn: Vec3 = Vec3::new(0.0, 0.0, 1.0);

                // area is r*dtheta*dr
                let area = r * delta_r * TAU / ntheta as f64;

                // nominal reference flux is used here. The disc's light will later be normalised
                let flux: f32 = (r/self.rout).powf(-self.exp) as f32;

                let mut eclipses: Etype = Etype::new();
                let mut ingress: f64 = 0.0;
                let mut egress: f64 = 0.0;
                // sometimes dies with linmin error: allow this to bubble up (accuracy must be 1.0-e7 to avoid next line hanging indefinitely)
                //q: f64, star: Star, spin: f64, ffac: f64, iangle: f64, delta: f64, r: &Vec3, ingress: &mut f64, egress: &mut f64
                let status: bool = ingress_egress(
                    self.q, Star::Secondary, 1.0, 1.0, iangle, 1.0e-8, &posn, &mut ingress, &mut egress
                );
                if status {
                    eclipses.push((ingress, egress));
                }

                // we don't use gravity for the disc, so set dummy value
                let gravity: f64 = 1.0; 
                let mut p = Point::new(posn, dirn, area, gravity, eclipses);
                p.set_flux(flux);   

                // calculate the maximum flux for normalisation
                let mu = dirn.dot(&earth_at_max);
                if mu > 0.0 && p.is_visible(0.25) {
                    maxflux += mu * p.flux as f64 * p.area as f64;
                }
                // grid takes ownership of p
                self.grid.push(p);

            }
        }
        self.normalisation = maxflux;
    }

    ////
    //// Computes flux of disc at given phase and inclination.
    //// This should not normally be called directly, but is used in calcflux
    //// We assume the grid is initialised.
    fn calcflux_at_point(&self, _q: f64, phi: f64, incl: f64) -> f64 {
        let earth: Vec3 = set_earth_iangle(incl, phi);
        // use rayon to parallelise the flux calculation across the grid points
        let sum: f32 = (0..self.grid.len()).into_par_iter().map(|i| {
            let mut flux: f32 = 0.0;
            let p: &Point = &self.grid[i];
            if p.is_visible(phi) {
                let mu: f32 = earth.dot(&p.direction) as f32;
                if mu > 0.0 {
                    flux = p.flux * p.area * mu;
                }
            }
            flux as f32
        }).sum();
        sum as f64 / self.normalisation
    }

    fn calcflux_over_bin(&self, q: f64, phi: f64, width: f64, incl: f64) -> f64 {
        // integrates over bin of finite phase width, width using trapezoidal integration
        let phi1: f64 = phi - width/2.0;
        let mut rflux: f64 = 0.0;
        let nphi: usize = 5; // number of points to sample across the bin
        for i in 0..nphi {
            let phi: f64 = phi1 + width * (i as f64) / (nphi-1) as f64;
            if i==0 || i==nphi-1 {
                rflux += self.calcflux_at_point(q, phi, incl) / 2.0;
            } else {
                rflux += self.calcflux_at_point(q, phi, incl);
            }
        }
        rflux / (nphi-1) as f64
    }

    #[pyo3(signature=(q, incl, phases, widths=None), text_signature="(q, incl, phases, widths=None)")]
    pub fn calcflux(&mut self, q: f64, incl: f64, phases: Vec<f64>, widths: Option<Vec<f64>>) -> PyResult<Vec<f64>> {

        let n: usize = phases.len();
        self.update_grid(incl);

        // calculate flux at each phase in phases, for given q and incl
        let mut fluxes: Vec<f64> = vec![0.0; n];

        match widths{
            Some(w) => {
                fluxes
                .par_iter_mut()
                 .enumerate()
                 .for_each(|(i, flux)| {
                     *flux = self.calcflux_over_bin(q, phases[i], w[i], incl);
                 });
            },
            None => {
                fluxes
                .par_iter_mut()
                 .enumerate()
                 .for_each(|(i, flux)| {
                     *flux = self.calcflux_at_point(q, phases[i], incl);
                 });
            }
        };
        Ok(fluxes)
    }
}
