use rayon::prelude::*;
use pyo3::prelude::*;
use std::f64::consts::{TAU, FRAC_PI_2};
use rust_roche::{
    x_l1,
    strinit,
    stradv,
    Vec3,
    Point,
    Etype,
    ingress_egress,
    Star,
    set_earth_iangle,
};
use crate::blink;

#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct Brightspot {
    // Roche lobe filling donor star
    complex: bool, // whether to use the complex model
    q: f64, // mass ratio M2/M1
    rd: f64, // disc radius
    az: f64, // BS azimuth
    frac: f64, // isotropic fraction
    scale: f64, // scale 
    exp1: f64, // exponent 1
    exp2: f64, // exponent 2
    tilt: f64, 
    yaw: f64,
    x: f64, // x position of BS
    y: f64, // y position of BS
    nspot: usize, // number of elements in the bright spot
    normalisation: f64, // normalisation factor for the flux 
    grid: Vec<Point>, // grid points on the bright spot
    xl1: f64, // distance from primary to L1 point (in units of separation)
}

#[pymethods]
impl Brightspot {
    #[new]
    #[pyo3(signature=(q, rd, az, frac, scale, exp1=None, exp2=None, tilt=None, yaw=None, nspot=None),
        text_signature="(q, rd, az, frac, scale, exp1=2, exp2=1, tilt=90, yaw=0, nspot=200)")]
    pub fn new(
        q: f64, rd: f64, az: f64, 
        frac: f64, scale: f64, 
        exp1: Option<f64>, exp2: Option<f64>, 
        tilt: Option<f64>, yaw: Option<f64>, 
        nspot: Option<usize>) -> Self {
        let xl1 = x_l1(q);
        let (x, y) = Self::spot_position(q, rd);

        // are we using the complex model?
        let complex = exp1.is_some() || exp2.is_some() || tilt.is_some() || yaw.is_some();
        let exp1 = exp1.unwrap_or(2.0);
        let exp2 = exp2.unwrap_or(1.0);
        let tilt = tilt.unwrap_or(90.0);
        let yaw = yaw.unwrap_or(0.0);
        let nspot = nspot.unwrap_or(200);

        let mut spot = Brightspot {
            complex,
            q,
            rd,
            az,
            frac,
            scale,
            exp1,
            exp2,
            tilt,
            yaw,
            x: x, 
            y: y, 
            nspot,
            normalisation: -1.0, // -1 indicates not calculated yet
            grid: Vec::with_capacity(nspot * 2), // factor of 2 is because we have isotropic and beamed components
            xl1: xl1,
        };
        if spot.complex{
            spot.update_grid(90.0); // default inclination of 90 degrees for initial normalisation
        }
        spot
    }
    
    /*
     Determines location of bright-spot from mass ratio and radius
     of disc assuming that it is located on free-particle trajectory
     at edge of disc.
     Calculates X,Y position of spot in units of XL1
    */
    #[staticmethod]
    pub fn spot_position(q: f64, rd: f64) -> (f64, f64) {
        // calculate the position of the bright spot
        let xl1 = x_l1(q);
        let rtest = rd * xl1;
        let mut r: Vec3;
        let mut v: Vec3;
        let acc = 1.0e-10;
        let smax = 1.0e-3;
        (r, v) = strinit(q);
        stradv(q, &mut r, &mut v, rtest, acc, smax);

        (r.x, r.y)
    }

    pub fn tweak(
        &mut self, q: f64, rd: f64, az: f64, 
        frac: f64, scale: f64, 
        exp1: Option<f64>, exp2: Option<f64>, 
        tilt: Option<f64>, yaw: Option<f64>){

        if (q - self.q).abs() < 1e-6 || (rd - self.rd).abs() < 1e-6 {
            self.q = q;
            self.rd = rd;
            let (x, y) = Self::spot_position(self.q, self.rd);
            self.x = x;
            self.y = y;
            // we've changed location of the spot, so...
            // reset normalisation
            self.normalisation = -1.0;
            // clear grid
            self.grid.clear();
            if (q - self.q).abs() < 1e-6 {
                // recalculate xl1
                self.xl1 = x_l1(q);
            }
        }
        self.az = az;
        self.frac = frac;
        self.scale = scale;
        self.exp1 = exp1.unwrap_or(self.exp1);
        self.exp2 = exp2.unwrap_or(self.exp2);
        self.tilt = tilt.unwrap_or(self.tilt);
        self.yaw = yaw.unwrap_or(self.yaw);
    }

    pub fn update_grid(&mut self, iangle: f64) {
        if !self.complex {
            // we don't need a grid for the simple model
            return;
        }
        // angles in radians
        let theta = TAU * self.az / 360.0;
        let alpha = TAU * self.yaw / 360.0;
        let tilt_spot = TAU * self.tilt / 360.0;

        // the direction of the bright spot line is set by az, but the beaming direction adds in yaw as well
        let bspot: Vec3 = Vec3::new(self.x, self.y, 0.0);
        let bvec: Vec3 = Vec3::new(theta.cos(), theta.sin(), 0.0);
        let pvec: Vec3 = Vec3::new(0.0, 0.0, 1.0);
        let tvec: Vec3 = Vec3::new(
            tilt_spot.sin() * (theta + alpha).sin(),
            -tilt_spot.sin() * (theta + alpha).cos(),
            tilt_spot.cos()
        );

        // find phase of brightest flux
        let tan_maxphi = 1.0 / (theta + alpha).tan();
        let mut maxphi = tan_maxphi.atan() / TAU; // lies between -1/2 and 1/2
        if maxphi < 0.0 {
            maxphi += 1.0;
        }
        let earthmax: Vec3 = set_earth_iangle(iangle, maxphi);
        // projected direction of the tilted spot at this phase
        let maxproj = tvec.dot(&earthmax);

        // now we find the length of the bright spot in units of scale length
        // step 1: location of brightest point (scaled by scale length) is BMAX
        let bmax: f64 = (self.exp1/self.exp2).powf(1.0/self.exp2);
        // find the max flux of the spot at this position
        let spot_max = bmax.powf(self.exp1) * (-bmax.powf(self.exp2)).exp();

        // step2: find the end of the BS strip
        // our target flux is 1/1000th of the max flux
        let mut curr_flux = spot_max;
        let mut ppos = bmax;
        while curr_flux > spot_max/1000.0 {
            ppos += bmax/10.0;
            curr_flux = ppos.powf(self.exp1) * (-ppos.powf(self.exp2)).exp();
        }

        // now calculate end position in scale lengths
        // (limit of 20 scale lengths past max or min)
        let sfac: f64 = (bmax + ppos).min(20.0 + bmax);

        // the scaling of nspot below ensures we have at least 50 points between
        // start and location of maximum flux
        let mut nspot = ((50.0 * sfac/bmax).ceil() as usize).max(200);

        // the above calc produces unreasonable spot sizes if BMAX is close to
        // the start of the spot, so we also enforce a maximum spot size of 1000
        nspot = nspot.min(1000);
        // finally set the number of points in the spot
        self.nspot = nspot;

        // factor of 2 is because we have isotropic and beamed components
        let dummy_point = Point::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0, 1.0, Etype::new());
        self.grid = vec![dummy_point; nspot * 2];

        // now we can setup the frid points
        let area = sfac * self.scale * self.scale / (nspot-1) as f64;
        
        let mut posn: Vec3;
        // maximum projected area of tilted spot
        let mumax_tilted: f64 = tvec.dot(&earthmax); 
        let mumax_parallel: f64 = pvec.dot(&earthmax);
        let mut maxflux: f64 = 0.0; // maximum flux of the spot
        for i in 0..nspot {
            // println!("Setting grid point {}/{}", i, nspot);
            // spot position along the strip in scale lengths
            let dist = sfac * i as f64 / (nspot-1) as f64;
            posn = bspot + self.scale * (dist - bmax) * bvec;

            // ingress and egress phases
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

            // Factor here is adjusted to equal 1 at its peak
            let bright = (dist/bmax).powf(self.exp1) * (self.exp1/self.exp2 - dist.powf(self.exp2)).exp();

            // now add this point to the tilted strip
            let gravity = 1.0; // we don't use gravity for the bright spot, but we need to set it to something non-zero for the limb darkening
            self.grid.insert(i, Point::new(posn, tvec, area, gravity, eclipses.clone()));
            self.grid[i].set_flux((bright * (1.0 - self.frac) * area) as f32);
            if mumax_tilted > 0.0{
                maxflux += mumax_tilted * self.grid[i].flux as f64;
            }

            // the parallel strip
            self.grid.insert(i+nspot, Point::new(posn, pvec, area, gravity, eclipses.clone()));
            self.grid[i+nspot].set_flux((bright * self.frac * area) as f32);
            if mumax_parallel > 0.0 {
                maxflux += maxproj * self.grid[i+nspot].flux as f64;
            }
        }
        self.normalisation = maxflux;
    }

    pub fn get_tangent(&self) -> f64 {
        let mut alpha = (self.x).atan2(self.y);
        if alpha < 0.0 {
            // alpha is between -pi and pi
            // if negative, the spot is slightly behind the disc (alpha > pi)
            alpha  =  FRAC_PI_2 - alpha;
        } 
        alpha + TAU/4.0
    }

    ///
    /// Computes light curve of spot assuming that it is a linear feature
    /// with an intensity that varies as (X/LS)**N*EXP(-X/LS) where X
    /// is measured from one end. The peak at X = N*LS is positioned on the
    /// at XS, YS and the line is rotated relative to X,Y axes by AZ
    /// Normalised to equal 1 at maximum if no eclipse.
    ///
    pub fn simple_flux(&self, q: f64, phi: f64, incl: f64) -> f64 {
        let ept: u32 = 2; // ept cannot be changed at will but must be 1,2 or 3
        let acc: f64 = 1.0e-3;
        let tot:f64 = 8.0;
        
        // some time saving variables]
        let earth: Vec3 = set_earth_iangle(incl, phi);
        let cphi = (TAU * phi).cos();
        let sphi = (TAU * phi).sin();
        let raz  = self.az * TAU / 360.0;
        let caz = raz.cos();
        let saz = raz.sin();

        // ept is exponent in x**ept*exp(-x)
        // integration is carried out from x=0 to x=tot
        let mut out = 2.0 - (tot * (tot + 2.0) + 2.0) * (-tot).exp();
        out = match ept {
            1 => 2.0 * (1.0 - (-tot).exp()),
            2 => out,
            3 => 3.0 * out - tot.powf(3.0) * (-tot).exp(),
            _ => panic!("ept must be 1, 2 or 3")
        };
        let x1 = self.x - ept as f64 * self.scale * caz;
        let y1 = self.y - ept as f64 * self.scale * saz;
        let dx = tot * self.scale * caz;
        let dy = tot * self.scale * saz;
        let proj = (saz * cphi + caz * sphi).max(0.0);

        // are ends occulted?
        let mut x: Vec3 = Vec3::new(x1, y1, 0.0);
        let p1: bool;
        let p2: bool;
        p1 = blink(q, self.xl1, &x, &earth, 0.05);
        x.set(x1 + dx, y1 + dy, 0.0);
        p2 = blink(q, self.xl1, &x, &earth, 0.05);

        if p1 && p2 {
            // total eclipse
            return 0.0;
        } else if !p1 && !p2 {
            // no eclipse
            return self.frac + (1.0 - self.frac) * proj;
        }
        // partial eclipse, so do calculation
        let mut a: f64;
        let mut a1: f64;
        let mut a2: f64;
        // a1 is the occulted, end. Find it...
        if p1 {
            a1 = 0.0;
            a2 = 1.0;
        } else {
            a1 = 1.0;
            a2 = 0.0;
        }
        loop {
            a = (a1 + a2) / 2.0;
            x.set(x1 + a * dx, y1 + a * dy, 0.0);
            if blink(q, self.xl1, &x, &earth, 0.05) {
                a1 = a;
            } else {
                a2 = a;
            }
            if (a2 - a1).abs() <= acc {
                break;
            }
        }
        a = tot * (a1 + a2) / 2.0;
        let mut frac = 2.0 - (a * (a + 2.0) + 2.0) * (-a).exp();
        frac = match ept {
            1 => 2.0 * (1.0 - (-a).exp()),
            2 => frac,
            3 => 3.0 * frac - a.powf(3.0) * (-a).exp(),
            _ => panic!("ept must be 1, 2 or 3")
        };

        let bflux = match p1 {
            true => (self.frac + (1.0 - self.frac) * proj) * (1.0 - frac / out),
            false => (self.frac + (1.0 - self.frac) * proj) * frac / out,
        };
        bflux
    }

    ////
    //// Computes flux of bright spot at given phase and inclination.
    //// This should not normally be called directly, but is used in calcflux
    //// We assume the grid is initialised.
    fn calcflux_at_point(&self, q: f64, phi: f64, incl: f64) -> f64 {

        // do simple BS calculation if appropriate
        if !self.complex {
            //println!("Using simple bright spot model");
            return self.simple_flux(q, phi, incl);
        }
        //println!("Using complex bright spot model");

        let theta = TAU * self.az / 360.0;
        let alpha = TAU * self.yaw / 360.0;
        // find phase of brightest flux
        let tan_maxphi = 1.0 / (theta + alpha).tan();
        let mut maxphi = tan_maxphi.atan() / TAU; // lies between -1/2 and 1/2
        if maxphi < 0.0 {
            maxphi += 1.0;
        }
        let earthmax: Vec3 = set_earth_iangle(incl, maxphi);
        // first element is tilted strip
        let maxproj = self.grid[0].direction.dot(&earthmax);

        // setup variables
        let earth: Vec3 = set_earth_iangle(incl, phi);
        // parallel computation seems slower here
        
        let mut sum = 0.0 as f32;
        for i in 0..self.grid.len() {
            let mut flux: f32 = 0.0;
            let p: &Point = &self.grid[i];
            let mu: f32 = earth.dot(&p.direction) as f32;
            if mu > 0.0 && p.is_visible(phi) {
                flux = match i < self.nspot {
                    // tilted strip
                    true => p.flux * mu,
                    // parallel strip
                    false => p.flux * maxproj as f32,
                }
            }
            sum += flux;
        }
         
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