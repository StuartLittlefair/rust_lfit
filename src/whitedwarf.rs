use rayon::prelude::*;
use pyo3::prelude::*;
use std::f64::consts::{PI, TAU};
use roche::{
    Vec3, 
    set_earth_iangle,
    x_l1,
};
use crate::blink;

#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct Whitedwarf {
    // Roche lobe filling donor star
    pub radius: f64,
    pub ulimb: f64,
    // for internal use only, stores q at which we last calculated visibility
    _qcalc: f64,
    // for internal use only, stores x_l1 at which we last calculated visibility 
    _xl1: f64,
}

#[pymethods]
impl Whitedwarf {
    #[new]
    pub fn new(radius: f64, ulimb: f64) -> Self {
        let wd = Whitedwarf {
            radius,
            ulimb,
            _qcalc: -1.0, // _qcalc
            _xl1: -1.0, // _xl1
        };
        wd
    }

    fn tweak(&mut self, radius: f64, ulimb: f64) {
        self.radius = radius;
        self.ulimb = ulimb;
    }

    /// Is WD eclipsed?
    /// Calculates if the WD is eclipsed, and if so, calculates the 
    /// circular projection of the donor star limb across the WD
    /// disc. Allows very fast computation of WD eclipse lightcurves
    /// \param q the mass ratio = M2/M1
    /// \param phi the orbital phase
    /// \param incl  the inclination in degrees
    /// \return status - 0 for eclipsed, 1 for no eclipse, 2 for total eclipse
    /// \return xc - x coordinate of circular shadow of RL
    /// \return yc - y coordinate of circular shadow of RL
    /// \return rc - radius of circular shadow of RL
    fn circfit(&self, q: f64, xl1: f64, phi: f64, incl: f64) -> (usize, f64, f64, f64){
        const ACC: f64 = 5.0e-6;
        
        // some computations for speedup
        let phase = phi * TAU; // orbital phase in radians
        let sphi = phase.sin();
        let cphi = phase.cos();
        let cosi = (incl * PI / 180.0).cos();
        let sini = (incl * PI / 180.0).sin();

        // first point to consider - where donor cuts the line of centres
        // x0 is center of WD. We see if there is no eclipse at all
        // We use something slightly larger than the WD as the donor
        // limb is not tangential at this point
        let earth: Vec3 = set_earth_iangle(incl, phi);
        // TODO: this will fail at phi=0, i=90 - special case
        let mut alpha = 1.2 * self.radius / (1.0 - earth.x*earth.x).sqrt();
        let mut x0 = Vec3::new(-alpha, 0.0, 0.0);

        // use acc of 0.05, this doesn't need to be accurate
        // will panic on fail
        if blink(q, xl1, &x0, &earth, 0.05) {
            // total eclipse
            return (2, 0.0, 0.0, 0.0);
        }
        // check front to see if there is no eclipse at all
        x0.set(alpha, 0.0, 0.0);
        if !blink(q, xl1, &x0, &earth, ACC) {
            // no eclipse
            return (1, 0.0, 0.0, 0.0);
        }
        // from here on in, the code is Tom Marsh magic. 
        // I won't pretend to fully understand it.
        let mut a1 = -alpha;
        let mut a2 = alpha;
        loop {
            let x1 = (a1 + a2)/2.0;
            x0.set(x1, 0.0, 0.0);
            if blink(q, xl1, &x0, &earth, 0.05) {
                a2 = x1;
            } else {
                a1 = x1;
            }
            if (a2-a1).abs() <= ACC {
                break;
            }
        }
        alpha = (a1+a2)/2.0;
        // we have found position of first point to fit circle to
        let x1 = alpha * sphi;
        let y1 = -alpha * cosi * cphi;

        /*
        2nd and third points are on circle of twice radius of white dwarf
        existence of such points apparently guaranteed by existence of 1st point
        */
        let trw = 2.0 * self.radius;
        let ax1 = trw * sphi;
        let ax2 = -trw * cphi * cosi;
        let ay1 = trw * cphi;
        let ay2 = trw * sphi * cosi;
        let az = trw * sini;
        /*
        theta 1 is angle on circle where line of centres towards donor
        crosses and is thus in eclipse
        */
        let theta1 = (-cosi*cphi).atan2(sphi);
        
        // now we perform a binary chop to find the next two points
        let mut x2 = 0.0;
        let mut y2 = 0.0;
        let mut x3 = 0.0;
        let mut y3 = 0.0;
        let mut t1: f64;
        let mut t2: f64;
        for i in 1..=2 {
            if i == 1 {
                t1 = theta1;
                t2 = theta1 + PI;
            } else {
                t1 = theta1;
                t2 = theta1 - PI;
            }

            let mut theta: f64;
            let mut cthet: f64;
            let mut sthet: f64;

            loop {
                theta = (t1+t2)/2.0;
                cthet = theta.cos();
                sthet = theta.sin();
                x0.set(
                    ax1*cthet+ax2*sthet,
                    ay1*cthet+ay2*sthet,
                    az*sthet
                );
                if blink(q, xl1, &x0, &earth, 0.05) {
                    t1 = theta;
                } else {
                    t2 = theta;
                }
                if trw*(t2-t1).abs() < ACC {
                    break;
                }
            }
            theta = (t1 + t2)/2.0;
            cthet = theta.cos();
            sthet = theta.sin();
            if i==1 {
                x2 = trw * cthet;
                y2 = trw * sthet;
            } else {
                x3 = trw * cthet;
                y3 = trw * sthet;
            }
        }
        // now we have our three points. Fit circle to them and return
        // parameters of circle (xc,yc,rc)
        let c11 = 2.*(y2-y3);
        let c12 = 2.*(y3-y1);
        let c13 = 2.*(y1-y2);
        let c21 = 2.*(x3-x2);
        let c22 = 2.*(x1-x3);
        let c23 = 2.*(x2-x1);
        let c31 = 4.*(x2*y3-x3*y2);
        let c32 = 4.*(x3*y1-x1*y3);
        let c33 = 4.*(x1*y2-x2*y1);
        let b1 = x1*x1 + y1*y1;
        let b2 = x2*x2 + y2*y2;
        let b3 = x3*x3 + y3*y3;
        let delta = c31+c32+c33;
        let d1 = (c11*b1+c12*b2+c13*b3)/delta;
        let d2 = (c21*b1+c22*b2+c23*b3)/delta;
        let d3 = (c31*b1+c32*b2+c33*b3)/delta;
        let xc = d1;
        let yc = d2;
        let rc = (d3 + d1*d1 + d2*d2).sqrt();
        (0, xc, yc, rc)
    }

    /// Commputes flux of white dwarf relative to out of eclipse flux
    fn calcflux_at_point(&self, q: f64, phi: f64, incl: f64) -> f64 {
        const MAXRAD: usize = 100;
        let n_ecl: usize;
        let mut rc: f64;
        let mut xc: f64;
        let mut yc: f64;

        (n_ecl, xc, yc, rc) = self.circfit(q, self._xl1, phi, incl);
        // convert from units of separation to xl1
        rc /= self._xl1;
        yc /= self._xl1;
        xc /= self._xl1;

        if n_ecl == 1 {
            // no eclipse
            return 1.0;
        } else if n_ecl == 2 {
            // total eclipse
            return 0.0;
        } 

        // harder job. integrate over annuli of the white dwarf
        // compute projected distance between centres of annuli
        let pd = (xc*xc + yc*yc).sqrt(); // distance of circle from origin
        if rc >= pd + self.radius {
            // shadow covers entirely
            return 0.0;
        } else if rc <= pd - self.radius {
            // shadow doesn't quite cover WD at all
            return 1.0;
        }

        let mut wflux: f64;
        if rc > pd {
            let rlo = rc - pd;
            let mut nrad = ((MAXRAD as f64) * (1.0 - rlo / self.radius)) as u32;
            nrad = nrad.max(10);
            let fac = rlo * (rc + pd);
            let xlo = rlo * rlo;
            let rw2 = self.radius * self.radius;
            let range = rw2 - xlo;
            let dx = range / nrad as f64 / rw2;
            // sum over radial annuli
            // serial faster than parallel as we parellelise over phases instead
            wflux = 0.0;
            for i in 1..=nrad {
                let x = xlo + range * (i as f64 - 0.5) / nrad as f64;
                let theta = ((fac-x)/2.0/pd/x.sqrt()).acos();
                let flux: f64 = theta * (1.0 - self.ulimb * (1.0 - (1.0-x/rw2).sqrt()));
                wflux += flux;
            }
            
            wflux = wflux * dx / PI / (1.0 - self.ulimb/3.0);
        } else {
            let rlo = pd - rc;
            let mut nrad = ((MAXRAD as f64) * (1.0 - rlo / self.radius)) as u32;
            nrad = nrad.max(10);
            let fac = -rlo * (rc + pd);
            let xlo = rlo * rlo;
            let rw2 = self.radius * self.radius;
            let range = rw2 - xlo;
            let dx = range / nrad as f64 / rw2;
            // sum over radial annuli
            // serial faster than parallel as we parellelise over phases instead
            wflux = 0.0;
            for i in 1..=nrad {
                let x = xlo + range * (i as f64 - 0.5) / nrad as f64;
                let theta = PI - ((fac-x)/2.0/pd/x.sqrt()).acos();
                let flux: f64 = theta * (1.0 - self.ulimb * (1.0 - (1.0-x/rw2).sqrt()));
                wflux += flux;
            }
            wflux = 1.0 - dx * wflux/PI/(1.0-self.ulimb/3.0);
        }       
        wflux
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

    #[pyo3(signature=(q, incl, phases, widths=None))]
    pub fn calcflux(&mut self, q: f64, incl: f64, phases: Vec<f64>, widths: Option<Vec<f64>>) -> PyResult<Vec<f64>> {

        let n: usize = phases.len();
        // has q changed since last time? If so, recalculate xl1
        if (q-self._qcalc).abs() > 1e-6 {
            self._qcalc = q;
            self._xl1 = x_l1(q)?;
        }

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