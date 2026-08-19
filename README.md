# LFIT_RS

LFIT is a python library for fitting the lightcurves of eclipsing Dwarf
Nova. It leans *very* heavily on Tom Marsh's library for the Roche
geometry and it's dependencies.

This is a Rust translation of the original LFIT code, which is substantially
easier to install than the original C++ version, and no slower.

It is designed to be simpler to use than Tom's lightcurve fitting code
[lcurve](https://github.com/trm-astro/lcurve) and also faster, since 
it uses a much simpler model for the white dwarf component (uniform disk, linear
limb-darkening) and implements fewer features (no gravitational lensing, no
starspots etc.)

However, it shares a lot of DNA with [lcurve](https://github.com/trm-astro/lcurve),
using the same underlying libraries and the same model for the bright spot.

# INSTALLATION 

Installation should be as simple as 

```
pip install lfit-rs
```

# Usage

The most common usage is pattern is to create a `CV` object:

```python
import lfit_rs

cv = lfit_rs.CV(pars, nel_disc=1000, nlat_donor=20)
```

The parameters are a list of 14 or 18 parameters - depending on whether the simple or complex bright spot model are used. For an explanation of the parameters, see below. 

Once a `CV` object is created, we can calculate the flux at a list of orbital phases using:

```python
phi = np.linspace(-0.2, 0.2, 500)
flux = cv.calcFlux(pars, phi)
```

If you wish to change the parameters, you should not create a new `CV` object, but instead pass the new parameters to `calcFlux` and the CV parameters will be updated automatically:

```python
new_pars = [...]
flux = cv.calcFlux(new_pars, phi)
```

`lfit_rs.CV.calcFlux` also takes an optional `width` argument. If the optional
width argument is provided, the flux is calculated at five points over a bin 
of this width centered on the given phase and the average flux in this bin is 
returned. This is important to use when you have low time resolution, and the 
flux may change rapidly over the course of a time bin.

## Model parameters
For a fuller description of the model parameters, see [Savoury (2011)](https://ui.adsabs.harvard.edu/abs/2011MNRAS.415.2025S/abstract). The bright spot is modelled as two linear strips passing through the intersection of the gas stream and disc. One strip is isotropic, while the other beams in a given direction. Both strips occupy the same physical space.

 In brief, the CV parameters are (in order):


1.  wdFlux -  white dwarf flux at maximum light
2.  dFlux  -  disc flux at maximum light
3.  sFlux  -  bright spot flux at maximum light
4.  rsFlux -  donor flux at maximum light
5.  q      -  mass ratio
6.  dphi   -  full width of white dwarf at mid ingress/egress
7.  rdisc  -  radius of accretion disc (scaled by distance to inner lagrangian point XL1)
8.  ulimb  -  linear limb darkening parameter for white dwarf
9.  rwd    -  white dwarf radius (scaled to XL1)
10.  scale  -  bright spot scale (scaled to XL1)
11.  az     -  the azimuth of the bright spot strip (w.r.t to line of centres between stars)
12.  fis    -  the fraction of the bright spot's flux which radiates isotropically
13.  dexp   -  the exponent which governs how the brightness of the disc falls off with radius
14.  phi0   -  a phase offset

the next four parameters are only used for complex bright spot models

15.  exp1 - the `Y` exponent in [Savoury (2011)](https://ui.adsabs.harvard.edu/abs/2011MNRAS.415.2025S/abstract), which governs how rapidly the bright spot flux increases with distance along the strip
16.  exp2 - the `Z` exponent in [Savoury (2011)](https://ui.adsabs.harvard.edu/abs/2011MNRAS.415.2025S/abstract), which governs how rapidly the bright spot flux falls with distance along the strip.
17.  tilt - a parameter that allows the bright spot strip to beam in a different direction than the perpendicular to the strip itself. The tilt sets the beaming angle w.r.t to the disc plane.`tilt=90` beams light in the plane of the disc.  
18.  yaw - the bright spot yaw also affects the bright spot strip beaming angle. The yaw is added to the bright spot azimuth to set the beaming angle in the plane of the disc.

## Differences from the C++ LFIT version

This translation yields identical results to the C++ version within machine precision with **one** exception. The CVs normally modelled with LFIT usually have minimal contribution from the donor star, and so the grid-based donor flux calculation slowed things down for no advantage.

By default, `lfit_rs` uses the analytical formula of Kopal 1959 for ellipsoidal variability. If the donor does contribute to your lightcurve, and you are interested in the donor flux itself, this formula is not sufficiently accurate and you should set `nlat_donor` to a value ~ 20 to use the tiled donor model.
