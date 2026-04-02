
'''
A Wrapper object that holds a disc, donor, bright spot and white dwarf
and provides convenient routines for calculating the total flux.

Access is provided to the underlying components for advanced use, but
most users will only ever need to use the calcFlux method, and access
the ywd, yd, ys and yrs properties which, when calculated provide arrays
of the white dwarf, disc, bright spot, and donor star fluxes respectively'''
import roche
from .rust import Whitedwarf, Donor, Brightspot, Disc

import numpy as np

from matplotlib import pyplot as plt
import matplotlib.collections as mcoll


def colorline(
    x, y, z=None, cmap=plt.get_cmap('seismic'), norm=plt.Normalize(0.0, 1.0),
        linewidth=3, alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)

    ax = plt.gca()
    ax.add_collection(lc)

    return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments



class CV:
    def __init__(self, pars, nel_disc=1000, nlat_donor=0):
        '''Initialiser for CV object. 
        
        The parameters argument is a tuple, array or list which contains either 
        14 parameters, or 18 parameters for more complicated bright spot models.

        The bright spot is modelled as a linear strip at an angle to the line of centres.
        A fraction of the bright spot strip radiates isotropically, whilst the remainder
        is beamed normal to the surface (in the simple model). In the more complex model
        the bright spot can be made to decay in brightness along it's length at different
        rates (by the two exponent parameters), and beam in a direction other than the normal
        to the direction (using the tilt and yaw parameters

        The CV parameters are (in order):

        wdFlux -  white dwarf flux at maximum light
        dFlux  -  disc flux at maximum light
        sFlux  -  bright spot flux at maximum light
        rsFlux -  donor flux at maximum light
        q      -  mass ratio
        dphi   -  full width of white dwarf at mid ingress/egress
        rdisc  -  radius of accretion disc (scaled by distance to inner lagrangian point XL1)
        ulimb  -  linear limb darkening parameter for white dwarf
        rwd    -  white dwarf radius (scaled to XL1)
        scale  -  bright spot scale (scaled to XL1)
        az     -  the azimuth of the bright spot strip (w.r.t to line of centres between stars)
        fis    -  the fraction of the bright spot's flux which radiates isotropically
        dexp   -  the exponent which governs how the brightness of the disc falls off with radius
        phi0   -  a phase offset

        the next four parameters are only used for complex bright spot models
        exp1, exp2, tilt, yaw. Their use is described above.

        The accretion disc and donor are broken into tiles covering their surface. You can
        override the defaults for these tiles by setting the nel_disc or nel_donor arguments.
        This can increase numerical accuracy at the expense of computing time.
        
        The default value of `nlat_donor` uses the analytical formula of Kopal 1959
        for ellipsoidal variability. This is because the donor rarely contributes
        much flux. If the donor does contribute to your lightcurve, and you are 
        interested in the donor flux itself, this formula is not sufficient and you
        should set `nlat_donor` to a value ~ 20 to use the tiled donor model.
        '''
        if len(pars) != 14 and len(pars) != 18:
            raise ValueError("pars must be a list, array or tuple of length 14 or 18")
        
        wdFlux,dFlux,sFlux,rsFlux,q,dphi,rdisc,ulimb,rwd,scale,az,fis,dexp,phi0 = pars[0:14]
        self.complex = False if len(pars) == 14 else True
        if len(pars) == 18:
            exp1, exp2, tilt, yaw = pars[14:]

        xl1 = roche.xl1(q)
        incl = roche.findi(q, dphi)

        self.donor = Donor(q, 0.5, nlat_donor)
        self.wd = Whitedwarf(rwd, ulimb)
        self.disc = Disc(q, rwd, rdisc, dexp)
        if self.complex:
            self.brightspot = Brightspot(q, rdisc, az, fis, scale, exp1, exp2, tilt, yaw)
        else:
            self.brightspot = Brightspot(q, rdisc, az, fis, scale)

        # no fluxes
        self.ywd = None
        self.yd = None
        self.ys = None
        self.yrs = None

        # flag to see if grids need initialising
        self.computed = False

    def calcFlux(self, pars, phi, width=None):
        """
        Calculate the flux from the CV for given parameters and phases.

        Tweaks the parameters of the CVand calculates the flux from the CV 
        as a whole, and from the components of the CV.

        Parameters
        ----------
        pars : list, array or tuple
            A list, array or tuple of parameters to update the CV with. 

            The pars list is as described for creation of a CV, and you can switch between
            simple and complex bright spots on the fly just by providing different numbers
            of parameters.

            the flux at the phases given in phi is calculated and returned. If the optional
            width argument is provided, the flux is calculated in a bin of this width and the
            average flux in this bin is returned
        
        phi : list, array or tuple
            A list, array or tuple of phases at which to calculate the flux.
                
        width : list, array or tuple
            Phase widths for bins.

        Returns
        -------
        flux : array
            The total flux from the CV at the given phases.
        """
        wdFlux,dFlux,sFlux,rsFlux,q,dphi,rdisc,ulimb,rwd,scale,az,fis,dexp,phi0 = pars[0:14]
        self.complex = False if len(pars) == 14 else True
        if len(pars) == 18:
            exp1, exp2, tilt, yaw = pars[14:]

        xl1 = roche.xl1(q)
        incl = roche.findi(q, dphi)
        if incl < 0:
            raise ValueError(f"Invalid combination of q and dphi: {q}, {dphi}")
        
        # check that brightspot parameters are valid
        # angle can be this far from disc tangent, but no further
        slop = 80.0
        try:
            # position of proposed spot
            x, y = self.brightspot.spot_position(q, rdisc)
        except ValueError:
            raise ValueError(f"Gas stream trajectory does not intersect disc for q={q} and rdisc={rdisc}")
        
        # tangent to disc at this position
        alpha = np.degrees(np.arctan2(y, x))

        # alpha is between -90 and 90.
        # if negative spot lags disc ie alpha > 90
        alpha = 90 - alpha if alpha < 0 else alpha;
        tangent = alpha + 90

        # BS azimuth should be between 0 and 178, and less than slop degrees from
        # tangent to disc at this position
        if az < 0 or az > 178 or np.fabs(tangent - az) > slop:
            raise ValueError(f"Invalid bright spot azimuth: {az}.\n Must be between 0 and 178, and less than {slop} degrees from tangent to disc at this position ({tangent} degrees)")
        
        self.wd.tweak(rwd, ulimb)
        self.disc.tweak(q, rwd, rdisc, dexp)
        if self.complex:
            self.brightspot.tweak(q, rdisc, az, fis, scale, exp1, exp2, tilt, yaw)
        else:
            self.brightspot.tweak(q, rdisc, az, fis, scale)

        # calculate fluxes
        self.ywd = wdFlux * np.array(self.wd.calcflux(q, incl, phi-phi0, width))
        self.yd = dFlux * np.array(self.disc.calcflux(q, incl, phi-phi0, width))
        self.ys = sFlux * np.array(self.brightspot.calcflux(q, incl, phi-phi0, width))
        self.yrs = rsFlux * np.array(self.donor.calcflux(q, incl, phi-phi0, width))

        return self.ywd + self.yd + self.ys + self.yrs

    def plot(self, pars, phi=None):
        assert (len(pars) == 18) or (len(pars) == 14)
        wdFlux,dFlux,sFlux,rsFlux,q,dphi,rdisc,ulimb,rwd,scale,az,fis,dexp,phi0 = pars[0:14]
        if len(pars) > 14:
            exp1, exp2, tilt, yaw = pars[14:]

        xl1 = roche.xl1(q)
        incl = roche.findi(q,dphi)
        if incl < 0:
            raise Exception('invalid combination of q and dphi: %f %f' % (q, dphi))

        xl1_a = roche.xl1(q)
        x, y = roche.stream(q, 0.01)
        x, y = roche.stream(q, 0.01)
        plt.plot(x, y, ":")
        x2, y2 = roche.streamr(q, rdisc * xl1_a, n=400)
        plt.plot(x2, y2)
        xd, yd = roche.lobe2(q)
        plt.plot(xd, yd)
        disc = plt.Circle((0, 0), xl1_a * rdisc, color="r", alpha=0.5)
        plt.gca().add_patch(disc)
        wd = plt.Circle((0, 0), xl1_a * rwd, color="b", alpha=0.5)
        plt.gca().add_patch(wd)

        spotx, spoty, _, _ = roche.bspot(q, rdisc * xl1_a)
        BMAX = pow(exp1 / exp2, 1 / exp2)
        spot_max = pow(BMAX, exp1) * np.exp(-pow(BMAX, exp2))
        curr_flux = spot_max
        ppos = BMAX
        while curr_flux > spot_max / 1000:
            ppos += BMAX / 10
            curr_flux = pow(ppos, exp1) * np.exp(-pow(ppos, exp2))
        
        SFAC = min(20+BMAX, BMAX+ppos)

        nspot = max(200, int(50 * SFAC / BMAX))
        nspot = min(nspot, 1000)

        theta = az * 2 * np.pi / 360.0
        steps = scale * np.linspace(0, SFAC, nspot)
        u = steps/scale
        spotx, spoty = spotx + steps * np.cos(theta), spoty + steps * np.sin(theta)
        spot_flux =  u**exp1 * np.exp(-u**exp2) / spot_max
        colorline(spotx, spoty, spot_flux, cmap=plt.get_cmap('copper'), linewidth=2)

        if phi is not None:
            xs, ys, mask = roche.shadow(q, incl, phi)
            plt.fill(xs[mask], ys[mask], color="k", alpha=0.2)

        plt.gca().set_aspect("equal")
