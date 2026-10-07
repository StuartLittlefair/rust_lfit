"""
A Wrapper object that holds a disc, donor, bright spot and white dwarf
and provides convenient routines for calculating the total flux.

Access is provided to the underlying components for advanced use, but
most users will only ever need to use the calcFlux method, and access
the ywd, yd, ys and yrs properties which, when calculated provide arrays
of the white dwarf, disc, bright spot, and donor star fluxes respectively"""

import matplotlib.collections as mcoll
import matplotlib.tri as mtri
import numpy as np
import roche
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

from .rust import Brightspot, Disc, Donor, Whitedwarf


def colorline(
    x,
    y,
    z=None,
    cmap=plt.get_cmap("seismic"),
    norm=plt.Normalize(0.0, 1.0),
    linewidth=3,
    alpha=1.0,
):
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
    lc = mcoll.LineCollection(
        segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha
    )

    ax = plt.gca()
    ax.add_collection(lc)

    return lc


def _surface_colors(triangles, flux, visible, cmap_name):
    """
    Colour triangle faces by the mean flux of their vertices, using the given
    colormap. Faces whose vertices are all eclipsed are coloured dark grey.
    """
    cmap = plt.get_cmap(cmap_name)
    face_flux = flux[triangles].mean(axis=1)
    fmin, fmax = face_flux.min(), face_flux.max()
    norm = plt.Normalize(fmin, fmax if fmax > fmin else fmin + 1.0)
    colors = cmap(norm(face_flux))
    colors[~visible[triangles].all(axis=1)] = (0.1, 0.1, 0.1, 1.0)
    return colors


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
        """Initialiser for CV object.

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
        """
        if len(pars) != 14 and len(pars) != 18:
            raise ValueError("pars must be a list, array or tuple of length 14 or 18")

        (
            wdFlux,
            dFlux,
            sFlux,
            rsFlux,
            q,
            dphi,
            rdisc,
            ulimb,
            rwd,
            scale,
            az,
            fis,
            dexp,
            phi0,
        ) = pars[0:14]
        self.complex = False if len(pars) == 14 else True
        if len(pars) == 18:
            exp1, exp2, tilt, yaw = pars[14:]

        xl1 = roche.xl1(q)
        incl = roche.findi(q, dphi)

        self.donor = Donor(q, 0.5, nlat_donor)
        self.wd = Whitedwarf(rwd, ulimb)
        self.disc = Disc(q, rwd, rdisc, dexp)
        if self.complex:
            self.brightspot = Brightspot(
                q, rdisc, az, fis, scale, exp1, exp2, tilt, yaw
            )
        else:
            self.brightspot = Brightspot(q, rdisc, az, fis, scale)

        # store system parameters (used by plot3d)
        self.q = q
        self.dphi = dphi
        self.rwd = rwd
        self.rdisc = rdisc

        # no fluxes
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
        (
            wdFlux,
            dFlux,
            sFlux,
            rsFlux,
            q,
            dphi,
            rdisc,
            ulimb,
            rwd,
            scale,
            az,
            fis,
            dexp,
            phi0,
        ) = pars[0:14]
        self.complex = False if len(pars) == 14 else True
        if len(pars) == 18:
            exp1, exp2, tilt, yaw = pars[14:]

        xl1 = roche.xl1(q)
        incl = roche.findi(q, dphi)
        if incl < 0:
            raise ValueError(f"Invalid combination of q and dphi: {q}, {dphi}")

        self.wd.tweak(rwd, ulimb)
        self.disc.tweak(q, rwd, rdisc, dexp)
        if self.complex:
            self.brightspot.tweak(q, rdisc, az, fis, scale, exp1, exp2, tilt, yaw)
        else:
            self.brightspot.tweak(q, rdisc, az, fis, scale)

        # store system parameters (used by plot3d)
        self.q = q
        self.dphi = dphi
        self.rwd = rwd
        self.rdisc = rdisc

        # calculate fluxes
        self.ywd = wdFlux * np.array(self.wd.calcflux(q, incl, phi - phi0, width))
        self.yd = dFlux * np.array(self.disc.calcflux(q, incl, phi - phi0, width))
        self.ys = sFlux * np.array(self.brightspot.calcflux(q, incl, phi - phi0, width))
        self.yrs = rsFlux * np.array(self.donor.calcflux(q, incl, phi - phi0, width))

        return self.ywd + self.yd + self.ys + self.yrs

    def plot(self, pars, phi=None):
        assert (len(pars) == 18) or (len(pars) == 14)
        (
            wdFlux,
            dFlux,
            sFlux,
            rsFlux,
            q,
            dphi,
            rdisc,
            ulimb,
            rwd,
            scale,
            az,
            fis,
            dexp,
            phi0,
        ) = pars[0:14]
        if len(pars) > 14:
            exp1, exp2, tilt, yaw = pars[14:]
        else:
            exp1 = 2.0
            exp2 = 1.0

        xl1 = roche.xl1(q)
        incl = roche.findi(q, dphi)
        if incl < 0:
            raise Exception("invalid combination of q and dphi: %f %f" % (q, dphi))

        xl1_a = roche.xl1(q)
        x, y = roche.stream(q, 0.01)
        plt.plot(x, y, ":")
        x2, y2 = roche.streamr(q, rdisc * xl1_a, n_points=400)
        plt.plot(x2, y2)
        xd, yd = roche.lobe2(q)
        plt.plot(xd, yd)
        disc = plt.Circle((0, 0), xl1_a * rdisc, color="r", alpha=0.5)
        plt.gca().add_patch(disc)
        wd = plt.Circle((0, 0), xl1_a * rwd, color="b", alpha=0.5)
        plt.gca().add_patch(wd)

        r, _ = roche.bspot(q, rdisc * xl1_a, smax=1.0e-4)
        spotx = r.x
        spoty = r.y
        BMAX = pow(exp1 / exp2, 1 / exp2)
        spot_max = pow(BMAX, exp1) * np.exp(-pow(BMAX, exp2))
        curr_flux = spot_max
        ppos = BMAX
        while curr_flux > spot_max / 1000:
            ppos += BMAX / 10
            curr_flux = pow(ppos, exp1) * np.exp(-pow(ppos, exp2))

        SFAC = min(20 + BMAX, BMAX + ppos)

        nspot = max(200, int(50 * SFAC / BMAX))
        nspot = min(nspot, 1000)

        tangent = np.arctan2(spoty, spotx) + np.pi / 2.0
        theta = tangent + az * 2 * np.pi / 360.0
        steps = scale * np.linspace(0, SFAC, nspot)
        u = steps / scale
        spotx, spoty = spotx + steps * np.cos(theta), spoty + steps * np.sin(theta)
        spot_flux = u**exp1 * np.exp(-(u**exp2)) / spot_max
        colorline(spotx, spoty, spot_flux, cmap=plt.get_cmap("copper"), linewidth=2)

        if phi is not None:
            xs, ys, mask = roche.shadow(q, incl, phi)
            plt.fill(xs[mask], ys[mask], color="k", alpha=0.2)

        plt.gca().set_aspect("equal")

    def _plot3d_context(self):
        incl = roche.findi(self.q, self.dphi)
        if incl < 0:
            raise ValueError(
                f"Invalid combination of q and dphi: {self.q}, {self.dphi}"
            )
        xl1 = roche.xl1(self.q)

        # make sure the surface grids exist
        if len(self.disc.grid) == 0:
            self.disc.update_grid(incl)
        if len(self.donor.grid) == 0:
            self.donor.update_grid()

        donor_pts = np.array(
            [[p.position.x, p.position.y, p.position.z] for p in self.donor.grid]
        )
        donor_flux = np.array([p.flux for p in self.donor.grid])
        try:
            from scipy.spatial import ConvexHull

            donor_triangles = ConvexHull(donor_pts).simplices if len(donor_pts) > 0 else None
        except ImportError:
            donor_triangles = None

        disc_pts = np.array(
            [[p.position.x, p.position.y, p.position.z] for p in self.disc.grid]
        )
        disc_flux = np.array([p.flux for p in self.disc.grid])
        disc_tri = None
        disc_mask = None
        if len(disc_pts) > 0:
            tri = mtri.Triangulation(disc_pts[:, 0], disc_pts[:, 1])
            # Delaunay fills the central hole; mask triangles inside the inner edge
            rcent = np.hypot(
                disc_pts[:, 0][tri.triangles].mean(axis=1),
                disc_pts[:, 1][tri.triangles].mean(axis=1),
            )
            tri.set_mask(rcent < self.rwd * xl1)
            disc_tri = tri
            disc_mask = tri.mask

        u = np.linspace(0, 2 * np.pi, 40)
        v = np.linspace(0, np.pi, 20)
        rwd_abs = self.rwd * xl1
        xw = rwd_abs * np.outer(np.cos(u), np.sin(v))
        yw = rwd_abs * np.outer(np.sin(u), np.sin(v))
        zw = rwd_abs * np.outer(np.ones_like(u), np.cos(v))

        xs, ys = roche.stream(self.q, 0.01)
        xt, yt = roche.streamr(self.q, self.rdisc * xl1, n_points=400)

        rspot, _ = roche.bspot(self.q, self.rdisc * xl1, smax=1.0e-4)
        rs = 0.02
        xb = rspot.x + rs * np.outer(np.cos(u), np.sin(v))
        yb = rspot.y + rs * np.outer(np.sin(u), np.sin(v))
        zb = rs * np.outer(np.ones_like(u), np.cos(v))

        # equal aspect ratio, with limits covering everything plotted
        bounds = [p for p in (donor_pts, disc_pts) if len(p) > 0]
        bounds.append(np.column_stack([xs, ys, np.zeros_like(xs)]))
        bounds = np.vstack(bounds)
        mid = (bounds.max(axis=0) + bounds.min(axis=0)) / 2
        max_range = (bounds.max(axis=0) - bounds.min(axis=0)).max() / 2
        return {
            "incl": incl,
            "donor_pts": donor_pts,
            "donor_flux": donor_flux,
            "donor_triangles": donor_triangles,
            "disc_pts": disc_pts,
            "disc_flux": disc_flux,
            "disc_tri": disc_tri,
            "disc_mask": disc_mask,
            "wd_surface": (xw, yw, zw),
            "stream_all": (xs, ys),
            "stream_disc": (xt, yt),
            "impact_surface": (xb, yb, zb),
            "mid": mid,
            "max_range": max_range,
        }

    def _render_plot3d(self, ax, phi, ctx, preserve_limits=False):
        donor_vis = np.array([p.is_visible(phi) for p in self.donor.grid])
        disc_vis = np.array([p.is_visible(phi) for p in self.disc.grid])
        xlim = ax.get_xlim3d() if preserve_limits else None
        ylim = ax.get_ylim3d() if preserve_limits else None
        zlim = ax.get_zlim3d() if preserve_limits else None
        ax.clear()

        donor_pts = ctx["donor_pts"]
        donor_triangles = ctx["donor_triangles"]
        if len(donor_pts) > 0:
            if donor_triangles is not None:
                # plot_trisurf has no per-face colour support, so set the
                # colours on the collection it returns
                polyc = ax.plot_trisurf(
                    *donor_pts.T,
                    triangles=donor_triangles,
                    edgecolor="none",
                    shade=False,
                )
                polyc.set_facecolors(
                    _surface_colors(
                        donor_triangles, ctx["donor_flux"], donor_vis, "autumn"
                    )
                )
            else:
                ax.scatter(*donor_pts.T, c=ctx["donor_flux"], cmap="autumn", s=5)

        disc_pts = ctx["disc_pts"]
        disc_tri = ctx["disc_tri"]
        if len(disc_pts) > 0 and disc_tri is not None:
            polyc = ax.plot_trisurf(
                disc_tri, disc_pts[:, 2], edgecolor="none", shade=False
            )
            colors = _surface_colors(
                disc_tri.triangles, ctx["disc_flux"], disc_vis, "hot"
            )
            if ctx["disc_mask"] is not None:
                colors = colors[~ctx["disc_mask"]]
            polyc.set_facecolors(colors)

        xw, yw, zw = ctx["wd_surface"]
        ax.plot_surface(xw, yw, zw, color="lightblue", alpha=0.9)

        xs, ys = ctx["stream_all"]
        ax.plot(xs, ys, np.zeros_like(xs), ":", color="grey")
        xt, yt = ctx["stream_disc"]
        ax.plot(xt, yt, np.zeros_like(xt), color="k")

        xb, yb, zb = ctx["impact_surface"]
        ax.plot_surface(xb, yb, zb, color="gold")

        mid = ctx["mid"]
        max_range = ctx["max_range"]
        if preserve_limits:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)
        else:
            ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
            ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
            ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
        ax.set_box_aspect((1, 1, 1))

        # look at the system from the direction of the observer at this phase
        phi_rad = 2 * np.pi * phi
        azim = np.degrees(np.arctan2(-np.sin(phi_rad), np.cos(phi_rad)))
        ax.view_init(elev=90.0 - ctx["incl"], azim=azim)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(f"phase = {phi:.3f}")

    def plot3d(self, phi):
        """
        Produce a 3D visualisation of the CV at orbital phase ``phi``.

        The donor star and accretion disc are rendered as surfaces built from
        their ``grid`` attributes, coloured by the local surface brightness.
        Regions eclipsed by the donor at phase ``phi`` are shown dark grey.
        The white dwarf is rendered as a sphere of radius ``rwd``. The gas
        stream is shown as a line (solid up to the disc edge, dotted for the
        ballistic continuation), with a small sphere marking the point where
        the stream hits the disc.

        The view is oriented along the line of sight to the observer at phase
        ``phi``. If the surface grids have not been calculated yet (e.g.
        ``calcFlux`` has not been called) they are calculated here. For a
        smooth donor surface, create the CV with ``nlat_donor`` around 20 or
        larger; the default analytical donor model only produces a very
        coarse grid.

        Parameters
        ----------
        phi : float
            Orbital phase at which to view the system.

        Returns
        -------
        fig, ax
            The matplotlib figure and 3D axes.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        self._render_plot3d(ax, phi, self._plot3d_context())
        return fig, ax

    def plot3d_interactive(self, phi=0.0):
        """
        Produce an interactive 3D visualisation of the CV with a phase slider.

        Parameters
        ----------
        phi : float
            Initial orbital phase.

        Returns
        -------
        fig, ax, slider
            The matplotlib figure, 3D axes and phase slider.
        """
        ctx = self._plot3d_context()
        phi = phi % 1.0
        fig = plt.figure()
        fig.subplots_adjust(bottom=0.18)
        ax = fig.add_subplot(111, projection="3d")
        self._render_plot3d(ax, phi, ctx)

        slider_ax = fig.add_axes((0.15, 0.06, 0.7, 0.03))
        slider = Slider(slider_ax, "phase", 0.0, 1.0, valinit=phi)

        def _update(val):
            self._render_plot3d(ax, val, ctx, preserve_limits=True)
            fig.canvas.draw_idle()
        slider.on_changed(_update)
        return fig, ax, slider
