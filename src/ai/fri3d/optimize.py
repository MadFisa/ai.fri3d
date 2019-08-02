"""The module defines the fitting functions used to fit the model to
white-light and in-situ data.
"""
# pylint: disable=E1101
# pylint: disable=E1102
# pylint: disable=E0401
# pylint: disable=C0103
# pylint: disable=W0102
# pylint: disable=W0212
from datetime import datetime
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.distance import euclidean
from matplotlib import pyplot as plt
from matplotlib import gridspec
from astropy import units as u
from fastdtw import fastdtw
from ai.fri3d.model import StaticFRi3D, DynamicFRi3D
from ai import cs

BLIND_PALETTE = {
    "orange": (0.901960784, 0.623529412, 0.0),
    "sky-blue": (0.337254902, 0.705882353, 0.91372549),
    "bluish-green": (0.0, 0.619607843, 0.450980392),
    "yellow": (0.941176471, 0.894117647, 0.258823529),
    "blue": (0.0, 0.447058824, 0.698039216),
    "vermillion": (0.835294118, 0.368627451, 0.0),
    "reddish-purple": (0.8, 0.474509804, 0.654901961),
}


d_prev = np.inf

# TODO
# /users/cpa/alexeyi/.virtualenvs/ai.fri3d.py3/lib/python3.6/site-packages/numpy/core/_methods.py:112: RuntimeWarning: invalid value encountered in subtract
#   x = asanyarray(arr - arrmean)
# /users/cpa/alexeyi/Development/ai.fri3d/ai/fri3d/model.py:974: RuntimeWarning: invalid value encountered in less
#   if np.any(s < 0) or np.any(s > 1):
# /users/cpa/alexeyi/Development/ai.fri3d/ai/fri3d/model.py:974: RuntimeWarning: invalid value encountered in greater
#   if np.any(s < 0) or np.any(s > 1):
# /users/cpa/alexeyi/Development/ai.fri3d/ai/fri3d/model.py:999: RuntimeWarning: invalid value encountered in less
#   x[x < 0] = 0


def fit2insitu(
    x, y, z, t, b, vt=None, sampling=50, relative=True, weights={"t": 1, "b": 1, "vt": 1}, verbose=False, **kwargs
):
    """Fits FRi3D model to in-situ data (magnetic field and speed).

    Args:
        t (array_like): Vector of timestamps of size (3).
        x (scalar or callable): X-coordinate of the spacecraft.
        y (scalar or callable): Y-coordinate of the spacecraft.
        z (scalar or callable): Z-coordinate of the spacecraft.
        b (array_like): Array of magnetic field vectors of shape (n, 3).
        vt (array_like): Vector of absolute speed values of size (n).
        sampling (int): Number of sampling points used for the fitting.
        relative (bool): Flag for using relative time-profiles (relative
            to the starting time of the fitted data t[0]).
        weights (dict): Importance weigths for the fitted parameters.
        verbose (bool): Verbose mode - print parameters and draw plots
            during fitting.
        **kwargs: Keyworded profiles for all model parameters.

    Returns:
        tuple: (DynamicFRi3D, dict), fitted dynamic FRi3D model
        and dictionary of fitted profiles for all model parameters.
    """
    t_real = np.linspace(t[0], t[-1], sampling)
    dt_real = t_real[1] - t_real[0]
    m = np.logical_not(np.any(np.isnan(b), axis=1))
    tb_real = t[m]
    b_real = b[m, :]
    bt_real = np.sqrt(b_real[:, 0] ** 2 + b_real[:, 1] ** 2 + b_real[:, 2] ** 2)
    if vt is not None:
        m = np.logical_not(np.isnan(vt))
        tvt_real = t[m]
        vt_real = vt[m]
    dfr = DynamicFRi3D()
    profiles = {}
    for prop in dfr._props:
        if prop not in kwargs or not isinstance(kwargs[prop], BaseProfile):
            raise TypeError(prop + " profile object of BaseProfile class is requried.")
        else:
            profiles[prop] = kwargs[prop]
            if profiles[prop].bounds is None:
                if relative:
                    setattr(dfr, prop, lambda t, profile=profiles[prop]: profile.eval(t - t_real[0]))
                else:
                    setattr(dfr, prop, profiles[prop].eval)

    def residual(params):
        global d_prev
        i = 0
        for prop, profile in profiles.items():
            if profile.bounds is not None:
                n = len(profile.bounds)
                profile.params = params[i : i + n]
                i += n
                if relative:
                    setattr(dfr, prop, lambda t, profile=profile: profile.eval(t - t_real[0]))
                else:
                    setattr(dfr, prop, profile.eval)
        t_model = np.arange(
            t_real[0] - dt_real * np.round(sampling / 2), t_real[-1] + dt_real * np.round((sampling + 1) / 2), dt_real
        )
        b_model, vt_model = dfr.insitu(t_model, x, y, z)
        bt_model = np.sqrt(b_model[:, 0] ** 2 + b_model[:, 1] ** 2 + b_model[:, 2] ** 2)
        nonzero_indices = np.where(np.isfinite(bt_model))[0]
        if not (np.isnan(bt_model[0]) and np.isnan(bt_model[-1])):
            return np.inf
        if nonzero_indices.size >= 2:
            t_model = t_model[nonzero_indices[0] : nonzero_indices[-1] + 1]
            if t_model[0] > t_real[-1] or t_model[-1] < t_real[0]:
                return np.inf
            b_model = b_model[nonzero_indices[0] : nonzero_indices[-1] + 1, :]
            bt_model = bt_model[nonzero_indices[0] : nonzero_indices[-1] + 1]
            vt_model = vt_model[nonzero_indices[0] : nonzero_indices[-1] + 1]
            db = fastdtw(
                np.hstack(
                    (
                        (np.array([t_model]).T - t_real[0]) / (t_real[-1] - t_real[0]) * weights["t"],
                        b_model / np.amax(bt_real) * weights["b"],
                    )
                ),
                np.hstack(
                    (
                        (np.array([tb_real]).T - t_real[0]) / (t_real[-1] - t_real[0]) * weights["t"],
                        b_real / np.amax(bt_real) * weights["b"],
                    )
                ),
                dist=euclidean,
            )[0]
            if vt is not None:
                m = np.logical_and(tvt_real >= t_model[0], tvt_real <= t_model[-1])
            dv = (
                fastdtw(
                    np.vstack(
                        (
                            (t_model - t_real[0]) / (t_real[-1] - t_real[0]) * weights["t"],
                            (vt_model - np.amin(vt_real)) / (np.amax(vt_real) - np.amin(vt_real)) * weights["vt"],
                        )
                    ).T,
                    np.vstack(
                        (
                            (tvt_real - t_real[0]) / (t_real[-1] - t_real[0]) * weights["t"],
                            (vt_real - np.amin(vt_real)) / (np.amax(vt_real) - np.amin(vt_real)) * weights["vt"],
                        )
                    ).T,
                    dist=euclidean,
                )[0]
                if vt is not None
                else 0
            )
            if verbose and db + dv < d_prev:
                d_prev = db + dv

                db_real = np.array([datetime.fromtimestamp(t) for t in tb_real])
                if vt is not None:
                    dv_real = np.array([datetime.fromtimestamp(t) for t in tvt_real])
                d_model = np.array([datetime.fromtimestamp(t) for t in t_model])
                plt.close("all")
                fig = plt.figure()
                plt.subplots_adjust(hspace=0.001)
                ax1 = fig.add_subplot(211)
                ax1.plot(db_real, np.sqrt(b_real[:, 0] ** 2 + b_real[:, 1] ** 2 + b_real[:, 2] ** 2), "k")
                ax1.plot(db_real, b_real[:, 0], "r")
                ax1.plot(db_real, b_real[:, 1], "g")
                ax1.plot(db_real, b_real[:, 2], "b")
                ax1.plot(d_model, np.sqrt(b_model[:, 0] ** 2 + b_model[:, 1] ** 2 + b_model[:, 2] ** 2), "--k")
                ax1.plot(d_model, b_model[:, 0], "--r")
                ax1.plot(d_model, b_model[:, 1], "--g")
                ax1.plot(d_model, b_model[:, 2], "--b")
                if vt is not None:
                    ax2 = fig.add_subplot(212, sharex=ax1)
                    ax2.plot(dv_real, vt_real, "k")
                    ax2.plot(d_model, vt_model, "--k")

                    plt.setp(ax1.get_xticklabels(), visible=False)
                plt.ion()
                plt.draw()
                plt.pause(0.1)
                plt.show()

                print("fitness", d_prev)
                for prop, profile in profiles.items():
                    params = profile.params
                    if prop in ("latitude", "longitude", "half_width", "half_height", "tilt", "skew"):
                        params = u.rad.to(u.deg, params)
                    elif prop in ("toroidal_height"):
                        if len(profile.params) == 1:
                            params = u.m.to(u.au, params)
                        else:
                            params = profile.params
                            params[:-1] = u.Unit("m/s").to(u.Unit("km/s"), params[:-1])
                            params[-1] = u.m.to(u.au, params[-1])
                    print(prop, params)
                print("\n")
            return db + dv
        else:
            return np.inf

    bounds = []
    for prop in dfr._props:
        if kwargs[prop].bounds is not None:
            for i in range(len(kwargs[prop].bounds)):
                bounds.append(kwargs[prop].bounds[i])
    res = differential_evolution(
        residual,
        bounds=bounds,
        # maxiter=200,
        # popsize=1,
        # mutation=0.6702,
        # recombination=0.2368
        maxiter=2000,
        popsize=2,
        mutation=0.6714,
        recombination=0.5026,
    )
    i = 0
    for prop, profile in profiles.items():
        if profile.bounds is not None:
            n = len(profile.bounds)
            profile.params = res.x[i : i + n]
            i += n
        setattr(dfr, prop, profile.eval)
    return (dfr, profiles)


# TODO make independent of specific isntruments
def fit2cor(
    cor2a_img_path=None,
    cor2a_aov=u.deg.to(u.rad, 4),
    cor2a_dx=0,
    cor2a_dy=0,
    sta_r=u.au.to(u.m, 1),
    sta_lat=0,
    sta_lon=0,
    cor2b_img_path=None,
    cor2b_aov=u.deg.to(u.rad, 4),
    cor2b_dx=0,
    cor2b_dy=0,
    stb_r=u.au.to(u.m, 1),
    stb_lat=0,
    stb_lon=0,
    c2_img_path=None,
    c2_fov=u.R_sun.to(u.m, 6),
    c2_dx=0,
    c2_dy=0,
    c3_img_path=None,
    c3_fov=u.R_sun.to(u.m, 30),
    c3_dx=0,
    c3_dy=0,
    soho_r=u.au.to(u.m, 1),
    soho_lat=0,
    soho_lon=0,
    **kwargs
):
    """Fits FRi3D model to coronagraph image.

    Args:

    Returns:
        StaticFRi3D: Fitted static FRi3D model.
    """
    sfr = StaticFRi3D()
    sfr.modify(**kwargs)
    x0, y0, z0 = sfr.shell()
    ncols = (
        (cor2b_img_path is not None)
        + (c2_img_path is not None)
        + (c3_img_path is not None)
        + (cor2a_img_path is not None)
    )
    nrows = 2
    plt.figure(figsize=((ncols + 1) * 2, (nrows + 1) * 2))
    gs = gridspec.GridSpec(
        nrows,
        ncols,
        wspace=0,
        hspace=0,
        top=1.0 - 0.5 / (nrows + 1),
        bottom=0.5 / (nrows + 1),
        left=0.5 / (ncols + 1),
        right=1 - 0.5 / (ncols + 1),
    )
    i = 0
    if cor2b_img_path is not None:
        cor2b_fov = stb_r * np.tan(cor2b_aov)
        ax = plt.subplot(gs[i])
        ax.imshow(
            plt.imread(cor2b_img_path),
            zorder=0,
            extent=[-cor2b_fov - cor2b_dx, cor2b_fov - cor2b_dx, -cor2b_fov - cor2b_dy, cor2b_fov - cor2b_dy],
        )
        ax.set_xlim([-cor2b_fov - cor2b_dx, cor2b_fov - cor2b_dx])
        ax.set_ylim([-cor2b_fov - cor2b_dy, cor2b_fov - cor2b_dy])
        ax.set_facecolor("black")
        plt.axis("off")
        ax = plt.subplot(gs[i + ncols])
        ax.imshow(
            plt.imread(cor2b_img_path),
            zorder=0,
            extent=[-cor2b_fov - cor2b_dx, cor2b_fov - cor2b_dx, -cor2b_fov - cor2b_dy, cor2b_fov - cor2b_dy],
        )
        T = cs.mx_rot_y(stb_lat) * cs.mx_rot_z(-stb_lon)
        x, y, z = cs.mx_apply(T, x0, y0, z0)
        y = stb_r / (stb_r - x) * y
        z = stb_r / (stb_r - x) * z
        ax.scatter(y, z, 3, color=BLIND_PALETTE["yellow"], marker=".")
        ax.set_xlim([-cor2b_fov - cor2b_dx, cor2b_fov - cor2b_dx])
        ax.set_ylim([-cor2b_fov - cor2b_dy, cor2b_fov - cor2b_dy])
        ax.set_facecolor("black")
        plt.axis("off")
        i += 1
    if c3_img_path is not None:
        ax = plt.subplot(gs[i])
        ax.imshow(
            plt.imread(c3_img_path), zorder=0, extent=[-c3_fov - c3_dx, c3_fov - c3_dx, -c3_fov - c3_dy, c3_fov - c3_dy]
        )
        ax.set_xlim([-c3_fov - c3_dx, c3_fov - c3_dx])
        ax.set_ylim([-c3_fov - c3_dy, c3_fov - c3_dy])
        ax.set_facecolor("black")
        plt.axis("off")
        ax = plt.subplot(gs[i + ncols])
        ax.imshow(
            plt.imread(c3_img_path), zorder=0, extent=[-c3_fov - c3_dx, c3_fov - c3_dx, -c3_fov - c3_dy, c3_fov - c3_dy]
        )
        T = cs.mx_rot_y(soho_lat) * cs.mx_rot_z(-soho_lon)
        x, y, z = cs.mx_apply(T, x0, y0, z0)
        y = soho_r / (soho_r - x) * y
        z = soho_r / (soho_r - x) * z
        ax.scatter(y, z, 3, color=BLIND_PALETTE["yellow"], marker=".")
        ax.set_xlim([-c3_fov - c3_dx, c3_fov - c3_dx])
        ax.set_ylim([-c3_fov - c3_dy, c3_fov - c3_dy])
        ax.set_facecolor("black")
        plt.axis("off")
        i += 1
    if c2_img_path is not None:
        ax = plt.subplot(gs[i])
        ax.imshow(
            plt.imread(c2_img_path), zorder=0, extent=[-c2_fov - c2_dx, c2_fov - c2_dx, -c2_fov - c2_dy, c2_fov - c2_dy]
        )
        ax.set_xlim([-c2_fov - c2_dx, c2_fov - c2_dx])
        ax.set_ylim([-c2_fov - c2_dy, c2_fov - c2_dy])
        ax.set_facecolor("black")
        plt.axis("off")
        ax = plt.subplot(gs[i + ncols])
        ax.imshow(
            plt.imread(c2_img_path), zorder=0, extent=[-c2_fov - c2_dx, c2_fov - c2_dx, -c2_fov - c2_dy, c2_fov - c2_dy]
        )
        T = cs.mx_rot_y(soho_lat) * cs.mx_rot_z(-soho_lon)
        x, y, z = cs.mx_apply(T, x0, y0, z0)
        y = soho_r / (soho_r - x) * y
        z = soho_r / (soho_r - x) * z
        ax.scatter(y, z, 3, color=BLIND_PALETTE["yellow"], marker=".")
        ax.set_xlim([-c2_fov - c2_dx, c2_fov - c2_dx])
        ax.set_ylim([-c2_fov - c2_dy, c2_fov - c2_dy])
        ax.set_facecolor("black")
        plt.axis("off")
        i += 1
    if cor2a_img_path is not None:
        cor2a_fov = sta_r * np.tan(cor2a_aov)
        ax = plt.subplot(gs[i])
        ax.imshow(
            plt.imread(cor2a_img_path),
            zorder=0,
            extent=[-cor2a_fov - cor2a_dx, cor2a_fov - cor2a_dx, -cor2a_fov - cor2a_dy, cor2a_fov - cor2a_dy],
        )
        ax.set_xlim([-cor2a_fov - cor2a_dx, cor2a_fov - cor2a_dx])
        ax.set_ylim([-cor2a_fov - cor2a_dy, cor2a_fov - cor2a_dy])
        ax.set_facecolor("black")
        plt.axis("off")
        ax = plt.subplot(gs[i + ncols])
        ax.imshow(
            plt.imread(cor2a_img_path),
            zorder=0,
            extent=[-cor2a_fov - cor2a_dx, cor2a_fov - cor2a_dx, -cor2a_fov - cor2a_dx, cor2a_fov - cor2a_dy],
        )
        T = cs.mx_rot_y(sta_lat) * cs.mx_rot_z(-sta_lon)
        x, y, z = cs.mx_apply(T, x0, y0, z0)
        y = sta_r / (sta_r - x) * y
        z = sta_r / (sta_r - x) * z
        ax.scatter(y, z, 3, color=BLIND_PALETTE["yellow"], marker=".")
        ax.set_xlim([-cor2a_fov - cor2a_dx, cor2a_fov - cor2a_dx])
        ax.set_ylim([-cor2a_fov - cor2a_dy, cor2a_fov - cor2a_dy])
        ax.set_facecolor("black")
        plt.axis("off")
        i += 1
    plt.show()
    return sfr


class BaseProfile:
    """Base profile class."""

    def __init__(self, params=None, bounds=None, relative=None):
        self._params = None
        self._bounds = None
        self._relative = 0
        self.params = params
        self.bounds = bounds
        self.relative = relative

    @property
    def params(self):
        """sequence: parameters of the profile."""
        return self._params

    @params.setter
    def params(self, val):
        self._params = val

    @property
    def bounds(self):
        """sequence(n, 2): fitting bounds for each profile parameter."""
        return self._bounds

    @bounds.setter
    def bounds(self, val):
        self._bounds = val

    @property
    def relative(self):
        """scalar: relative timestamp for the profile."""
        return self._relative

    @relative.setter
    def relative(self, val):
        if np.isscalar(val):
            self._relative = val
        else:
            self._relative = 0


class SignProfile(BaseProfile):
    """Binary sign profile. Corresponds to +1 or -1 values."""

    def signature(self):
        """Returns the sign profile function signature."""
        return "sign(p[0])"

    def eval(self, t):
        """Evaluates the profile function at a given time."""
        return int(np.copysign(1, self.params[0]))


class PolyProfile(BaseProfile):
    """Polynomial profile."""

    def signature(self):
        """Returns the profile function signature."""
        return "p[0]*t^(n-1)+p[1]*t^(n-2)+...+p[n-1]"

    def eval(self, t):
        """Evaluates the profile function at a given time."""
        return np.polyval(self.params, t - self.relative)


class ExpProfile(BaseProfile):
    """Exponential profile."""

    def signature(self):
        """Returns the profile function signature."""
        return "p[0]*exp(p[1]*t)+p[2]"

    def eval(self, t):
        """Evaluates the profile function at a given time."""
        return self.params[0] * np.exp(self.params[1] * (t - self.relative)) + self.params[2]

