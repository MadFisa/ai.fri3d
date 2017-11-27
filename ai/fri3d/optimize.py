"""The module defines the fitting functions used to fit the model to
white-light and in-situ data.
"""
# pylint: disable=E1101
# pylint: disable=E1102
# pylint: disable=E0401
# pylint: disable=C0103
# pylint: disable=W0212
from datetime import datetime
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.distance import euclidean
from matplotlib import pyplot as plt
from astropy import units as u
from fastdtw import fastdtw
from ai.fri3d.model import StaticFRi3D, DynamicFRi3D

d_prev = np.inf

def fit2insitu(
        x, y, z, t, b, vt=None,
        sampling=50, relative=True, weights={'t': 1, 'b': 1, 'vt': 1},
        verbose=False, **kwargs):
    """Fits FRi3D model to in-situ data (magnetic field and speed).

    Args:
        t (np.ndarray): array of timestamps (n)
        x (float): x-coordinate of the spacecraft
        y (float): y-coordinate of the spacecraft
        z (float): z-coordinate of the spacecraft
        b (np.ndarray): array of magnetic field vectors (n, 3)
        vt (np.ndarray): array of absolute speed values (n)
        sampling (int): number of sampling points used for fitting
        relative (bool): flag for using relative time-profiles (relative
            to the starting time of the fitted data t[0])
        weights (dict): importance weigths for the fitting
        verbose (bool): verbose mode - print parameters and draw plots
            during fitting
        **kwargs: keyworded profiles for all model parameters

    Returns:
        (DynamicFRi3D) fitted dynamic FRi3D model
        (dict) dictionary of fitted profiles for all the model
            parameters
    """
    t_real = np.linspace(t[0], t[-1], sampling)
    dt_real = t_real[1]-t_real[0]
    m = np.logical_not(np.any(np.isnan(b), axis=1))
    tb_real = t[m]
    b_real = b[m, :]
    bt_real = np.sqrt(b_real[:, 0]**2+b_real[:, 1]**2+b_real[:, 2]**2)
    if vt is not None:
        m = np.logical_not(np.isnan(vt))
        tvt_real = t[m]
        vt_real = vt[m]
    dfr = DynamicFRi3D()
    profiles = {}
    for prop in dfr._props:
        if prop not in kwargs or not isinstance(kwargs[prop], BaseProfile):
            raise TypeError(
                prop+' profile object of BaseProfile class is requried.'
            )
        else:
            profiles[prop] = kwargs[prop]
            if profiles[prop].bounds is None:
                if relative:
                    setattr(
                        dfr,
                        prop,
                        lambda t, profile=profiles[prop]:
                        profile.eval(t-t_real[0])
                    )
                else:
                    setattr(dfr, prop, profiles[prop].eval)
    def residual(params):
        global d_prev
        i = 0
        for prop, profile in profiles.items():
            if profile.bounds is not None:
                n = len(profile.bounds)
                profile.params = params[i:i+n]
                i += n
                if relative:
                    setattr(
                        dfr,
                        prop,
                        lambda t, profile=profile: profile.eval(t-t_real[0])
                    )
                else:
                    setattr(dfr, prop, profile.eval)
        t_model = np.arange(
            t_real[0]-dt_real*sampling,
            t_real[-1]+dt_real*(sampling+1),
            dt_real
        )
        b_model, vt_model = dfr.insitu(t_model, x, y, z)
        bt_model = np.sqrt(b_model[:, 0]**2+b_model[:, 1]**2+b_model[:, 2]**2)
        nonzero_indices = np.where(np.isfinite(bt_model))[0]
        if nonzero_indices.size >= 2:
            t_model = t_model[nonzero_indices[0]:nonzero_indices[-1]+1]
            if t_model[0] > t_real[-1] or t_model[-1] < t_real[0]:
                return np.inf
            b_model = b_model[nonzero_indices[0]:nonzero_indices[-1]+1, :]
            vt_model = vt_model[nonzero_indices[0]:nonzero_indices[-1]+1]
            m = np.logical_and(
                tb_real >= t_model[0],
                tb_real <= t_model[-1]
            )
            db = fastdtw(
                np.hstack((
                    (np.array([t_model]).T-t_real[0])
                    /(t_real[-1]-t_real[0])*weights['t'],
                    b_model/np.amax(bt_real)*weights['b']
                )),
                np.hstack((
                    (np.array([tb_real]).T-t_real[0])
                    /(t_real[-1]-t_real[0])*weights['t'],
                    b_real/np.amax(bt_real)*weights['b']
                )),
                dist=euclidean
            )[0]
            m = np.logical_and(
                tvt_real >= t_model[0],
                tvt_real <= t_model[-1]
            )
            dv = fastdtw(
                np.vstack((
                    (t_model-t_real[0])
                    /(t_real[-1]-t_real[0])*weights['t'],
                    (vt_model-np.amin(vt_real))
                    /(np.amax(vt_real)-np.amin(vt_real))*weights['vt']
                )).T,
                np.vstack((
                    (tvt_real-t_real[0])
                    /(t_real[-1]-t_real[0])*weights['t'],
                    (vt_real-np.amin(vt_real))
                    /(np.amax(vt_real)-np.amin(vt_real))*weights['vt']
                )).T,
                dist=euclidean
            )[0] if vt is not None else 0
            if verbose and db+dv < d_prev:
                d_prev = db+dv

                db_real = np.array(
                    [datetime.fromtimestamp(t) for t in tb_real]
                )
                dv_real = np.array(
                    [datetime.fromtimestamp(t) for t in tvt_real]
                )
                d_model = np.array(
                    [datetime.fromtimestamp(t) for t in t_model]
                )
                plt.close('all')
                fig = plt.figure()
                plt.subplots_adjust(hspace=0.001)
                ax1 = fig.add_subplot(211)
                ax1.plot(
                    db_real,
                    np.sqrt(
                        b_real[:, 0]**2+
                        b_real[:, 1]**2+
                        b_real[:, 2]**2
                    ),
                    'k'
                )
                ax1.plot(
                    db_real,
                    b_real[:, 0],
                    'r'
                )
                ax1.plot(
                    db_real,
                    b_real[:, 1],
                    'g'
                )
                ax1.plot(
                    db_real,
                    b_real[:, 2],
                    'b'
                )
                ax1.plot(
                    d_model,
                    np.sqrt(
                        b_model[:, 0]**2+
                        b_model[:, 1]**2+
                        b_model[:, 2]**2
                    ),
                    '--k'
                )
                ax1.plot(
                    d_model,
                    b_model[:, 0],
                    '--r'
                )
                ax1.plot(
                    d_model,
                    b_model[:, 1],
                    '--g'
                )
                ax1.plot(
                    d_model,
                    b_model[:, 2],
                    '--b'
                )
                ax2 = fig.add_subplot(212, sharex=ax1)
                ax2.plot(
                    dv_real,
                    vt_real,
                    'k'
                )
                ax2.plot(
                    d_model,
                    vt_model,
                    '--k'
                )

                plt.setp(ax1.get_xticklabels(), visible=False)
                plt.ion()
                plt.draw()
                plt.pause(0.1)
                plt.show()

                print('---------------------')
                for prop, profile in profiles.items():
                    if prop in (
                            'latitude', 'longitude', 'half_width', 'tilt',
                            'pancaking', 'skew'):
                        print(prop, u.rad.to(u.deg, profile.params))
                    elif prop in ('toroidal_height', 'poloidal_height'):
                        print(prop, u.m.to(u.au, profile.params))
                    else:
                        print(prop, profile.params)
            return db+dv
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
        maxiter=200,
        popsize=1,
        mutation=0.6702,
        recombination=0.2368
    )
    i = 0
    for prop, profile in profiles.items():
        if profile.bounds is not None:
            n = len(profile.bounds)
            profile.params = res.x[i:i+n]
            i += n
        setattr(dfr, prop, profile.eval)
    return (dfr, profiles)

def fit2cor(**kwargs):
    """Fits FRi3D model to coronagraph image.

    Args:


    Returns:
        (StaticFRi3D) fitted static FRi3D model
    """
    sfr = StaticFRi3D()
    return sfr

def fit2hi(**kwargs):
    sfr = StaticFRi3D()
    return sfr

class BaseProfile:
    """Base profile class."""
    def __init__(self, params=None, bounds=None):
        self._params = None
        self._bounds = None
        self.params = params
        self.bounds = bounds

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

class PolyProfile(BaseProfile):
    """Polynomial profile."""
    def signature(self):
        """Returns the profile function signature."""
        return 'p[0]*t^(n-1)+p[1]*t^(n-2)+...+p[n-1]'

    def eval(self, t):
        """Evaluates the profile function at a given time."""
        return np.polyval(self.params, t)

class ExpProfile(BaseProfile):
    """Exponential profile."""
    def signature(self):
        """Returns the profile function signature."""
        return 'p[0]*exp(p[1]*t)+p[2]'

    def eval(self, t):
        """Evaluates the profile function at a given time."""
        return self.params[0]*np.exp(self.params[1]*t)+self.params[2]
