"""The module defines the fitting functions used to fit teh model to
white-light and in-situ data.
"""
# pylint: disable=E1101
# pylint: disable=E1102
# pylint: disable=E0401
# pylint: disable=C0103
# pylint: disable=W0212
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import differential_evolution
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from ai.fri3d.model import StaticFRi3D, DynamicFRi3D

def fit2insitu(t, x, y, z, b, v=None, sampling=50, **kwargs):
    """Fits FRi3D model to in-situ data (magnetic field and speed).

    Args:
        t (np.ndarray): array of timestamps (n)
        b (np.ndarray): array of magnetic field vectors (n, 3)
        v (np.ndarray): array of absolute speed values (n)

    Returns:
        (DynamicFRi3D) fitted dynamic FRi3D model
    """
    t_real = np.linspace(t[0], t[-1], sampling)
    dt_real = t_real[1]-t_real[0]
    f = interp1d(t, b, kind='linear', axis=0)
    b_real = f(t_real)
    mask = np.logical_not(np.any(np.isnan(b_real), axis=1))
    tb_real = t_real[mask]
    b_real = b_real[mask, :]
    if v is not None:
        f = interp1d(t, v, kind='linear', axis=0)
        v_real = f(t_real)
        mask = np.logical_not(np.isnan(v_real))
        tv_real = t_real[mask]
        v_real = v_real[mask]

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
                setattr(dfr, prop, profiles[prop].eval)
    def residual(params):
        i = 0
        for prop, profile in profiles.items():
            if profile.bounds is not None:
                n = len(profile.bounds)
                profile.params = params[i:i+n]
                i += n
                setattr(dfr, prop, profile.eval)
        t_model = np.arange(
            t_real[0]-dt_real*sampling,
            t_real[-1]+dt_real*(sampling+1),
            dt_real
        )
        b_model, v_model = dfr.insitu(t_model, x, y, z)
        nonzero_indices = np.where(np.isfinite(np.sqrt(
            b_model[:, 0]**2+b_model[:, 1]**2+b_model[:, 2]**2
        )))[0]
        if nonzero_indices.size >= 2:
            t_model = t_model[nonzero_indices[0]:nonzero_indices[-1]+1]
            if t_model[0] > t_real[-1] or t_model[-1] < t_real[0]:
                return np.inf
            b_model = b_model[nonzero_indices[0]:nonzero_indices[-1]+1, :]
            v_model = v_model[nonzero_indices[0]:nonzero_indices[-1]+1]
            db = fastdtw(
                np.hstack((np.array([t_model]).T, b_model)),
                np.hstack((np.array([tb_real]).T, b_real)),
                dist=euclidean
            )
            dv = fastdtw(
                np.vstack((t_model, v_model)).T,
                np.vstack((tv_real, v_real)).T,
                dist=euclidean
            ) if v is not None else 0
            return db+dv
        else:
            return np.inf
    bounds = []
    for prop in dfr._props:
        bounds.append(kwargs[prop].bounds)
    res = differential_evolution(residual, bounds=bounds)
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
