"""The module defines the fitting functions used to fit teh model to
white-light and in-situ data.
"""
# pylint: disable=E1101
# pylint: disable=E1102
# pylint: disable=C0103
import numpy as np
from ai.fri3d.model import StaticFRi3D, DynamicFRi3D

def fit2insitu(t, b, v=None, **kwargs):
    """Fits FRi3D model to in-situ data (magnetic field and speed).

    Args:
        t (np.ndarray): array of timestamps (n)
        b (np.ndarray): array of magnetic field vectors (n, 3)
        v (np.ndarray): array of absolute speed values (n)

    Returns:
        (DynamicFRi3D) fitted dynamic FRi3D model
    """
    dfr = DynamicFRi3D()
    return dfr

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
    def __init__(self, params, bounds=None):
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
