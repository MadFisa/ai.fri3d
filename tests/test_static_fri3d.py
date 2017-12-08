import numpy as np
import pytest
from astropy import units as u
from ai.fri3d.model import StaticFRi3D

class TestStaticFRi3D:
    _sfr = StaticFRi3D()

    def test_shell(self):
        x, y, z = self._sfr.shell(
            s=np.linspace(0, 1, 50),
            phi=np.linspace(0, np.pi*2, 24)
        )
        assert(
            x.shape == (50, 24)
            and y.shape == (50, 24)
            and z.shape == (50, 24)
        )
        assert(
            not np.any(np.isnan(x))
            and not np.any(np.isnan(y))
            and not np.any(np.isnan(z))
        )
        assert(
            x[0, 0] < u.R_sun.to(u.m, 1)
            and y[0, 0] < u.R_sun.to(u.m, 1)
            and z[0, 0] < u.R_sun.to(u.m, 1)
        )
        assert(
            x[-1, -1] < u.R_sun.to(u.m, 1)
            and y[-1, -1] < u.R_sun.to(u.m, 1)
            and z[-1, -1] < u.R_sun.to(u.m, 1)
        )
        x, y, z = self._sfr.shell(s=0.5, phi=np.pi)
        assert x.ndim == 0 and y.ndim == 0 and z.ndim == 0

    def test_line(self):
        xyz, b = self._sfr.line(r=0, phi=0)
        assert xyz.ndim == 2
        assert xyz.shape[1] == 3
        assert b.ndim == 1
        assert not np.any(np.isnan(xyz))
        assert not np.any(np.isnan(b))
        xyz, b = self._sfr.line(r=0, phi=0, s=0.5)
        assert xyz.ndim == 1
        assert xyz.size == 3
        assert b.ndim == 0
        assert not np.any(np.isnan(xyz))
        assert not np.isnan(b)
        with pytest.raises(ValueError):
            self._sfr.line(r=2, phi=np.pi/2, s=np.linspace(0, 1, 50))
        with pytest.raises(ValueError):
            self._sfr.line(r=0, phi=np.pi/2, s=10)

    def test_data(self):
        self._sfr.toroidal_height = u.au.to(u.m, 1)
        b, vc = self._sfr.data(u.au.to(u.m, [1, 0, 0]))
        assert b.ndim == 1
        assert b.size == 3
        assert vc.ndim == 1
        assert vc.size == 2
        assert not np.any(np.isnan(b))
        assert not np.any(np.isnan(vc))
        b, vc = self._sfr.data(
            u.au.to(u.m, np.array([np.ones(5), np.zeros(5), np.zeros(5)]).T)
        )
        assert b.shape == (5, 3)
        assert vc.shape == (5, 2)
        assert not np.any(np.isnan(b))
        assert not np.any(np.isnan(vc))

    def test_axis_min_distance(self): pass
