import copy
import numpy as np
import pytest
from pytest import approx
from astropy import units as u
from ai.fri3d.model import DynamicFRi3D

class TestDynamicFRi3D:
    _dfr = DynamicFRi3D(
        latitide=lambda t: 0,
        longitude=lambda t: 0,
        toroidal_height=lambda t: 450e3*t,
        poloidal_height=lambda t: u.au.to(u.m, 0.1),
        half_width=lambda t: u.deg.to(u.rad, 40),
        tilt=lambda t: 0,
        flattening=lambda t: 0.5,
        pancaking=lambda t: u.deg.to(u.rad, 30),
        skew=lambda t: 0,
        twist=lambda t: 2,
        flux=lambda t: 5e14,
        sigma=lambda t: 2,
        polarity=lambda t: 1,
        chirality=lambda t: 1
    )

    def test_snapshot(self):
        sfr = self._dfr.snapshot(u.au.to(u.m, 1)/450e3)
        assert sfr.toroidal_height == approx(u.au.to(u.m, 1))

    def test_insitu(self):
        data = self._dfr.insitu(
            np.linspace(u.au.to(u.m, 0.9)/450e3, u.au.to(u.m, 1.1)/450e3, 100),
            u.au.to(u.m, 1), 0, 0
        )
        assert data[0].ndim == 2
        assert not np.all(np.isnan(data[0]))
        assert np.all(data[0][:, 1] > 0)
        assert data[0][0, 2] < 0 and data[0][-1, 2] > 0
        assert np.all(data[0][:, 0] == approx(0))
        assert data[1].ndim == 1
        assert not np.all(np.isnan(data[1]))
        assert np.all(data[1] == 450e3)

    def test_impact(self):
        data = self._dfr.impact(
            np.linspace(u.au.to(u.m, 0.9)/450e3, u.au.to(u.m, 1.1)/450e3, 100),
            u.au.to(u.m, 1), 0, 0
        )
        assert data[0] == approx(0, abs=1e-3)
        assert data[1] == approx(u.au.to(u.m, 1)/450e3)
