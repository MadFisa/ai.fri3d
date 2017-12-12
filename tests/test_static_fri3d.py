import copy
import numpy as np
import pytest
from pytest import approx
from astropy import units as u
from ai.fri3d.model import StaticFRi3D

class TestStaticFRi3D:
    _sfr = StaticFRi3D(
        latitide=0,
        longitude=0,
        toroidal_height=u.au.to(u.m, 1),
        poloidal_height=u.au.to(u.m, 0.1),
        half_width=u.deg.to(u.rad, 40),
        tilt=0,
        flattening=0.5,
        pancaking=u.deg.to(u.rad, 30),
        skew=0,
        twist=2,
        flux=5e14,
        sigma=2
    )

    def test_vanilla_axis_height(self):
        res = self._sfr.vanilla_axis_height(
            np.linspace(-self._sfr.half_width, self._sfr.half_width, 5)
        )
        assert res.ndim == 1 and res.size == 5
        assert not np.any(np.isnan(res))
        res = self._sfr.vanilla_axis_height(
            np.linspace(-self._sfr.half_width, self._sfr.half_width, 5),
            toroidal_height=np.linspace(1, 10, 5)
        )
        assert res.ndim == 1 and res.size == 5
        assert not np.any(np.isnan(res))
        res = self._sfr.vanilla_axis_height(0)
        assert res.ndim == 0

    def test_vanilla_axis_dheight(self):
        res = self._sfr.vanilla_axis_dheight(
            np.linspace(-self._sfr.half_width, self._sfr.half_width, 5)
        )
        assert res.ndim == 1 and res.size == 5
        assert not np.any(np.isnan(res))
        res = self._sfr.vanilla_axis_dheight(
            np.linspace(-self._sfr.half_width, self._sfr.half_width, 5),
            toroidal_height=np.linspace(1, 10, 5)
        )
        assert res.ndim == 1 and res.size == 5
        assert not np.any(np.isnan(res))
        res = self._sfr.vanilla_axis_dheight(0)
        assert res.ndim == 0

    def test_vanilla_axis_distance(self):
        res = self._sfr.vanilla_axis_distance(0, 0, 0)
        assert res.ndim == 0
        assert res == u.au.to(u.m, 1)
        res = self._sfr.vanilla_axis_distance(
            np.linspace(-self._sfr.half_width, self._sfr.half_width, 5),
            u.au.to(u.m, 1),
            0
        )
        assert res.ndim == 1 and res.size == 5
        assert not np.any(np.isnan(res))

    def test_vanilla_axis_min_distance(self):
        res = self._sfr.vanilla_axis_min_distance(u.au.to(u.m, 2), 0)
        assert res[0] == approx(u.au.to(u.m, 1))
        assert res[1] == approx(0)

    def test_vanilla_axis_tan(self):
        assert self._sfr.vanilla_axis_tan(0) == approx(0)
        assert self._sfr.vanilla_axis_tan(
            [-self._sfr.half_width, self._sfr.half_width]
        ) == approx(np.array([np.pi/2, -np.pi/2]))

    def test_vanilla_axis_dlength(self):
        res = self._sfr.vanilla_axis_dlength(0)
        assert res.ndim == 0
        assert not np.isnan(res)
        res = self._sfr.vanilla_axis_dlength(
            [-self._sfr.half_width, self._sfr.half_width]
        )
        assert res.ndim == 1 and res.size == 2
        assert not np.any(np.isnan(res))
        res = self._sfr.vanilla_axis_dlength(
            [-self._sfr.half_width, self._sfr.half_width],
            u.au.to(u.m, [0.9, 1.1])
        )
        assert res.ndim == 1 and res.size == 2
        assert not np.any(np.isnan(res))

    def test_vanilla_axis_length(self):
        res = self._sfr.vanilla_axis_length(
            [-self._sfr.half_width, 0, self._sfr.half_width]
        )
        assert res.ndim == 1 and res.size == 3
        assert res[0]/res[2] < 1e-5
        assert res[2]/res[1] == approx(2)
        sfr = copy.copy(self._sfr)
        sfr.flattening = 1
        sfr.half_width = np.pi/2
        res = sfr.vanilla_axis_length(sfr.half_width)
        assert res == approx(np.pi*sfr.toroidal_height)

    def test_vanilla_axis_phi(self):
        res = self._sfr.vanilla_axis_phi(
            self._sfr.vanilla_axis_length(0)
        )
        assert res == approx(0)
        res = self._sfr.vanilla_axis_phi(
            self._sfr.vanilla_axis_length(
                [-self._sfr.half_width, self._sfr.half_width]
            )
        )
        assert res.ndim == 1 and res.size == 2
        assert res[0] == approx(-self._sfr.half_width)
        assert res[1] == approx(self._sfr.half_width)

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
        x, y, z, b = self._sfr.line(r=0, phi=0)
        assert x.ndim == 1 and y.ndim == 1 and z.ndim == 1
        assert b.ndim == 1
        assert not np.any(np.isnan(np.array([x, y, z])))
        assert not np.any(np.isnan(b))
        x, y, z, b = self._sfr.line(r=0, phi=0, s=0.5)
        assert x.ndim == 0 and y.ndim == 0 and z.ndim == 0
        assert b.ndim == 0
        assert not np.any(np.isnan(np.array([x, y, z])))
        assert not np.isnan(b)
        with pytest.raises(ValueError):
            self._sfr.line(r=2, phi=np.pi/2, s=np.linspace(0, 1, 50))
        with pytest.raises(ValueError):
            self._sfr.line(r=0, phi=np.pi/2, s=10)

    def test_data(self):
        b, vc = self._sfr.data(u.au.to(u.m, 1), 0, 0)
        assert b.ndim == 1
        assert b.size == 3
        assert vc.ndim == 1
        assert vc.size == 2
        assert not np.any(np.isnan(b))
        assert not np.any(np.isnan(vc))
        b, vc = self._sfr.data(
            u.au.to(u.m, np.ones(5)),
            np.zeros(5),
            np.zeros(5)
        )
        assert b.shape == (5, 3)
        assert vc.shape == (5, 2)
        assert not np.any(np.isnan(b))
        assert not np.any(np.isnan(vc))

    def test_axis_min_distance(self):
        res = self._sfr.axis_min_distance(u.au.to(u.m, 1), 0, 0)
        assert res[0] == approx(0, abs=1e-3)
        assert res[1] == approx(np.array([u.au.to(u.m, 1), 0, 0]), abs=1e-3)
        assert res[2] == approx(np.array([0, 1, 0]))

