"""Dynamic FRi3D class definition.

This module defines dynamic FRi3D class. It provides a dynamic
description of a CME (varying in time). The CME parameters are provided
as time-dendent profiles.
"""
# pylint: disable=too-many-instance-attributes
# pylint: disable=E1101
# pylint: disable=C0103
# pylint: disable=E1102
import numpy as np
from ai.fri3d.base_fri3d import BaseFRi3D
from ai.fri3d import StaticFRi3D

class DynamicFRi3D(BaseFRi3D):
    """FRi3D model dynamic class. It provides static description of the
    model.
    """
    def __init__(self, **kwargs):
        super(DynamicFRi3D, self).__init__()
        self.__sfr = StaticFRi3D()
        self.latitude = kwargs.get('latitude', lambda t: self.__sfr.latitude)
        self.longitude = kwargs.get(
            'longitude',
            lambda t: self.__sfr.longitude
        )
        self.toroidal_height = kwargs.get(
            'toroidal_height',
            lambda t: self.__sfr.toroidal_height
        )
        self.poloidal_height = kwargs.get(
            'poloidal_height',
            lambda t: self.__sfr.poloidal_height
        )
        self.half_width = kwargs.get(
            'half_width',
            lambda t: self.__sfr.half_width
        )
        self.tilt = kwargs.get('tilt', lambda t: self.__sfr.tilt)
        self.flattening = kwargs.get(
            'flattening',
            lambda t: self.__sfr.flattening
        )
        self.pancaking = kwargs.get(
            'pancaking',
            lambda t: self.__sfr.pancaking
        )
        self.skew = kwargs.get('skew', lambda t: self.__sfr.skew)
        self.twist = kwargs.get('twist', lambda t: self.__sfr.twist)
        self.flux = kwargs.get('flux', lambda t: self.__sfr.flux)
        self.sigma = kwargs.get('sigma', lambda t: self.__sfr.sigma)
        self.polarity = kwargs.get('polarity', lambda t: self.__sfr.polarity)
        self.chirality = kwargs.get(
            'chirality',
            lambda t: self.__sfr.chirality
        )

    def modify(self, **kwargs):
        """Modify the time profiles."""
        for k, v in kwargs.items():
            if k in self._props:
                setattr(self, k, v)
            else:
                raise KeyError('Unsupported parameter encountered.')

    @BaseFRi3D.latitude.setter
    def latitude(self, func):
        if callable(func):
            self._latitude = func
        else:
            raise ValueError('Latitude profile is expected to be a callable.')

    @BaseFRi3D.longitude.setter
    def longitude(self, func):
        if callable(func):
            self._longitude = func
        else:
            raise ValueError('Longitude profile is expected to be a callable.')

    @BaseFRi3D.toroidal_height.setter
    def toroidal_height(self, func):
        if callable(func):
            self._toroidal_height = func
        else:
            raise ValueError(
                'Toroidal height profile is expected to be a callable.'
            )

    @BaseFRi3D.poloidal_height.setter
    def poloidal_height(self, func):
        if callable(func):
            self._poloidal_height = func
        else:
            raise ValueError(
                'Poloidal height profile is expected to be a callable.'
            )

    @BaseFRi3D.half_width.setter
    def half_width(self, func):
        if callable(func):
            self._half_width = func
        else:
            raise ValueError(
                'Half width profile is expected to be a callable.'
            )

    @BaseFRi3D.tilt.setter
    def tilt(self, func):
        if callable(func):
            self._tilt = func
        else:
            raise ValueError(
                'Tilt profile is expected to be a callable.'
            )

    @BaseFRi3D.flattening.setter
    def flattening(self, func):
        if callable(func):
            self._flattening = func
        else:
            raise ValueError(
                'Flattening profile is expected to be a callable.'
            )

    @BaseFRi3D.pancaking.setter
    def pancaking(self, func):
        if callable(func):
            self._pancaking = func
        else:
            raise ValueError(
                'Pancaking profile is expected to be a callable.'
            )

    @BaseFRi3D.skew.setter
    def skew(self, func):
        if callable(func):
            self._skew = func
        else:
            raise ValueError(
                'Skew profile is expected to be a callable.'
            )

    @BaseFRi3D.twist.setter
    def twist(self, func):
        if callable(func):
            self._twist = func
        else:
            raise ValueError(
                'Twist profile is expected to be a callable.'
            )

    @BaseFRi3D.flux.setter
    def flux(self, func):
        if callable(func):
            self._flux = func
        else:
            raise ValueError(
                'Flux profile is expected to be a callable.'
            )

    @BaseFRi3D.sigma.setter
    def sigma(self, func):
        if callable(func):
            self._sigma = func
        else:
            raise ValueError(
                'Sigma profile is expected to be a callable.'
            )

    @BaseFRi3D.polarity.setter
    def polarity(self, func):
        if callable(func):
            self._polarity = func
        else:
            raise ValueError(
                'Polarity profile is expected to be a callable.'
            )

    @BaseFRi3D.chirality.setter
    def chirality(self, func):
        if callable(func):
            self._chirality = func
        else:
            raise ValueError(
                'Chirality profile is expected to be a callable.'
            )

    def snapshot(self, t):
        """Returns a snapshot of the FRi3D model at a given moment of
        time.

        Args:
            t (float): timestamp

        Returns:
            StaticFRi3D object
        """
        self.__sfr.modify(
            latitude=self.latitude(t),
            longitude=self.longitude(t),
            toroidal_height=self.toroidal_height(t),
            poloidal_height=self.poloidal_height(t),
            half_width=self.half_width(t),
            tilt=self.tilt(t),
            flattening=self.flattening(t),
            pancaking=self.pancaking(t),
            skew=self.skew(t),
            twist=self.twist(t),
            flux=self.flux(t),
            sigma=self.sigma(t),
            polarity=self.polarity,
            chirality=self.chirality
        )
        return self.__sfr

    def insitu(self, t, x, y, z):
        """Calculate synthetic in-situ measurements.

        Args:
            t (float or np.ndarray): time (unix timestamp) for which
                the in-situ measurements are estimated. Can be a single
                time or an array of timestamps.
            x, y, z (float or func): synthetic spacecraft coordinates.
                Can be a single point in space or func(t) which describe
                the spacecraft trajectory.

        Returns:
            (np.ndarray(3), np.ndarray): magnetic field components and
                absolute speed.
        """
        t = np.array(t, copy=False, ndmin=1)
        if not callable(x):
            _x = x
            x = lambda t: _x
        if not callable(y):
            _y = y
            y = lambda t: _y
        if not callable(z):
            _z = z
            z = lambda t: _z
        b = []
        v = []
        for _t in t:
            _b, _c = self.snapshot(_t).data(x(_t), y(_t), z(_t))
            _b = _b[0, :]
            _c = _c[0, :]
            b.append(_b.ravel())
            v.append(
                _c[0]*(self.toroidal_height(_t)-self.toroidal_height(_t-1))+
                _c[1]*(self.poloidal_height(_t)-self.poloidal_height(_t-1))
            )
        return (np.array(b), np.array(v))

    # def impact(self, t, x, y, z):
    #     fr = FRi3D()
    #     fr.polarity = self.polarity
    #     fr.chirality = self.chirality
    #     fr.spline_s_phi_kind = self.spline_s_phi_kind
    #     fr.spline_s_phi_n = self.spline_s_phi_n
    #     impacts = []
    #     times = []
    #     for i, t in enumerate(t):
    #         fr.latitude = self.latitude(t)
    #         fr.longitude = self.longitude(t)
    #         fr.toroidal_height = self.toroidal_height(t)
    #         fr.poloidal_height = self.poloidal_height(t)
    #         fr.half_width = self.half_width(t)
    #         fr.tilt = self.tilt(t)
    #         fr.flattening = self.flattening(t)
    #         fr.pancaking = self.pancaking(t)
    #         fr.skew = self.skew(t)
    #         fr.twist = self.twist(t)
    #         fr.flux = self.flux(t)
    #         fr.sigma = self.sigma(t)
    #         if i == 0:
    #             fr.toroidal_height = 1.0
    #             fr.init()
    #             fr._unit_spline_initial_axis_s_phi = \
    #                 fr._spline_initial_axis_s_phi
    #             fr.toroidal_height = self.toroidal_height(t)
    #             fr.init()
    #         fr._spline_initial_axis_s_phi = lambda s: \
    #             fr._unit_spline_initial_axis_s_phi(s/fr.toroidal_height)
    #         impact, _, _, _, _, _, _ = fr.impact(x, y, z)
    #         impacts.append(impact)
    #         times.append(t)
    #     impacts = np.array(impacts)
    #     times = np.array(times)
    #     index = np.argmin(impacts)
    #     return (impacts[index], times[index])

    # def map(self, t, x, y, z, 
    #     dx=u.au.to(u.m, np.linspace(-0.2, 0.2, 100)),
    #     dy=u.au.to(u.m, np.linspace(-0.2, 0.2, 100))):

    #     _, t = self.impact(t, x, y, z)

    #     fr = FRi3D()
    #     fr.polarity = self.polarity
    #     fr.chirality = self.chirality
    #     fr.spline_s_phi_kind = self.spline_s_phi_kind
    #     fr.spline_s_phi_n = self.spline_s_phi_n
    #     fr.latitude = self.latitude(t)
    #     fr.longitude = self.longitude(t)
    #     fr.toroidal_height = self.toroidal_height(t)
    #     fr.poloidal_height = self.poloidal_height(t)
    #     fr.half_width = self.half_width(t)
    #     fr.tilt = self.tilt(t)
    #     fr.flattening = self.flattening(t)
    #     fr.pancaking = self.pancaking(t)
    #     fr.skew = self.skew(t)
    #     fr.twist = self.twist(t)
    #     fr.flux = self.flux(t)
    #     fr.sigma = self.sigma(t)
    #     fr.toroidal_height = 1.0
    #     fr.init()
    #     fr._unit_spline_initial_axis_s_phi = \
    #         fr._spline_initial_axis_s_phi
    #     fr.toroidal_height = self.toroidal_height(t)
    #     fr.init()

    #     _, xa, ya, za, xt, yt, zt = fr.impact(x, y, z)
    #     vtan = np.array([np.mean(xt), np.mean(yt), np.mean(zt)])
    #     if np.dot(vtan, fr.data(xa, ya, za)) < 0.0:
    #         vtan = -vtan
    #     # vtan = fr.data(xa, ya, za)
    #     # vtan /= np.linalg.norm(vtan)
    #     vsc = np.array([x, y, z])
    #     vsc /= np.linalg.norm(vsc)

    #     vmcy = np.cross(vtan, vsc)
    #     vmcy /= np.linalg.norm(vmcy)
    #     if vmcy[0] < 0.0:
    #         vmcy = -vmcy
    #     vmcx = np.cross(vmcy, vtan)
    #     vmcx /= np.linalg.norm(vmcx)

    #     print(vmcx, vmcy, vtan)

    #     xg = np.zeros([dx.size, dy.size])
    #     yg = np.zeros([dx.size, dy.size])
    #     zg = np.zeros([dx.size, dy.size])

    #     for i in range(dx.size):
    #         for k in range(dy.size):
    #             p = np.array([x, y, z])+dx[i]*vmcx+dy[k]*vmcy
    #             xg[i,k] = p[0]
    #             yg[i,k] = p[1]
    #             zg[i,k] = p[2]
    #     print(
    #         xg.shape, xg.flatten().shape,
    #         yg.shape, yg.flatten().shape,
    #         zg.shape, zg.flatten().shape
    #     )
    #     b = fr.data(xg.flatten(), yg.flatten(), zg.flatten())
    #     print(b.shape)
    #     bmap = np.zeros(b.shape[0])
    #     for i in range(b.shape[0]):
    #         bmap[i] = np.dot(b[i,:], vtan)
    #     bmap = np.reshape(bmap, [dx.size, dy.size])

    #     return bmap.T
