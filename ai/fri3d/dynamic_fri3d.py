"""Dynamic FRi3D class definition.

This module defines dynamic FRi3D class. It provides a dynamic
description of a CME (varying in time). The CME parameters are provided
as time-dendent profiles.
"""
# pylint: disable=too-many-instance-attributes
# pylint: disable=E1101
# pylint: disable=C0103
# pylint: disable=C0302
import numpy as np
from ai.fri3d import StaticFRi3D

class DynamicFRi3D:
    """FRi3D model dynamic class. It provides static description of the
    model.
    """
    def __init__(self, **kwargs):
        self._fr = StaticFRi3D()
        self.latitude = kwargs.get('latitude', lambda t: self._fr.latitude)
        self.longitude = kwargs.get('longitude', lambda t: self._fr.longitude)
        self.toroidal_height = kwargs.get(
            'toroidal_height',
            lambda t: self._fr.toroidal_height
        )
        self.poloidal_height = kwargs.get(
            'poloidal_height',
            lambda t: self._fr.poloidal_height
        )
        self.half_width = kwargs.get(
            'half_width',
            lambda t: self._fr.half_width
        )
        self.tilt = kwargs.get('tilt', lambda t: self._fr.tilt)
        self.flattening = kwargs.get(
            'flattening',
            lambda t: self._fr.flattening
        )
        self.pancaking = kwargs.get('pancaking', lambda t: self._fr.pancaking)
        self.skew = kwargs.get('skew', lambda t: self._fr.skew)
        self.twist = kwargs.get('twist', lambda t: self._fr.twist)
        self.flux = kwargs.get('flux', lambda t: self._fr.flux)
        self.sigma = kwargs.get('sigma', lambda t: self._fr.sigma)
        self.polarity = kwargs.get('polarity', self._fr.polarity)
        self.chirality = kwargs.get('chirality', self._fr.chirality)

    def snapshot(self, t):
        self._fr.modify(
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
        return self._fr

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
        for t_ in t:
            b_, c_ = self.snapshot(t_).data(x(t_), y(t_), z(t_))
            b_ = b_[0, :]
            c_ = c_[0, :]
            b.append(b_.ravel())
            v.append(
                c_[0]*(self.toroidal_height(t_)-self.toroidal_height(t_-1))+
                c_[1]*(self.poloidal_height(t_)-self.poloidal_height(t_-1))
            )
        return (np.array(b), np.array(v))

    def map(self, t, x, y, z):
        pass

    def impact(self, t, x, y, z):
        pass
