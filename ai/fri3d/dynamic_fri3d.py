from ai.fri3d import StaticFRi3D

class DynamicFRi3D:
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
        pass

    def map(self, t, x, y, z):
        pass

    def impact(self, t, x, y, z):
        pass
