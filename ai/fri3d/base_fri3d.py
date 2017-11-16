"""Base FRi3D class. Provides a common interface to model-related
objects.
"""
class BaseFRi3D:
    """Parent class for all FRi3D-related classes. Defines the common
    model properties.
    """
    def __init__(self):
        self._latitude = None
        self._longitude = None
        self._toroidal_height = None
        self._poloidal_height = None
        self._half_width = None
        self._coeff_angle = None
        self._tilt = None
        self._flattening = None
        self._pancaking = None
        self._skew = None
        self._twist = None
        self._flux = None
        self._sigma = None
        self._polarity = None
        self._chirality = None
        self._props = [
            p for p in dir(BaseFRi3D)
            if isinstance(getattr(BaseFRi3D, p), property)
        ]

    @property
    def latitude(self):
        """float, profile function or profile object:
        latitude orientation of CME [rad].
        """
        return self._latitude
    @latitude.setter
    def latitude(self, val):
        self._latitude = val

    @property
    def longitude(self):
        """float, profile function or profile object:
        longitude orientation of CME [rad].
        """
        return self._longitude
    @longitude.setter
    def longitude(self, val):
        self._longitude = val

    @property
    def toroidal_height(self):
        """float, profile function or profile object:
        distance from the origin (Sun) to the apex of
        the CME's axis [m].
        """
        return self._toroidal_height
    @toroidal_height.setter
    def toroidal_height(self, val):
        self._toroidal_height = val

    @property
    def poloidal_height(self):
        """float, profile function or profile object:
        distance from the apex of the CME's axis to its global apex [m].
        """
        return self._poloidal_height
    @poloidal_height.setter
    def poloidal_height(self, val):
        self._poloidal_height = val

    @property
    def half_width(self):
        """float, profile function or profile object:
        angular half width of the CME [rad].
        """
        return self._half_width
    @half_width.setter
    def half_width(self, val):
        self._half_width = val

    @property
    def tilt(self):
        """float, profile function or profile object:
        tilt of the CME, measured from equatorial plane using right-hand
        rule around the axis with origin in the Sun [rad].
        """
        return self._tilt
    @tilt.setter
    def tilt(self, val):
        self._tilt = val

    @property
    def flattening(self):
        """float, profile function or profile object:
        coefficient that controls the front flattening
        of the CME [unitless].
        0 corresponds to total flattening,
        1 corresponds to no flattening, i.e., circular axis.
        """
        return self._flattening
    @flattening.setter
    def flattening(self, val):
        self._flattening = val

    @property
    def pancaking(self):
        """float, profile function or profile object:
        angular half height of the CME, measured in the plane
        of the CME [rad].
        """
        return self._pancaking
    @pancaking.setter
    def pancaking(self, val):
        self._pancaking = val

    @property
    def skew(self):
        """float, profile function or profile object:
        rotational skewing angle of the CME, happens due to rotation
        of the Sun [rad], corresponds to rotation angle of the Sun.
        """
        return self._skew
    @skew.setter
    def skew(self, val):
        self._skew = val

    @property
    def twist(self):
        """float, profile function or profile object:
        constant twist of the flux rope, measured as number of full
        rotations of magnetic fields around CME's axis [unitless].
        """
        return self._twist
    @twist.setter
    def twist(self, val):
        self._twist = val

    @property
    def flux(self):
        """float, profile function or profile object:
        total magnetic flux of the CME [Wb].
        """
        return self._flux
    @flux.setter
    def flux(self, val):
        self._flux = val

    @property
    def sigma(self):
        """float, profile function or profile object:
        sigma coefficient of the Gaussian distribution of total magnetic
        field in cross-section of CME [unitless].
        """
        return self._sigma
    @sigma.setter
    def sigma(self, val):
        self._sigma = val

    @property
    def polarity(self):
        """float, profile function or profile object:
        defines the polarity of the flux rope (direction of the axial
        magnetic field) [unitless].
        +1 corresponds to east-to-west direction of magnetic field from
        footpoint to footpoint.
        -1 corresponds to west-to-east direction of magnetic field from
        footpoint to footpoint.
        """
        return self._polarity
    @polarity.setter
    def polarity(self, val):
        self._polarity = val

    @property
    def chirality(self):
        """float, profile function or profile object:
        defines the chirality (handedness) of the flux rope [unitless].
        +1 correponds to right-handed twist of magnetic field lines.
        -1 correponds to left-handed twist of magnetic field lines.
        """
        return self._chirality
    @chirality.setter
    def chirality(self, val):
        self._chirality = val


class BaseStaticFRi3DFit(BaseFRi3D):
    def __init__(self, **kwargs):
        super(BaseStaticFRi3DFit, self).__init__()
        self.__sfr = StaticFRi3D()
        self.latitude = kwargs.get('latitude', self.__sfr.latitude)
    # same properties as base class
    def fit(self, **kwargs):
        pass

class BaseDynamicFRi3DFit(BaseFRi3D):
    def __init__(self, **kwargs):
        super(BaseDynamicFRi3DFit, self).__init__()
        self.__dfr = DynamicFRi3D()
        self.latitude = kwargs.get(
            'latitude',
            PolyProfile(self.__dfr.latitude(0))
        )
    @latitude.setter
    def latitude(self, prof):
        if isinstance(prof, BaseProfile):
            self._latitude = prof
    def fit(self, **kwargs):
        pass










print(BaseFRi3D.__dict__)  
        # for prop in dir(self.__class__):
        #     if isinstance(getattr(self.__class__, prop), property):
        #         setattr(self, '_'+prop, None)

for prop in BaseFRi3D.PROPS:
    setattr(
        BaseFRi3D,
        prop,
        property(
            lambda self: getattr(self, '_'+prop),
            lambda self, val: setattr(self, '_'+prop, val)
        )
    )



class BaseFRi3DFit(BaseFRi3D):
    def __init__(self, fri3d_class, **kwargs):
        super(BaseFRi3DFit, self).__init__()
        self._fr = fri3d_class(**kwargs)

for prop in BaseFRi3DFit.PROPS:
    setattr(
        BaseFRi3DFit,
        prop,
        property(
            lambda self: getattr(self, '_'+prop),
            lambda self, val: setattr(self, '_'+prop, val)
        )
    )
