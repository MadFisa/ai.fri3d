class BaseFRi3D:
    def __init__(self):
        self._latitude = None
    @property
    def latitude(self):
        return self._latitude
    @latitude.setter
    def latitude(self, val):
        self._latitude = val

class StaticFRi3D(BaseFRi3D):
    def __init__(self, **kwargs):
        super(StaticFRi3D, self).__init__()
        self.latitude = kwargs.get('latitude', 0)
    @latitude.setter
    def latitude(self, val):
        self._latitude = subtract_period(val, np.pi*2)

class DynamicFRi3D(BaseFRi3D):
    def __init__(self, **kwargs):
        super(DynamicFRi3D, self).__init__()
        self.__sfr = StaticFRi3D()
        self.latitude = kwargs.get(
            'latitude',
            lambda t: self.__sfr.latitude
        )
    @latitude.setter
    def latitude(self, func):
        if callable(func):
            self._latitude = func

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
