from ai.fri3d import FRi3D

class BaseFRi3DFit:
    def __init__(self, fri3d_class, **kwargs):
        self._fr = fri3d_class(**kwargs)
        self._latitude = None
        self._longitude = None
        self._toroidal_height = None
        self._poloidal_height = None
        self._half_width = None

    @property
    def fr(self):
        return self._fr
