from ai.fri3d import FRi3D

class BaseFRi3DFit:
    def __init__(self, fri3d_class, **kwargs):
        self._fr = fri3d_class(**kwargs)

    def fit(self, **kwargs):
        raise NotImplementedError

    @property
    def fr(self):
        return self._fr
