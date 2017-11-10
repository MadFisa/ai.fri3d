from ai.fri3d import BaseFRi3DFit
from ai.fri3d import DynamicFRi3D

class FRi3DFitInSitu(BaseFRi3DFit):
    def __init__(self, **kwargs):
        super(FRi3DFitInSitu, self).__init__(DynamicFRi3D, **kwargs)

    def fit(self, **kwargs):
        def residual(**kwargs):
            pass

class BaseProfile:
    def __init__(self, profile):
        self._profile = profile

    @property
    def profile(self):
        return self._profile

    def eval(self, t):
        raise NotImplementedError

class PolyProfile(BaseProfile):
    def __init__(self, params, bounds):
        super(PolyProfile, self).__init__('poly')
        self._params = params
        self._bounds = bounds

    def eval(self, t):
        np.polyval(self._params, t)
    
    @property
    def params(self):
        return self._params

    @property
    def bounds(self):
        return self._bounds

class ExpProfile(BaseProfile):
    pass
