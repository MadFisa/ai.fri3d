from ai.fri3d import BaseFRi3DFit
from ai.fri3d import DynamicFRi3D

class FRi3DFitInSitu(BaseFRi3DFit):
    def __init__(self, **kwargs):
        super(FRi3DFitInSitu, self).__init__(DynamicFRi3D, **kwargs)

    def fit(self, **kwargs):
        pass
