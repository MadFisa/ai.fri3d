from ai.fri3d import BaseFRi3DFit
from ai.fri3d import StaticFRi3D

class FRi3DFitCor(BaseFRi3DFit):
    def __init__(self, **kwargs):
        super(FRi3DFitCor, self).__init__(StaticFRi3D, **kwargs)

    def fit(self, **kwargs):
        pass
