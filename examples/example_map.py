import numpy as np
from matplotlib import pyplot as plt

from ai.fri3d.model import StaticFRi3D

sfr = StaticFRi3D(
    toroidal_height=1,
    poloidal_height=0.2,
    pancaking=0.6,
    skew=np.pi / 6,
    flattening=0.5,
)

xgrid = np.linspace(-0.2, 0.2, 50)
ygrid = np.linspace(-0.2, 0.2, 50)

bmap = sfr.map(1, 0, 0, [1, 0, 0], [0, 0, 1], xgrid=xgrid, ygrid=ygrid)

plt.pcolormesh(xgrid, ygrid, bmap)
plt.colorbar()

plt.show()
