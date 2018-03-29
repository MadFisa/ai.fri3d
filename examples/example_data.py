import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from ai.fri3d.model import StaticFRi3D

sfr = StaticFRi3D(
    toroidal_height=1,
    poloidal_height=0.2,
    pancaking=0.5,
    # skew=np.pi/6,
    flattening=0.5
)

x = np.linspace(0.8, 1.2, 50)
y = np.zeros(x.shape)
z = np.zeros(x.shape)

b, _ = sfr.data(x, y, z)

plt.plot(x, np.linalg.norm(b, axis=1), 'k')
plt.plot(x, b[:, 0], 'r')
plt.plot(x, b[:, 1], 'g')
plt.plot(x, b[:, 2], 'b')

plt.show()
