import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from astropy import units as u
from ai.fri3d.model import StaticFRi3D

sfr = StaticFRi3D(
    toroidal_height=u.au.to(u.m, 1),
    poloidal_height=u.au.to(u.m, 0.2),
    # toroidal_height=1,
    # poloidal_height=0.3,
    pancaking=0.5,
    # skew=np.pi/6,
    flattening=0.5,
    flux=1e12
)

x = u.au.to(u.m, np.linspace(0.8, 1.2, 50))
# x = np.linspace(0.8, 1.2, 50)
y = np.zeros(50)
z = np.zeros(50)

b, _ = sfr.data(x, y, z)

b *= 1e9

plt.plot(x, np.linalg.norm(b, axis=1), 'k')
plt.plot(x, b[:, 0], 'r')
plt.plot(x, b[:, 1], 'g')
plt.plot(x, b[:, 2], 'b')

plt.show()
