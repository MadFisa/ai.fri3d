import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt

from ai import cs
from ai.fri3d.model import StaticFRi3D

sfr = StaticFRi3D(
    toroidal_height=u.au.to(u.m, 1),
    # poloidal_height=u.au.to(u.m, 0.4),
    half_width=u.deg.to(u.rad, 45),
    half_height=np.arctan(0.4),
    # toroidal_height=1,
    # poloidal_height=0.3,
    pancaking=0.2,
    # skew=np.pi/6,
    flattening=0.5,
    flux=1e12,
)

n = 100
r = u.au.to(u.m, np.linspace(1.2, 0.8, n))
theta = np.ones(n) * u.deg.to(u.rad, 0)
phi = np.ones(n) * u.deg.to(u.rad, 0)

x, y, z = cs.sp2cart(r, theta, phi)

# x = u.au.to(u.m, np.linspace(0.8, 1.2, 50))
# x = np.linspace(0.8, 1.2, 50)
# y = np.zeros(50)
# z = np.zeros(50)

b, _ = sfr.data(x, y, z)

b *= 1e9

r = u.m.to(u.au, r)

plt.plot(r, np.linalg.norm(b, axis=1), "k")
plt.plot(r, b[:, 0], "r")
plt.plot(r, b[:, 1], "g")
plt.plot(r, b[:, 2], "b")

plt.show()
