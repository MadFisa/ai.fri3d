"Sandbox for benchmarking the model."
# pylint: disable=E1101
# pylint: disable=W0611
# pylint: disable=C0103
# pylint: disable=C0413
import numpy as np
from matplotlib import pyplot as plt
from astropy import units as u
from ai.fri3d.model import DynamicFRi3D
from ai.shared import cs

dfr = DynamicFRi3D(
    longitude=lambda t: u.deg.to(u.rad, 0),
    toroidal_height=lambda t: u.au.to(u.m, 0.8)+500e3*t,
    half_width=lambda t: u.deg.to(u.rad, 40),
    pancaking=lambda t: u.deg.to(u.rad, 20),
    poloidal_height=lambda t: u.au.to(u.m, 0.1)+10e3*t,
    flattening=lambda t: 0.4,
    twist=lambda t: 1,
)

t = np.linspace(0, 3600*24*3, 1000)

r = u.au.to(u.m, 1)
theta = u.deg.to(u.rad, 0)
phi = u.deg.to(u.rad, 0)

x, y, z = cs.sp2cart(r, theta, phi)

# b, v = dfr.insitu(t, x, y, z)

# fig = plt.figure()
# ax = fig.add_subplot(211)
# ax.plot(t, b[:, 0]*1e9, 'r')
# ax.plot(t, b[:, 1]*1e9, 'g')
# ax.plot(t, b[:, 2]*1e9, 'b')
# ax.plot(t, np.linalg.norm(b*1e9, axis=1), 'k')
# ax = fig.add_subplot(212)
# ax.plot(t, v*1e-3)

# plt.show()

impact, _ = dfr.impact(t, x, y++u.au.to(u.m, 0.1), z+u.au.to(u.m, 0.2))

print(u.m.to(u.au, impact))
