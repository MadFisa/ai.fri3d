
import numpy as np
from matplotlib import pyplot as plt
from astropy import units as u
from ai.fri3d import DynamicFRi3D

fr = DynamicFRi3D(
    toroidal_height=lambda t: u.au.to(u.m, 0.5)+500e3*t,
    pancaking=lambda t: u.deg.to(u.rad, 30),
    poloidal_height=lambda t: u.au.to(u.m, 0.1),
    twist=lambda t: 2,
)

t = np.linspace(0, 3600*24*3, 1000)

b, v = fr.insitu(t, u.au.to(u.m, 1), u.au.to(u.m, 0.1), u.au.to(u.m, 0.1))

fig = plt.figure()
ax = fig.add_subplot(211)
ax.plot(t, b[:,0], 'r')
ax.plot(t, b[:,1], 'g')
ax.plot(t, b[:,2], 'b')
ax.plot(t, np.linalg.norm(b, axis=1), 'k')
ax = fig.add_subplot(212)
ax.plot(t, v)

plt.show()
