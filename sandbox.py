"Sandbox for benchmarking the model."
# pylint: disable=E1101
# pylint: disable=W0611
# pylint: disable=C0103
# pylint: disable=C0413
import numpy as np
from matplotlib import pyplot as plt
from astropy import units as u
from ai.fri3d.model import StaticFRi3D

sfr = StaticFRi3D()
# sfr.pancaking = u.deg.to(u.rad, 30)
sfr.poloidal_height = u.au.to(u.m, 0.05)
sfr.toroidal_height = u.au.to(u.m, 0.2)
fmap = sfr.forcemap(u.au.to(u.m, 0.2), 0, 0, [1, 0, 0], [0, 0, 1])
plt.imshow(u.rad.to(u.deg, fmap), origin='lower')
plt.colorbar()
plt.show()
