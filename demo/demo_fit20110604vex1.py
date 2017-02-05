
from ai.fri3d.optimize import fit2remote, fit2insitu
from ai.fri3d import Evolution
from astropy import units as u
from astropy import constants as c
from datetime import datetime, timedelta
import numpy as np
from ai.shared.data import getVEX
from matplotlib import pyplot as plt
from matplotlib import transforms
from matplotlib import dates as mdates
import ai.cdas as cdas
from ai.shared.color import BLIND_PALETTE
from astropy.io import ascii as ascii_
from astropy import table
from matplotlib.colors import LogNorm
from matplotlib import gridspec
from scipy.interpolate import interp1d
import time

u.nT = u.def_unit('nT', 1e-9*u.T)

# 1.0833910918 1307203500.0 [  5.38912279e-01   2.10890650e+00   1.10000000e+06   7.47989354e+10
#    6.21718592e+09   7.78005220e-01  -6.15921147e-01   4.26680311e-01
#    4.37929186e-01   2.51668966e+00   1.43845472e+14]

# 0.874028634756 1307195100.0 [  5.88553373e-01   2.17355456e+00   1.10000000e+06   7.47989354e+10
#    4.43868326e+09   7.04165522e-01  -6.90167148e-01   4.21854015e-01
#    6.09083418e-01   1.78126489e+00   1.27327641e+14]

# 0.860157714189 1307193900.0 [  5.91941439e-01   2.19263396e+00   1.10000000e+06   7.47989354e+10
#    4.23512887e+09   6.98343498e-01  -6.97320648e-01   4.48338453e-01
#    6.43544276e-01   9.45311102e-01   1.23913180e+14]

#!theta       33.91574617
# phi         125.62867193
# Vt          1100
# Rp          0.02831008790554931
#!thetaHW     40.01213509
#!gamma       -39.9535301
#!n           0.448338453
#?thetaP      36.872370944602196
#?tau         0.945311102
# F           1.23913180e+14

def demo_fit2insitu():
    t, b, _, p = getVEX(
        datetime(2011, 6, 5, 8, 45),
        datetime(2011, 6, 5, 11, 50)
    )

    # print(np.sqrt(np.mean(p[:,0])**2+np.mean(p[:,1])**2+np.mean(p[:,2])**2))
    
    fit2insitu(t, b, np.ones(t.shape)*1100e3,
        x=np.mean(p[:,0]),
        y=np.mean(p[:,1]),
        z=np.mean(p[:,2]),
        period=2.0*24.0*3600.0,
        step_coarse=1200.0,
        step_fine=300.0,
        latitude=np.array([
            u.deg.to(u.rad, [-8.0, 34.0])
        ]),
        longitude=np.array([
            u.deg.to(u.rad, [102.0, 130.0])
        ]), 
        toroidal_height=np.array([
            u.Unit('km/s').to(u.Unit('m/s'), [1100.0, 1100.0]), 
            u.au.to(u.m, [0.5, 0.5])
        ]),
        poloidal_height=np.array([
            u.au.to(u.m, [0.02, 0.045])
        ]), 
        half_width=np.array([
            u.deg.to(u.rad, [40.0, 45.0])
        ]), 
        tilt=np.array([
            u.deg.to(u.rad, [-35.0, -40.0])
        ]), 
        flattening=np.array([
            [0.35, 0.45]
        ]), 
        pancaking=np.array([
            u.deg.to(u.rad, [20.0, 38.0])
        ]), 
        skew=u.deg.to(u.rad, 0.0),
        twist=np.array([
            [0.0, 1.0]
        ]), 
        flux=np.array([
            [1e14, 1e15]
        ]),
        sigma=2.0,
        polarity=1.0,
        chirality=1.0, 
        spline_s_phi_kind='linear',
        spline_s_phi_n=100,
        max_pre_time=0.5*3600.0,
        max_post_time=0.5*3600.0,
        verbose=True,
        timestamp_mask=None,
        fit_speed=False)

# demo_fit2remote()
demo_fit2insitu()
