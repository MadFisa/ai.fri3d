
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

# theta       30.87739911
# phi         120.83144184
# Vt          1100
# Rp          0.0415593209375807
# thetaHW     44.57641555
# gamma       -35.28968224
# n           0.426680311
# thetaP      25.09149408339962
# tau         2.51668966
# F           1.43845472e+14

def demo_fit2insitu():
    t, b, _, p = getVEX(
        datetime(2011, 6, 5, 15, 30),
        datetime(2011, 6, 5, 22, 30)
    )

    # print(
    #     u.m.to(u.au, np.mean(p[:,0])),
    #     u.m.to(u.au, np.mean(p[:,1])),
    #     u.m.to(u.au, np.mean(p[:,2]))
    # )

    # print(np.sqrt(np.mean(p[:,0])**2+np.mean(p[:,1])**2+np.mean(p[:,2])**2))

    # plt.plot(t, b)
    # plt.show()
    
    fit2insitu(t, b, np.ones(t.shape)*1300e3,
        x=np.mean(p[:,0]),
        y=np.mean(p[:,1]),
        z=np.mean(p[:,2]),
        period=2.0*24.0*3600.0,
        step_coarse=1200.0,
        step_fine=300.0,
        latitude=np.array([
            u.deg.to(u.rad, [0.0, 22.0])
        ]),
        longitude=np.array([
            u.deg.to(u.rad, [122.0, 125.0])
        ]), 
        toroidal_height=np.array([
            u.Unit('km/s').to(u.Unit('m/s'), [1300.0, 1300.0]), 
            u.au.to(u.m, [0.5, 0.5])
        ]),
        poloidal_height=np.array([
            u.au.to(u.m, [0.02, 0.008])
        ]), 
        half_width=np.array([
            u.deg.to(u.rad, [35.0, 43.0])
        ]), 
        tilt=np.array([
            u.deg.to(u.rad, [0.0, 35.0])
        ]), 
        flattening=np.array([
            [0.4, 0.5]
        ]), 
        pancaking=np.array([
            u.deg.to(u.rad, [27.0, 30.0])
        ]), 
        skew=u.deg.to(u.rad, 0.0),
        twist=np.array([
            [1.0, 3.0]
        ]), 
        flux=np.array([
            [5e13, 5e14]
        ]),
        sigma=2.0,
        polarity=1.0,
        chirality=1.0, 
        spline_s_phi_kind='linear',
        spline_s_phi_n=100,
        max_pre_time=0.5*3600.0,
        max_post_time=2.0*3600.0,
        verbose=True,
        timestamp_mask=None,
        fit_speed=False)

# demo_fit2remote()
demo_fit2insitu()
