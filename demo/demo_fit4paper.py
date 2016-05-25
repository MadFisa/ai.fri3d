
from ai.fri3d.optimize import fit2insitu
import ai.cdas as cdas
import numpy as np
from datetime import datetime
from astropy import units as u
import time
from scipy.io import readsav
from matplotlib import pyplot as plt
from ai.shared.data import getVEX

u.nT = u.def_unit('nT', 1e-9*u.T)

def fit2vex():
    t, b, p = getVEX(
        datetime(2013, 1, 8, 18),
        datetime(2013, 1, 9, 16)
    )
    fit2insitu(t, b,
        latitude=np.array([
            u.deg.to(u.rad, [-10.0, 10.0])
        ]),
        longitude=np.array([
            u.deg.to(u.rad, [100.0, 140.0])
        ]), 
        toroidal_height=np.array([
            [-5.0, 5.0],
            u.Unit('km/s').to(u.Unit('m/s'), [400.0, 600.0]), 
            u.au.to(u.m, [0.3, 0.4])
        ]),
        poloidal_height=np.array([
            u.Unit('km/s').to(u.Unit('m/s'), [0.0, 100.0]), 
            u.au.to(u.m, [0.01, 0.25])
        ]), 
        half_width=u.deg.to(u.rad, 43.0), 
        tilt=np.array([
            u.deg.to(u.rad, [0.0, 90.0])
        ]), 
        flattening=np.array([
            [0.4, 0.6]
        ]), 
        pancaking=u.deg.to(u.rad, 18.0), 
        skew=u.deg.to(u.rad, 0.0),
        twist=np.array([
            [0.1, 5.0]
        ]), 
        flux=np.array([
            [1e13, 1e16]
        ]),
        sigma=np.array([
            [1.0, 3.0]
        ]),
        polarity=1.0,
        chirality=1.0,
        max_pre_time=2.0*3600.0,
        max_post_time=2.0*3600.0,
        x=np.mean(p[:,0]),
        y=np.mean(p[:,1]),
        z=np.mean(p[:,2]),
        verbose=True
    )

fit2vex()


