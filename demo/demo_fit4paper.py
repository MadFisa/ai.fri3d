
from ai.fri3d.optimize import fit2insitu
import ai.cdas as cdas
import numpy as np
from datetime import datetime
from astropy import units as u
import time
from scipy.io import readsav
from matplotlib import pyplot as plt
from ai.shared.data import getVEX, getSTA
from ai.fri3d import Evolution

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
        verbose=True,
        timestamp_mask=lambda t: np.logical_or(
            t <= time.mktime(datetime(2013, 1, 9, 4, 10).timetuple()),
            t >= time.mktime(datetime(2013, 1, 9, 7, 17).timetuple())
        )
    )

def forecast():

    t, b, p = getSTA(
        datetime(2013, 1, 9, 14),
        datetime(2013, 1, 10, 16)
    )
    
    evo = Evolution()
    evo.latitude = lambda t: np.polyval(
        np.array([-1.44628158e-01]), 
        t
    )
    evo.longitude = lambda t: np.polyval(
        np.array([2.40406402e+00]),
        t
    )
    evo.toroidal_height = lambda t: np.polyval(
        np.array([5.19288360e-01, 5.70490815e+05, 5.87875685e+10]),
        t
    )
    evo.poloidal_height = lambda t: np.polyval(
        np.array([2.66249606e+04, 2.63674675e+10]),
        t
    )
    evo.half_width = lambda t: np.polyval(
        np.array([u.deg.to(u.rad, 43.0)]),
        t
    )
    evo.tilt = lambda t: np.polyval(
        np.array([9.86547935e-01]),
        t
    )
    evo.flattening = lambda t: np.polyval(
        np.array([4.90521657e-01]),
        t
    )
    evo.pancaking = lambda t: np.polyval(
        np.array([u.deg.to(u.rad, 18.0)]),
        t
    )
    evo.skew = lambda t: np.polyval(
        np.array([u.deg.to(u.rad, 0.0)]),
        t
    )
    evo.twist = lambda t: np.polyval(
        np.array([2.71957227e+00]),
        t
    )
    evo.flux = lambda t: np.polyval(
        np.array([8.98471009e+14]),
        t
    )
    evo.sigma = lambda t: np.polyval(
        np.array([2.75450672e+00]),
        t
    )
    tm = np.arange(0.0, 4.0*24.0*3600.0, 200)
    bm = evo.insitu(
        tm, 
        np.mean(p[:,0]),
        np.mean(p[:,1]),
        np.mean(p[:,2])
    )
    d = np.array([datetime.fromtimestamp(x) for x in tm+1357617600.0])
    fig = plt.figure()
    plt.plot(t, np.sqrt(b[:,0]**2+b[:,1]**2+b[:,2]**2), 'k')
    plt.plot(t, b[:,0], 'r')
    plt.plot(t, b[:,1], 'g')
    plt.plot(t, b[:,2], 'b')
    plt.plot(d, np.sqrt(bm[:,0]**2+bm[:,1]**2+bm[:,2]**2), '--k')
    plt.plot(d, bm[:,0], '--r')
    plt.plot(d, bm[:,1], '--g')
    plt.plot(d, bm[:,2], '--b')
    plt.show()

fit2vex()
# forecast()
