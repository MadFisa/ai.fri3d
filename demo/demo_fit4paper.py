
from ai.fri3d.optimize import fit2insitu
import ai.cdas as cdas
import numpy as np
from datetime import datetime
from astropy import units as u
import time
from scipy.io import readsav
from matplotlib import pyplot as plt
from ai.shared.data import getVEX, getSTA, getMES
from ai.fri3d import Evolution

u.nT = u.def_unit('nT', 1e-9*u.T)

def fit2mes():
    t, b, p = getMES(
        datetime(2013, 1, 7, 18, 17),
        datetime(2013, 1, 8, 14, 45)
    )

    # r = np.sqrt(p[:,0]**2+p[:,1]**2+p[:,2]**2)*u.m.to(u.au)
    # print(r.min(), r.max())

    # fig = plt.figure()
    # plt.plot(t, b)
    # plt.show()

    fit2insitu(t, b,
        latitude=np.array([
            u.deg.to(u.rad, [-10.0, 0.0])
        ]),
        longitude=np.array([
            u.deg.to(u.rad, [120.0, 150.0])
        ]), 
        toroidal_height=np.array([
            [0.0, 0.2],
            u.Unit('km/s').to(u.Unit('m/s'), [410.0, 440.0]), 
            u.au.to(u.m, [0.2, 0.2])
        ]),
        poloidal_height=np.array([
            u.Unit('km/s').to(u.Unit('m/s'), [0.0, 60.0]), 
            u.au.to(u.m, [0.1, 0.15])
        ]), 
        half_width=u.deg.to(u.rad, 43.0), 
        tilt=np.array([
            u.deg.to(u.rad, [40.0, 60.0])
        ]), 
        flattening=np.array([
            [0.4, 0.7]
        ]), 
        pancaking=u.deg.to(u.rad, 18.0), 
        skew=u.deg.to(u.rad, 0.0),
        twist=np.array([
            [0.1, 3.0]
        ]), 
        flux=np.array([
            [1e14, 1e16]
        ]),
        sigma=np.array([
            [1.0, 3.0]
        ]),
        polarity=1.0,
        chirality=1.0,
        max_pre_time=1.0*3600.0,
        max_post_time=5.0*3600.0,
        x=np.mean(p[:,0]),
        y=np.mean(p[:,1]),
        z=np.mean(p[:,2]),
        verbose=True,
        timestamp_mask=lambda t: np.logical_or.reduce([
            np.logical_and(
                t >= time.mktime(datetime(2013, 1, 7, 18, 17).timetuple()),
                t <= time.mktime(datetime(2013, 1, 7, 22, 52).timetuple())
            ),
            np.logical_and(
                t >= time.mktime(datetime(2013, 1, 8, 1, 28).timetuple()),
                t <= time.mktime(datetime(2013, 1, 8, 6, 52).timetuple())
            ),
            np.logical_and(
                t >= time.mktime(datetime(2013, 1, 8, 9, 37).timetuple()),
                t <= time.mktime(datetime(2013, 1, 8, 14, 45).timetuple())
            )
        ])
    )

def fit2vex():
    t, b, p = getVEX(
        datetime(2013, 1, 8, 18),
        datetime(2013, 1, 9, 16)
    )

    # r = np.sqrt(p[:,0]**2+p[:,1]**2+p[:,2]**2)*u.m.to(u.au)
    # print(r.min(), r.max())

    fit2insitu(t, b,
        latitude=np.array([
            u.deg.to(u.rad, [-10.0, 0.0])
        ]),
        longitude=np.array([
            u.deg.to(u.rad, [110.0, 140.0])
        ]), 
        toroidal_height=np.array([
            # [0.0, 0.2],
            u.Unit('km/s').to(u.Unit('m/s'), [420.0, 450.0]), 
            u.au.to(u.m, [0.4, 0.4])
        ]),
        poloidal_height=np.array([
            u.Unit('km/s').to(u.Unit('m/s'), [0.0, 80.0]), 
            u.au.to(u.m, [0.01, 0.15])
        ]), 
        half_width=u.deg.to(u.rad, 43.0), 
        tilt=np.array([
            u.deg.to(u.rad, [40.0, 80.0])
        ]), 
        flattening=np.array([
            [0.4, 0.7]
        ]), 
        pancaking=u.deg.to(u.rad, 18.0), 
        skew=u.deg.to(u.rad, 0.0),
        twist=np.array([
            [0.1, 3.0]
        ]), 
        flux=np.array([
            [1e14, 1e16]
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

def fit2sta():
    t, b, p = getSTA(
        datetime(2013, 1, 9, 14),
        datetime(2013, 1, 10, 16)
    )

    fit2insitu(t, b,
        latitude=np.array([
            u.deg.to(u.rad, [-10.0, 0.0])
        ]),
        longitude=np.array([
            u.deg.to(u.rad, [110.0, 150.0])
        ]), 
        toroidal_height=np.array([
            u.Unit('km/s').to(u.Unit('m/s'), [450.0, 450.0]), 
            u.au.to(u.m, [0.7, 0.7])
        ]),
        poloidal_height=np.array([
            u.Unit('km/s').to(u.Unit('m/s'), [0.0, 60.0]), 
            u.au.to(u.m, [0.01, 0.25])
        ]), 
        half_width=u.deg.to(u.rad, 43.0), 
        tilt=np.array([
            u.deg.to(u.rad, [30.0, 80.0])
        ]), 
        flattening=np.array([
            [0.4, 0.7]
        ]), 
        pancaking=u.deg.to(u.rad, 18.0), 
        skew=u.deg.to(u.rad, 0.0),
        twist=np.array([
            [0.1, 3.0]
        ]), 
        flux=np.array([
            [1e14, 1e16]
        ]),
        sigma=np.array([
            [1.0, 3.0]
        ]),
        polarity=1.0,
        chirality=1.0,
        max_pre_time=1.0*3600.0,
        max_post_time=2.0*3600.0,
        x=np.mean(p[:,0]),
        y=np.mean(p[:,1]),
        z=np.mean(p[:,2]),
        verbose=True,
        timestamp_mask=lambda t: np.logical_or(
            t <= time.mktime(datetime(2013, 1, 10, 2).timetuple()),
            t >= time.mktime(datetime(2013, 1, 10, 7).timetuple())
        )
    )

# longitude
# 123
# tilt
# 44

# longitude
# 2.37035386e+00 136
# tilt
# 9.46053416e-01 54
# twist
# 1.58728098e+00   
# flux
# 5.47900789e+14   

# longitude
# 2.17625421e+00 125
# tilt
# 1.28848680e+00 74  
# twist
# 1.22226705e+00   
# flux
# 5.18637309e+14   


def forecast():

    t, b, p = getSTA(
        datetime(2013, 1, 9, 14),
        datetime(2013, 1, 10, 16)
    )
    
    evo = Evolution()
    evo.latitude = lambda t: np.polyval(
        np.array([-1.72146822e-01]), 
        t
    )
    evo.longitude = lambda t: np.polyval(
        np.array([2.37035386e+00]),
        t
    )
    evo.toroidal_height = lambda t: np.polyval(
        np.array([8.74162447e-02, 4.48261857e+05, 5.98391483e+10]),
        t
    )
    evo.poloidal_height = lambda t: np.polyval(
        np.array([5.95868918e+04, 1.47818198e+10]),
        t
    )
    evo.half_width = lambda t: np.polyval(
        np.array([u.deg.to(u.rad, 43.0)]),
        t
    )
    evo.tilt = lambda t: np.polyval(
        np.array([9.46053416e-01]),
        t
    )
    evo.flattening = lambda t: np.polyval(
        np.array([6.76780990e-01]),
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
        np.array([1.58728098e+00]),
        t
    )
    evo.flux = lambda t: np.polyval(
        np.array([5.47900789e+14]),
        t
    )
    evo.sigma = lambda t: np.polyval(
        np.array([2.53772719e+00]),
        t
    )
    tm = np.arange(0.0, 4.0*24.0*3600.0, 200)
    bm = evo.insitu(
        tm, 
        np.mean(p[:,0]),
        np.mean(p[:,1]),
        np.mean(p[:,2])
    )
    d = np.array([datetime.fromtimestamp(x) for x in tm+1357588800.0])
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

# fit2mes()
fit2vex()
# fit2sta()
# forecast()
