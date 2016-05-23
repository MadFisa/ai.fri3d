
from ai.fri3d.optimize import fit2insitu
import ai.cdas as cdas
import numpy as np
from datetime import datetime
from astropy import units as u
import time
from scipy.io import readsav
from matplotlib import pyplot as plt

MESSENGER_SAV = 'demo/MES_2007to2014_HEEQ.sav'
VEX_SAV = 'demo/VEX_2007to2014_HEEQ_removed.sav'
VEX_NPZ = 'demo/vex.npz'
STA_SAV = 'demo/STA_2007to2015_HEEQ.sav'

u.nT = u.def_unit('nT', 1e-9*u.T)

def fit2vex(
    latitude=np.array([
        u.deg.to(u.rad, [-10.0, 10.0])
    ]),
    longitude=np.array([
        u.deg.to(u.rad, [100.0, 140.0])
    ]), 
    toroidal_height=np.array([
        u.Unit('km/s').to(u.Unit('m/s'), [400.0, 700.0]), 
        u.au.to(u.m, [0.5, 0.6])
    ]),
    poloidal_height=np.array([
        u.au.to(u.m, [0.01, 0.2])
    ]), 
    half_width=np.array([
        u.deg.to(u.rad, [30.0, 50.0])
    ]), 
    tilt=np.array([
        u.deg.to(u.rad, [0.0, 60.0])
    ]), 
    flattening=np.array([
        [0.4, 0.6]
    ]), 
    pancaking=np.array([
        u.deg.to(u.rad, [10.0, 30.0])
    ]), 
    skew=np.array([
        u.deg.to(u.rad, [0.0, 10.0])
    ]),
    twist=np.array([
        [0.5, 5.0]
    ]), 
    flux=np.array([
        [1e13, 1e15]
    ]),
    sigma=np.array([
        [1.0, 3.0]
    ]),
    polarity=1.0,
    chirality=1.0,
    max_pre_time=2.0*3600.0,
    max_post_time=2.0*3600.0):

    datetime0 = datetime(2013, 1, 8, 18)
    datetime1 = datetime(2013, 1, 9, 16)

    sav = False
    if sav:
        data = readsav(VEX_SAV, python_dict=True)
        t = np.array([
            datetime.fromtimestamp(x) for x in 
            data['vex']['time']+time.mktime(datetime(1979,1,1).timetuple())
        ])
        mask = np.logical_and(t >= datetime0, t <= datetime1)
        b = np.stack([
            data['vex']['bx'][mask],
            data['vex']['by'][mask],
            data['vex']['bz'][mask]
        ], axis=1)*u.nT.to(u.T)
        np.savez(VEX_NPZ, t=t, b=b)
    
    data = np.load(VEX_NPZ)
    t = data['t']
    b = data['b']

    mask = np.logical_and(t >= datetime0, t <= datetime1)
    t = t[mask]
    b = b[mask,:]

    mask = np.logical_not(np.logical_and.reduce(
        [np.isnan(b[:,0]), np.isnan(b[:,1]), np.isnan(b[:,2])]
    ))
    t = t[mask]
    b = b[mask,:]
    # print(np.any(np.isnan(b)))

    # fig = plt.figure()
    # plt.plot(t, np.sqrt(b[:,0]**2+b[:,1]**2+b[:,2]**2), 'k')
    # plt.plot(t, b[:,0], 'r')
    # plt.plot(t, b[:,1], 'g')
    # plt.plot(t, b[:,2], 'b')
    # plt.show()

    fit2insitu(t, b,
        latitude=latitude,
        longitude=longitude, 
        toroidal_height=toroidal_height,
        poloidal_height=poloidal_height, 
        half_width=half_width, 
        tilt=tilt, 
        flattening=flattening, 
        pancaking=pancaking, 
        skew=skew,
        twist=twist, 
        flux=flux,
        sigma=sigma,
        polarity=polarity,
        chirality=chirality,
        max_pre_time=max_pre_time,
        max_post_time=max_post_time,
        x=-76561214534.109406,
        y=76841408532.553192,
        z=-1491428980.3105478
    )

fit2vex(
    latitude=np.array([
        u.deg.to(u.rad, [-10.0, 10.0])
    ]),
    # longitude=u.deg.to(u.rad, 120.0),
    longitude=np.array([
        u.deg.to(u.rad, [100.0, 140.0])
    ]), 
    toroidal_height=np.array([
        u.Unit('km/s').to(u.Unit('m/s'), [400.0, 800.0]), 
        u.au.to(u.m, [0.3, 0.4])
    ]),
    poloidal_height=np.array([
        u.au.to(u.m, [0.01, 0.2])
    ]), 
    half_width=u.deg.to(u.rad, 43.0), 
    tilt=np.array([
        u.deg.to(u.rad, [0.0, 60.0])
    ]), 
    flattening=np.array([
        [0.4, 0.6]
    ]), 
    pancaking=u.deg.to(u.rad, 18.0), 
    skew=u.deg.to(u.rad, 0.0),
    twist=np.array([
        [0.1, 5.0]
    ]), 
    flux=1e14,
    sigma=2.05,
    polarity=1.0,
    chirality=1.0,
    max_pre_time=2.0*3600.0,
    max_post_time=2.0*3600.0
)
