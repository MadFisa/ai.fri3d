
from ai.fri3d.optimize import fit2insitu
import ai.cdas as cdas
import numpy as np
from datetime import datetime
from astropy import units as u

u.nT = u.def_unit('nT', 1e-9*u.T)

# [  8.51140005 -19.76761041   0.12946792   1.13415351   0.47783838
#    5.57315287   0.83631745]
# 2.71855769001

# latitude=np.pi/180.0*8.51140005, 
# longitude=-np.pi/180.0*19.76761041, 
# toroidal_height=0.7,
# poloidal_height=0.12946792,
# half_width=np.pi/180.0*40, 
# tilt=np.pi/180.0*1.13415351, 
# flattening=0.47783838, 
# pancaking=np.pi/180.0*20.0, 
# skew=np.pi/180.0*0.0, 
# twist=5.57315287, 
# flux=1e14,
# sigma=2.05,
# polarity=-1.0,
# chirality=1.0,
# ratio=0.83631745,

def demo_fit2insitu():
    cdas.set_cache(True, 'data')
    # cdas.set_cache(False)
    data = cdas.get_data(
        'sp_phys', 
        'STA_L1_MAG_RTN', 
        # datetime(2010, 12, 15, 10, 20), 
        # datetime(2010, 12, 15, 13, 30), 
        datetime(2013, 1, 9, 14), 
        # datetime(2010, 12, 16, 4), 
        datetime(2013, 1, 10, 15), 
        ['BFIELD'],
        cdf=True
    )
    t = data['Epoch']
    bx = data['BFIELD'][:,0]
    by = data['BFIELD'][:,1]
    bz = data['BFIELD'][:,2]
    b = np.stack([bx, by, bz], axis=1)*u.nT.to(u.T)
    fit2insitu(t, b,
        latitude=np.array([
            u.deg.to(u.rad, [-15.0, 15.0])
        ]),
        # latitude=u.deg.to(u.rad, 0.0),
        longitude=np.array([
            u.deg.to(u.rad, [-10.0, 10.0])
        ]), 
        # longitude=u.deg.to(u.rad, 0.0),
        toroidal_height=np.array([
            u.Unit('km/s').to(u.Unit('m/s'), [400.0, 500.0]), 
            u.au.to(u.m, [0.6, 0.8])
        ]),
        poloidal_height=np.array([
            u.au.to(u.m, [0.05, 0.25])
        ]), 
        # half_width=np.array([
        #     u.deg.to(u.rad, [30.0, 50.0])
        # ]), 
        half_width=u.deg.to(u.rad, 43.0), 
        tilt=np.array([
            u.deg.to(u.rad, [0.0, 60.0])
        ]), 
        # tilt=u.deg.to(u.rad, 0.0), 
        flattening=np.array([
            [0.4, 0.6]
        ]), 
        # flattening=0.5, 
        # pancaking=np.array([
        #     u.deg.to(u.rad, [10.0, 30.0])
        # ]), 
        pancaking=u.deg.to(u.rad, 18.0), 
        # skew=np.array([
        #     u.deg.to(u.rad, [0.0, 10.0])
        # ]),
        skew=u.deg.to(u.rad, 0.0),
        twist=np.array([
            [0.5, 5.0]
        ]), 
        flux=np.array([
            [1e13, 1e15]
        ]),
        # flux=1e14,
        sigma=np.array([
            [1.0, 3.0]
        ]),
        # sigma=2.05,
        polarity=1.0,
        chirality=1.0,
        max_pre_time=1.0*3600.0,
        max_post_time=2.0*3600.0
    )

demo_fit2insitu()
