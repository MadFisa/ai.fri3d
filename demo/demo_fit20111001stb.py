
from ai.fri3d.optimize import fit2remote, fit2insitu
from astropy import units as u
from datetime import datetime
import numpy as np
from ai.shared.data import getSTB

u.nT = u.def_unit('nT', 1e-9*u.T)

def demo_fit2remote():
    fit2remote(
        cor2a=True,
        cor2a_img='data/cor2a_20111001_233900.png',
        cor2a_aov=u.deg.to(u.rad, 4.0),
        cor2a_xc=3.43595e7-7.15174e7,
        cor2a_yc=2.31816e7-3.21944e7,
        sta_r=u.au.to(u.m, 0.967106),
        sta_lon=u.deg.to(u.rad, 103.933),
        sta_lat=u.deg.to(u.rad, -4.464),

        cor2b=True,
        cor2b_img='data/cor2b_20111001_233900.png',
        cor2b_aov=u.deg.to(u.rad, 4.0),
        cor2b_xc=9.89179e7-4.22797e7,
        cor2b_yc=-7.36381e7+8.75883e7,
        stb_r=u.au.to(u.m, 1.078776),
        stb_lon=u.deg.to(u.rad, -97.833),
        stb_lat=u.deg.to(u.rad, 1.598),
        
        c3=True,
        c3_img='data/c3_20111001_233916.png',
        c3_fov=u.R_sun.to(u.m, 30.0),
        c3_xc=4.62916e8-1.94402e8,
        c3_yc=1.64648e9-8.5263e8,
        soho_r=u.au.to(u.m, 1.0),
        soho_lat=u.deg.to(u.rad, 0.0),
        soho_lon=u.deg.to(u.rad, 0.0),

        latitude=u.deg.to(u.rad, 5.5),
        longitude=u.deg.to(u.rad, -95.0),
        toroidal_height=u.R_sun.to(u.m, 14.9),
        poloidal_height=u.R_sun.to(u.m, 4.5),
        half_width=u.deg.to(u.rad, 75.0),
        tilt=u.deg.to(u.rad, 21.0),
        flattening=0.55,
        pancaking=u.deg.to(u.rad, 27.0),
        skew=u.deg.to(u.rad, 0.0),
        
        spline_s_phi_kind='cubic',
        spline_s_phi_n=500)

def demo_fit2insitu():
    t, b, p = getSTB(
        datetime(2011, 10, 4, 2),
        datetime(2011, 10, 4, 12, 40)
    )

    fit2insitu(t, b, 
        x=np.mean(p[:,0]),
        y=np.mean(p[:,1]),
        z=np.mean(p[:,2]),
        period=3.0*24.0*3600.0,
        step_coarse=3600.0,
        step_fine=600.0,
        latitude=np.array([
            u.deg.to(u.rad, [-5.0, 10.0])
        ]),
        longitude=np.array([
            u.deg.to(u.rad, [-110.0, -80.0])
        ]), 
        toroidal_height = lambda t: np.polyval(
            np.array([
                u.Unit('km/s').to(u.Unit('m/s'), 680.0), 
                u.au.to(u.m, 0.5)
            ]), 
            t
        ),
        poloidal_height=np.array([
            u.au.to(u.m, [0.1, 0.3])
        ]), 
        half_width=np.array([
            u.deg.to(u.rad, [60.0, 90.0])
        ]), 
        tilt=np.array([
            u.deg.to(u.rad, [5.0, 35.0])
        ]), 
        flattening=np.array([
            [0.4, 0.8]
        ]), 
        pancaking=np.array([
            u.deg.to(u.rad, [20.0, 40.0])
        ]), 
        skew=u.deg.to(u.rad, 0.0),
        twist=np.array([
            [1.0, 10.0]
        ]), 
        flux=np.array([
            [1e13, 1e15]
        ]),
        sigma=2.0,
        polarity=1.0,
        chirality=1.0, 
        spline_s_phi_kind='linear',
        spline_s_phi_n=100,
        max_pre_time=2.0*3600.0,
        max_post_time=2.0*3600.0,
        verbose=True,
        timestamp_mask=None)

demo_fit2insitu()
