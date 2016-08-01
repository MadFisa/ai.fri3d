
from ai.fri3d.optimize import fit2remote, fit2insitu
from astropy import units as u
from datetime import datetime
import numpy as np
from ai.shared.data import getSTA

u.nT = u.def_unit('nT', 1e-9*u.T)

def demo_fit2remote():
    fit2remote(
        cor2a=True,
        cor2a_img='data/cor2a_20101212_083900.png',
        cor2a_aov=u.deg.to(u.rad, 4.0),
        cor2a_xc=-1.16802e7,
        cor2a_yc=-2.43284e7,
        sta_r=u.au.to(u.m, 0.966087),
        sta_lon=u.deg.to(u.rad, 85.198),
        sta_lat=u.deg.to(u.rad, -7.346),

        cor2b=True,
        cor2b_img='data/cor2b_20101212_083900.png',
        cor2b_aov=u.deg.to(u.rad, 4.0),
        cor2b_xc=-3.85197e7,
        cor2b_yc=9.39991e7,
        stb_r=u.au.to(u.m, 1.070067),
        stb_lon=u.deg.to(u.rad, -87.282),
        stb_lat=u.deg.to(u.rad, 7.281),
        
        c3=True,
        c3_img='data/c3_20101212_083934.png',
        c3_fov=u.R_sun.to(u.m, 30.0),
        c3_xc=-2.07815e8,
        c3_yc=-8.50331e8,
        soho_r=u.au.to(u.m, 1.0),
        soho_lat=u.deg.to(u.rad, 0.0),
        soho_lon=u.deg.to(u.rad, 0.0),

        latitude=u.deg.to(u.rad, -14.5),
        longitude=u.deg.to(u.rad, 55.0),
        toroidal_height=u.R_sun.to(u.m, 12.5),
        poloidal_height=u.R_sun.to(u.m, 3.5),
        half_width=u.deg.to(u.rad, 55.0),
        tilt=u.deg.to(u.rad, 16.0),
        flattening=0.6,
        pancaking=u.deg.to(u.rad, 23.0),
        skew=u.deg.to(u.rad, 0.0),
        
        spline_s_phi_kind='cubic',
        spline_s_phi_n=500)

def demo_fit2insitu():
    t, b, p = getSTA(
        datetime(2010, 12, 15, 10, 20),
        datetime(2010, 12, 16, 4)
    )

    fit2insitu(t, b, 
        x=np.mean(p[:,0]),
        y=np.mean(p[:,1]),
        z=np.mean(p[:,2]),
        period=4.0*24.0*3600.0,
        step_coarse=3600.0,
        step_fine=600.0,
        latitude=np.array([
            u.deg.to(u.rad, [-15.0, 0.0])
        ]),
        longitude=np.array([
            u.deg.to(u.rad, [40.0, 70.0])
        ]), 
        toroidal_height=lambda t: np.polyval(
            np.array([
                u.Unit('km/s').to(u.Unit('m/s'), 480.0), 
                u.au.to(u.m, 0.5)
            ]), t
        ),
        poloidal_height=np.array([
            u.au.to(u.m, [0.1, 0.3])
        ]), 
        half_width=np.array([
            u.deg.to(u.rad, [40.0, 70.0])
        ]), 
        tilt=np.array([
            u.deg.to(u.rad, [0.0, 30.0])
        ]), 
        flattening=np.array([
            [0.4, 0.8]
        ]), 
        pancaking=np.array([
            u.deg.to(u.rad, [10.0, 30.0])
        ]), 
        skew=u.deg.to(u.rad, 0.0),
        twist=np.array([
            [1.0, 10.0]
        ]), 
        flux=np.array([
            [1e13, 1e15]
        ]),
        sigma=2.0,
        polarity=-1.0,
        chirality=1.0, 
        spline_s_phi_kind='linear',
        spline_s_phi_n=100,
        max_pre_time=2.0*3600.0,
        max_post_time=2.0*3600.0,
        verbose=True,
        timestamp_mask=None)

demo_fit2insitu()
