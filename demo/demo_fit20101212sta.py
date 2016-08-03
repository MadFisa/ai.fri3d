
from ai.fri3d.optimize import fit2remote, fit2insitu
from ai.fri3d import Evolution
from astropy import units as u
from datetime import datetime, timedelta
import numpy as np
from ai.shared.data import getSTA
from matplotlib import pyplot as plt
from ai.shared.color import BLIND_PALETTE

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
            u.au.to(u.m, [0.02, 0.2])
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

def demo_insitu(
    t0=1292253600.0,
    period=3.0*24.0*3600.0,
    step=600.0,
    latitude=lambda t: -5.76484870e-04,
    longitude=lambda t: 1.08243448e+00,
    toroidal_height=lambda t: np.polyval(
        np.array([
            u.Unit('km/s').to(u.Unit('m/s'), 480.0), 
            u.au.to(u.m, 0.5)
        ]), 
        t
    ),
    poloidal_height=lambda t: 1.52375410e+10,
    half_width=lambda t: 9.24032770e-01,
    tilt=lambda t: 1.64470079e-02,
    flattening=lambda t: 4.57895263e-01,
    pancaking=lambda t: 4.83729672e-01,
    skew=lambda t: 0.0,
    twist=lambda t: 4.12708116e+00,
    flux=lambda t: 4.29414200e+14,
    sigma=lambda t: 2.0,
    polarity=-1.0,
    chirality=1.0,
    spline_s_phi_kind='cubic',
    spline_s_phi_n=500):

    d1 = datetime(2010, 12, 15, 10, 20)
    d2 = datetime(2010, 12, 16, 4)
    dd = d2-d1

    t, b, p = getSTA(
        # datetime(2010, 12, 15, 19)-timedelta(hours=12),
        # datetime(2010, 12, 15, 19)+timedelta(hours=12)
        d1-dd*0.8,
        d2+dd*0.8
    )

    evo = Evolution(
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
        spline_s_phi_kind=spline_s_phi_kind,
        spline_s_phi_n=spline_s_phi_n)

    tt = np.arange(0.0, period+step, step)

    bb = evo.insitu(
        tt, 
        x=np.mean(p[:,0]),
        y=np.mean(p[:,1]),
        z=np.mean(p[:,2])
    )

    nonzero_indices = np.nonzero(np.sqrt(bb[:,0]**2+bb[:,1]**2+bb[:,2]**2))[0]

    if nonzero_indices.size >= 2:
        tt = tt[nonzero_indices[0]:nonzero_indices[-1]+1]
        bb = bb[nonzero_indices[0]:nonzero_indices[-1]+1,:]

    tt = np.array([datetime.fromtimestamp(tt) for tt in tt+t0+600])
    bb = u.T.to(u.nT, bb)

    b = u.T.to(u.nT, b)

    fig = plt.figure()
    plt.plot(t, np.sqrt(b[:,0]**2+b[:,1]**2+b[:,2]**2), 'k')
    plt.plot(tt, np.sqrt(bb[:,0]**2+bb[:,1]**2+bb[:,2]**2), color=BLIND_PALETTE['reddish-purple'], linewidth=4, linestyle='dashed')
    plt.plot(t, b[:,0], color=BLIND_PALETTE['vermillion'])
    plt.plot(tt, bb[:,0], color=BLIND_PALETTE['reddish-purple'], linewidth=4, linestyle='dashed')
    plt.plot(t, b[:,1], color=BLIND_PALETTE['bluish-green'])
    plt.plot(tt, bb[:,1], color=BLIND_PALETTE['reddish-purple'], linewidth=4, linestyle='dashed')
    plt.plot(t, b[:,2], color=BLIND_PALETTE['blue'])
    plt.plot(tt, bb[:,2], color=BLIND_PALETTE['reddish-purple'], linewidth=4, linestyle='dashed')
    plt.show()

#             remote  insitu
# theta        -14.5     0.0
# varphi        55.0    62.0
# Rp                     0.1
# varphi_hw     55.0    52.9
# gamma         16.0     0.9
# n              0.6     0.46
# theta_p       23.0    27.7
# tau                    4.1
# Phi                 4.3e14

# demo_fit2remote()

# Run 01.08
# [ -1.39915561e-02   9.03661698e-01   1.49928626e+10   1.19992133e+00
#    3.39431978e-02   5.59760053e-01   3.99451581e-01   6.22890240e+00
#    3.35992244e+14]
# Run 02.08
# 1292253600.0
# [ -5.76484870e-04   1.08243448e+00   1.52375410e+10   9.24032770e-01
#    1.64470079e-02   4.57895263e-01   4.83729672e-01   4.12708116e+00
#    4.29414200e+14]
# demo_fit2insitu()

demo_insitu()