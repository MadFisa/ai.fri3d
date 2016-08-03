
from ai.fri3d.optimize import fit2remote, fit2insitu
from ai.fri3d import Evolution
from astropy import units as u
from datetime import datetime, timedelta
import numpy as np
from ai.shared.data import getSTB
from matplotlib import pyplot as plt
import ai.cdas as cdas
from ai.shared.color import BLIND_PALETTE

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
        # datetime(2011, 10, 4, 2),
        datetime(2011, 10, 4, 3, 30),
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
            u.deg.to(u.rad, [-110.0, -70.0])
        ]), 
        toroidal_height = lambda t: np.polyval(
            np.array([
                u.Unit('km/s').to(u.Unit('m/s'), 680.0), 
                u.au.to(u.m, 0.5)
            ]), 
            t
        ),
        poloidal_height=np.array([
            u.Unit('km/s').to(u.Unit('m/s'), [0.0, 40.0]),
            u.au.to(u.m, [0.02, 0.2])
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
        chirality=-1.0, 
        spline_s_phi_kind='linear',
        spline_s_phi_n=100,
        max_pre_time=2.0*3600.0,
        max_post_time=2.0*3600.0,
        verbose=True,
        timestamp_mask=None)

def demo_insitu(
    t0=1317556200.0,
    period=3.0*24.0*3600.0,
    step=600.0,
    latitude=lambda t: 2.70502968e-02,
    longitude=lambda t: -1.33790814e+00,
    toroidal_height=lambda t: np.polyval(
        np.array([
            u.Unit('km/s').to(u.Unit('m/s'), 680.0), 
            u.au.to(u.m, 0.5)
        ]), 
        t
    ),
    poloidal_height=lambda t: 1.22125871e+10,
    half_width=lambda t: 1.28981669e+00,
    tilt=lambda t: 2.18918189e-01,
    flattening=lambda t: 7.44157800e-01,
    pancaking=lambda t: 5.52014484e-01,
    skew=lambda t: 0.0,
    twist=lambda t: 1.20193832e+00,
    flux=lambda t: 5.88473470e+14,
    sigma=lambda t: 2.0,
    polarity=1.0,
    chirality=-1.0,
    spline_s_phi_kind='cubic',
    spline_s_phi_n=500):

    d1 = datetime(2011, 10, 4, 3, 30)
    d2 = datetime(2011, 10, 4, 12, 40)
    dd = d2-d1

    t, b, p = getSTB(
        # datetime(2011, 10, 4, 2),
        # datetime(2011, 10, 4, 8)-timedelta(hours=12),
        # datetime(2011, 10, 4, 8)+timedelta(hours=12)
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

    tt = np.array([datetime.fromtimestamp(tt) for tt in tt+t0])
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
# theta          5.5     1.6
# varphi       -95.0   -76.7
# Rp                    0.08
# varphi_hw     75.0    73.9
# gamma         21.0    12.5
# n              0.55    0.74
# theta_p       27.0    31.6
# tau                    1.2
# Phi                 5.9e14

# demo_fit2remote()

# Run 01.08
# [ -1.41151664e-02  -1.41494278e+00   1.54493304e+10   1.16863167e+00
#    1.68696879e-01   6.16380124e-01   4.15676262e-01   1.93219497e+00
#    5.57677769e+14]
# Run 02.08
# 1317556200.0
# [  2.70502968e-02  -1.33790814e+00   1.22125871e+10   1.28981669e+00
#    2.18918189e-01   7.44157800e-01   5.52014484e-01   1.20193832e+00
#    5.88473470e+14]
# demo_fit2insitu()

demo_insitu()