
from ai.fri3d.optimize import fit2remote, fit2insitu
from ai.fri3d import Evolution
from astropy import units as u
from datetime import datetime, timedelta
import numpy as np
from ai.shared.data import getSTB
from matplotlib import pyplot as plt
import ai.cdas as cdas
from ai.shared.color import BLIND_PALETTE
from astropy.io import ascii as ascii_
from astropy import table
from matplotlib.colors import LogNorm
import matplotlib.dates as mdates

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
    t0=1317562200.0,
    period=3.0*24.0*3600.0,
    step=600.0,
    latitude=lambda t: -5.13294711e-03,
    longitude=lambda t: -1.28380991e+00,
    toroidal_height=lambda t: np.polyval(
        np.array([
            u.Unit('km/s').to(u.Unit('m/s'), 680.0), 
            u.au.to(u.m, 0.5)
        ]), 
        t
    ),
    poloidal_height=lambda t: np.polyval(
        np.array([3.96482238e+04, 6.63635890e+09]),
        t
    ),
    half_width=lambda t: 1.38802556e+00,
    tilt=lambda t: 1.47944608e-01,
    flattening=lambda t: 7.11867793e-01,
    pancaking=lambda t: 6.31054126e-01,
    skew=lambda t: 0.0,
    twist=lambda t: 1.14877898e+00,
    flux=lambda t: 6.78825922e+14,
    sigma=lambda t: 2.0,
    polarity=1.0,
    chirality=-1.0,
    spline_s_phi_kind='cubic',
    spline_s_phi_n=500):

    d1 = datetime(2011, 10, 4, 3, 30)
    d2 = datetime(2011, 10, 4, 12, 40)
    dd = d2-d1

    t, b, p = getSTB(
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

    tt = np.array([datetime.utcfromtimestamp(tt) for tt in tt+t0])
    bb = u.T.to(u.nT, bb)

    b = u.T.to(u.nT, b)


    cdas.set_cache(True, 'data')
    data = cdas.get_data(
        'istp_public', 
        'STB_L2_PLA_1DMAX_1MIN', 
        d1-dd*0.8, d2+dd*0.8, 
        ['proton_number_density', 'proton_bulk_speed', 'proton_temperature']
    )
    data['EPOCH'] = np.array(data['EPOCH'])


    pa = table.vstack([
        ascii_.read('data/STB_L2_SWEA_PAD_20111003_V04.cef',data_start=129),
        ascii_.read('data/STB_L2_SWEA_PAD_20111004_V04.cef',data_start=129)
    ])
    pa_time = np.array(pa.columns[0])
    pa_time = np.array([datetime.strptime(t, "%Y-%m-%dT%H:%M:%S.%fZ") for t in pa_time])
    pa_angles = np.array([7.50, 22.50, 37.50, 52.50, 67.50, 82.50, 97.50, 112.50, 127.50, 142.50, 157.50, 172.50])
    pa_energy = table.Table(pa.columns[4:20])
    print(pa_energy[0])
    pa = np.array(table.Table(pa.columns[52:244]))
    pa = np.array([np.array(list(pa[i])) for i in range(pa.size)])
    m = np.logical_and(pa_time >= d1-0.8*dd, pa_time <= d2+0.8*dd)
    pa_time = pa_time[m]
    pa = pa[m,:]
    print(pa.shape)
    pa = (
        # pa[:,3::16]+
        pa[:,4::16]+
        pa[:,5::16]
        # pa[:,6::16]
    )
    print(pa.shape)

    pa = np.transpose(pa)

    


    fig = plt.figure()
    plt.subplots_adjust(hspace=0.001)
    ax = fig.add_subplot(511)
    ax.plot(t, np.sqrt(b[:,0]**2+b[:,1]**2+b[:,2]**2), 'k')
    ax.plot(tt, np.sqrt(bb[:,0]**2+bb[:,1]**2+bb[:,2]**2), color=BLIND_PALETTE['reddish-purple'], linewidth=4, linestyle='dashed')
    ax.plot(t, b[:,0], color=BLIND_PALETTE['vermillion'])
    ax.plot(tt, bb[:,0], color=BLIND_PALETTE['reddish-purple'], linewidth=4, linestyle='dashed')
    ax.plot(t, b[:,1], color=BLIND_PALETTE['bluish-green'])
    ax.plot(tt, bb[:,1], color=BLIND_PALETTE['reddish-purple'], linewidth=4, linestyle='dashed')
    ax.plot(t, b[:,2], color=BLIND_PALETTE['blue'])
    ax.plot(tt, bb[:,2], color=BLIND_PALETTE['reddish-purple'], linewidth=4, linestyle='dashed')
    plt.setp(ax.get_xticklabels(), visible=False)
    
    ax = fig.add_subplot(512)
    ax.imshow(
        pa, 
        aspect='auto', 
        cmap='RdBu_r', 
        norm=LogNorm(), 
        extent=[
            mdates.date2num(d1-0.8*dd),
            mdates.date2num(d2+0.8*dd),
            0.0, 180.0
        ]
    )
    ax.xaxis_date()
    plt.setp(ax.get_xticklabels(), visible=False)
    
    ax = fig.add_subplot(513)
    m = data['SPEED'] >= 0.0
    ax.plot(data['EPOCH'][m], data['SPEED'][m], 'k')
    plt.setp(ax.get_xticklabels(), visible=False)
    
    ax = fig.add_subplot(514)
    m = data['DENSITY'] >= 0.0
    ax.plot(data['EPOCH'][m], data['DENSITY'][m], 'k')
    plt.setp(ax.get_xticklabels(), visible=False)
    
    ax = fig.add_subplot(515)
    m = data['TEMPERATURE'] >= 0.0
    ax.plot(data['EPOCH'][m], data['TEMPERATURE'][m], 'k')

    plt.show()

#             remote  insitu
# theta          5.5    -0.3
# varphi       -95.0   -73.6
# Rp                    0.044+t*36.7km/s
# varphi_hw     75.0    79.5
# gamma         21.0     8.5
# n              0.55    0.71
# theta_p       27.0    36.2
# tau                    1.2
# Phi                 6.8e14

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
# Run 03.08
# 3.53156030898e-09
# 1317555000.0
# [ -4.57181375e-03  -1.29396113e+00   3.96861699e+04   6.65527090e+09
#    1.35619279e+00   1.39100660e-01   7.10848444e-01   6.38561795e-01
#    1.14614495e+00   6.92870250e+14]
# Final
# 3.52774378605e-09
# 1317555000.0 
# [ -5.13294711e-03  -1.28380991e+00   3.96482238e+04   6.63635890e+09
#    1.38802556e+00   1.47944608e-01   7.11867793e-01   6.31054126e-01
#    1.14877898e+00   6.78825922e+14]
# demo_fit2insitu()

demo_insitu()
