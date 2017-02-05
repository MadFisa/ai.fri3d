
from ai.fri3d.optimize import fit2remote, fit2insitu
from ai.fri3d import Evolution
from astropy import units as u
from astropy import constants as c
from datetime import datetime, timedelta
import numpy as np
from ai.shared.data import getSTA
from matplotlib import pyplot as plt
from matplotlib import transforms
from matplotlib import dates as mdates
import ai.cdas as cdas
from ai.shared.color import BLIND_PALETTE
from astropy.io import ascii as ascii_
from astropy import table
from matplotlib.colors import LogNorm
from matplotlib import gridspec
from scipy.interpolate import interp1d
import time

u.nT = u.def_unit('nT', 1e-9*u.T)

# theta       27.0
# phi         132.0
# Rt          6.0
# Rp          2.0
# thetaHW     40.0
# gamma       -30.0
# n           0.3
# thetaP      18.0

# def demo_fit2remote():
#     fit2remote(
#         cor2a=True,
#         cor2a_img='data/cor2a_20110604_073900.png',
#         cor2a_aov=u.deg.to(u.rad, 4.0),
#         cor2a_xc=-1.16802e7,
#         cor2a_yc=-2.43284e7,
#         sta_r=u.au.to(u.m, 0.957576),
#         sta_lon=u.deg.to(u.rad, 94.600),
#         sta_lat=u.deg.to(u.rad, 7.338),
#         sta_datetime=datetime(2011,6,4,7,39),

#         cor2b=True,
#         cor2b_img='data/cor2b_20110604_073900.png',
#         cor2b_aov=u.deg.to(u.rad, 4.0),
#         cor2b_xc=-3.85197e7,
#         cor2b_yc=9.39991e7,
#         stb_r=u.au.to(u.m, 1.007703),
#         stb_lon=u.deg.to(u.rad, -93.182),
#         stb_lat=u.deg.to(u.rad, -7.235),
#         stb_datetime=datetime(2011,6,4,7,39),
        
#         c3=True,
#         c3_img='data/c3_20110604_074046.png',
#         c3_fov=u.R_sun.to(u.m, 30.0),
#         c3_xc=-2.07815e8,
#         c3_yc=-8.50331e8,
#         soho_r=u.au.to(u.m, 1.0),
#         soho_lat=u.deg.to(u.rad, 0.0),
#         soho_lon=u.deg.to(u.rad, 0.0),
#         soho_datetime=datetime(2011,6,4,7,40,46),

#         latitude=u.deg.to(u.rad, 27.0),
#         longitude=u.deg.to(u.rad, 132.0),
#         toroidal_height=u.R_sun.to(u.m, 6.0),
#         poloidal_height=u.R_sun.to(u.m, 2.0),
#         half_width=u.deg.to(u.rad, 40.0),
#         tilt=u.deg.to(u.rad, -30.0),
#         flattening=0.3,
#         pancaking=u.deg.to(u.rad, 18.0),
#         skew=u.deg.to(u.rad, 0.0),
        
#         spline_s_phi_kind='cubic',
#         spline_s_phi_n=500)

# theta       34.0
# phi         132.0
# Rt          9.0
# Rp          3.0
# thetaHW     42.0
# gamma       -35.0
# n           0.3
# thetaP      18.0

# def demo_fit2remote():
#     fit2remote(
#         cor2a=True,
#         cor2a_img='data/cor2a_20110604_082400.png',
#         cor2a_aov=u.deg.to(u.rad, 4.0),
#         cor2a_xc=-1.16802e7,
#         cor2a_yc=-2.43284e7,
#         sta_r=u.au.to(u.m, 0.957576),
#         sta_lon=u.deg.to(u.rad, 94.600),
#         sta_lat=u.deg.to(u.rad, 7.338),
#         sta_datetime=datetime(2011,6,4,8,24),

#         cor2b=True,
#         cor2b_img='data/cor2b_20110604_082400.png',
#         cor2b_aov=u.deg.to(u.rad, 4.0),
#         cor2b_xc=-3.85197e7,
#         cor2b_yc=9.39991e7,
#         stb_r=u.au.to(u.m, 1.007703),
#         stb_lon=u.deg.to(u.rad, -93.182),
#         stb_lat=u.deg.to(u.rad, -7.235),
#         stb_datetime=datetime(2011,6,4,8,24),
        
#         c3=True,
#         c3_img='data/c3_20110604_082845.png',
#         c3_fov=u.R_sun.to(u.m, 30.0),
#         c3_xc=-2.07815e8,
#         c3_yc=-8.50331e8,
#         soho_r=u.au.to(u.m, 1.0),
#         soho_lat=u.deg.to(u.rad, 0.0),
#         soho_lon=u.deg.to(u.rad, 0.0),
#         soho_datetime=datetime(2011,6,4,8,28,45),

#         latitude=u.deg.to(u.rad, 34.0),
#         longitude=u.deg.to(u.rad, 132.0),
#         toroidal_height=u.R_sun.to(u.m, 9.0),
#         poloidal_height=u.R_sun.to(u.m, 3.0),
#         half_width=u.deg.to(u.rad, 42.0),
#         tilt=u.deg.to(u.rad, -35.0),
#         flattening=0.3,
#         pancaking=u.deg.to(u.rad, 18.0),
#         skew=u.deg.to(u.rad, 0.0),
        
#         spline_s_phi_kind='cubic',
#         spline_s_phi_n=500)

# theta       34.0
# phi         130.0
# Rt          12.0
# Rp          3.0
# thetaHW     44.0
# gamma       -35.0
# n           0.3
# thetaP      18.0

def demo_fit2remote():
    fit2remote(
        cor2a=True,
        cor2a_img='data/cor2a_20110604_085400.png',
        cor2a_aov=u.deg.to(u.rad, 4.0),
        cor2a_xc=-1.16802e7,
        cor2a_yc=-2.43284e7,
        sta_r=u.au.to(u.m, 0.957576),
        sta_lon=u.deg.to(u.rad, 94.600),
        sta_lat=u.deg.to(u.rad, 7.338),
        sta_datetime=datetime(2011,6,4,8,54),

        cor2b=True,
        cor2b_img='data/cor2b_20110604_085400.png',
        cor2b_aov=u.deg.to(u.rad, 4.0),
        cor2b_xc=-3.85197e7,
        cor2b_yc=9.39991e7,
        stb_r=u.au.to(u.m, 1.007703),
        stb_lon=u.deg.to(u.rad, -93.182),
        stb_lat=u.deg.to(u.rad, -7.235),
        stb_datetime=datetime(2011,6,4,8,54),
        
        c3=True,
        c3_img='data/c3_20110604_090454.png',
        c3_fov=u.R_sun.to(u.m, 30.0),
        c3_xc=-2.07815e8,
        c3_yc=-8.50331e8,
        soho_r=u.au.to(u.m, 1.0),
        soho_lat=u.deg.to(u.rad, 0.0),
        soho_lon=u.deg.to(u.rad, 0.0),
        soho_datetime=datetime(2011,6,4,9,4,54),

        latitude=u.deg.to(u.rad, 34.0),
        longitude=u.deg.to(u.rad, 130.0),
        toroidal_height=u.R_sun.to(u.m, 12.0),
        poloidal_height=u.R_sun.to(u.m, 3.0),
        half_width=u.deg.to(u.rad, 44.0),
        tilt=u.deg.to(u.rad, -35.0),
        flattening=0.3,
        pancaking=u.deg.to(u.rad, 18.0),
        skew=u.deg.to(u.rad, 0.0),
        
        spline_s_phi_kind='cubic',
        spline_s_phi_n=500)

# 0.295820064672 1307280600.0 [ -1.41787762e-01   1.79467028e+00   1.02446274e+06   7.47989354e+10
#    5.21459437e+09   6.91309410e-01  -6.77067398e-01   3.33450211e-01
#    6.20439886e-01   1.87608900e+00   3.04287234e+14]

# 0.297211009311 1307280600.0 [ -1.41713590e-01   1.78074116e+00   1.03461028e+06   7.47989354e+10
#    6.85062459e+09   7.21880038e-01  -6.97403705e-01   4.01758992e-01
#    6.75686178e-01   2.73534510e+00   4.17596950e+14]

# theta       -8.11959061
# phi         102.02895287
# Vt          1034.61028
# Rp          0.04579359691380955
# thetaHW     41.36067949
# gamma       -39.95828891
# n           0.401758992
# thetaP      38.713966274725294
# tau         2.73534510
# F           4.17596950e+14

# 0.302630741777 1307285100.0 [  1.84632438e-02   1.75403112e+00   9.78437981e+05   7.47989354e+10
#    5.10609750e+09   6.89402283e-01  -8.64455026e-01   3.06020000e-01
#    5.21643254e-01   1.60766417e+00   2.43052878e+14]

# theta       1.05786595
# phi         100.49858031
# Vt          978.437981
# Rp          0.0341321535935471
# thetaHW     39.4998412
# gamma       -49.52962457
# n           0.306020000
# thetaP      29.887956865670798
# tau         1.60766417
# F           2.43052878e+14

def demo_fit2insitu():
    t, b, _, p = getSTA(
        datetime(2011, 6, 6, 12, 25),
        datetime(2011, 6, 6, 14, 10)
    )
    cdas.set_cache(True, './data')
    data = cdas.get_data(
        'sp_phys', 
        'STA_L2_PLA_1DMAX_1MIN', 
        datetime(2011, 6, 6, 12, 25),
        datetime(2011, 6, 6, 14, 10),
        ['proton_bulk_speed'],
        cdf=True
    )
    mask = data['proton_bulk_speed'] > 0.0
    f = interp1d(
        np.array([time.mktime(x.timetuple()) for x in data['epoch'][mask]]), 
        data['proton_bulk_speed'][mask], 
        kind='linear',
        fill_value='extrapolate'
    )
    v = f(np.array([time.mktime(x.timetuple()) for x in t]))
    v = u.Unit('km/s').to(u.Unit('m/s'), v)

    fit2insitu(t, b, v,
        x=np.mean(p[:,0]),
        y=np.mean(p[:,1]),
        z=np.mean(p[:,2]),
        period=3.0*24.0*3600.0,
        step_coarse=1200.0,
        step_fine=300.0,
        latitude=np.array([
            u.deg.to(u.rad, [0.0, 26.0])
        ]),
        longitude=np.array([
            u.deg.to(u.rad, [100.0, 130.0])
        ]), 
        toroidal_height=np.array([
            u.Unit('km/s').to(u.Unit('m/s'), [900.0, 1100.0]), 
            u.au.to(u.m, [0.5, 0.5])
        ]),
        poloidal_height=np.array([
            u.au.to(u.m, [0.01, 0.1])
        ]), 
        half_width=np.array([
            u.deg.to(u.rad, [30.0, 60.0])
        ]), 
        tilt=np.array([
            u.deg.to(u.rad, [-50.0, 0.0])
        ]), 
        flattening=np.array([
            [0.3, 0.5]
        ]), 
        pancaking=np.array([
            u.deg.to(u.rad, [15.0, 35.0])
        ]), 
        skew=u.deg.to(u.rad, 0.0),
        twist=np.array([
            [0.1, 2.0]
        ]), 
        flux=np.array([
            [1e13, 1e15]
        ]),
        sigma=2.0,
        polarity=1.0,
        chirality=1.0, 
        spline_s_phi_kind='linear',
        spline_s_phi_n=100,
        max_pre_time=0.5*3600.0,
        max_post_time=1.0*3600.0,
        verbose=True,
        timestamp_mask=None)

def demo_insitu(
    t0=1292253600.0+7200.0,
    period=3.0*24.0*3600.0,
    step=600.0,
    latitude=lambda t: -8.35255421e-04,
    longitude=lambda t: 1.04195088e+00,
    toroidal_height=lambda t: np.polyval(
        np.array([
            u.Unit('km/s').to(u.Unit('m/s'), 480.0), 
            u.au.to(u.m, 0.5)
        ]), 
        t
    ),
    poloidal_height=lambda t: 1.55072350e+10,
    half_width=lambda t: 1.16531074e+00,
    tilt=lambda t: 3.09957231e-03,
    flattening=lambda t: 6.16988347e-01,
    pancaking=lambda t: 5.10604056e-01,
    skew=lambda t: 0.0,
    twist=lambda t: 4.21465864e+00,
    flux=lambda t: 4.66728444e+14,
    sigma=lambda t: 2.0,
    polarity=-1.0,
    chirality=1.0,
    spline_s_phi_kind='cubic',
    spline_s_phi_n=500):

    # d1 = datetime(2010, 12, 15, 10, 20)
    # d2 = datetime(2010, 12, 16, 4)
    d1 = datetime(2010, 12, 15, 12, 20)
    d2 = datetime(2010, 12, 16, 6)
    dd = d2-d1

    print(d1-dd*0.8, d2+dd*0.8)

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

    tt = np.array([datetime.utcfromtimestamp(tt) for tt in tt+t0])
    bb = u.T.to(u.nT, bb)

    b = u.T.to(u.nT, b)

    m = np.logical_or(
        t <= datetime(2010, 12, 15, 1, 5),
        t >= datetime(2010, 12, 15, 4)
    )
    t = t[m]
    b = b[m,:]


    cdas.set_cache(True, 'data')
    data = cdas.get_data(
        'istp_public', 
        'STA_L2_PLA_1DMAX_1MIN', 
        d1-dd*0.8, d2+dd*0.8, 
        ['proton_number_density', 'proton_bulk_speed', 'proton_temperature'],
        cdf=True
    )
    print(data.keys())
    data['epoch'] = np.array(data['epoch'])


    pa = table.vstack([
        ascii_.read('data/STA_L2_SWEA_PAD_20101214_V04.cef',data_start=129),
        ascii_.read('data/STA_L2_SWEA_PAD_20101215_V04.cef',data_start=129),
        ascii_.read('data/STA_L2_SWEA_PAD_20101216_V04.cef',data_start=129)
    ])
    pa_time = np.array(pa.columns[0])
    pa_time = np.array([datetime.strptime(t, "%Y-%m-%dT%H:%M:%S.%fZ") for t in pa_time])
    pa_angles = np.array([7.50, 22.50, 37.50, 52.50, 67.50, 82.50, 97.50, 112.50, 127.50, 142.50, 157.50, 172.50])
    pa_energy = table.Table(pa.columns[4:20])
    print(pa_energy[0])
    print(table.Table(pa.columns[20:36])[0])
    print(table.Table(pa.columns[36:52])[0])
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

    
    major = mdates.DayLocator()
    minor = mdates.HourLocator()
    majorFormat = mdates.DateFormatter('%Y-%m-%d %H:%M')

    fig = plt.figure(figsize=[8,10])

    gs = gridspec.GridSpec(5, 1, height_ratios=[2, 1, 1, 1, 1])

    plt.subplots_adjust(hspace=0.001)
    ax = fig.add_subplot(gs[0])
    ax.plot(t, np.sqrt(b[:,0]**2+b[:,1]**2+b[:,2]**2), 'k', label='B')
    ax.plot(tt, np.sqrt(bb[:,0]**2+bb[:,1]**2+bb[:,2]**2), color=BLIND_PALETTE['reddish-purple'], linewidth=4, linestyle='dashed')
    ax.plot(t, b[:,0], color=BLIND_PALETTE['vermillion'], label='Bx')
    ax.plot(tt, bb[:,0], color=BLIND_PALETTE['reddish-purple'], linewidth=4, linestyle='dashed')
    ax.plot(t, b[:,1], color=BLIND_PALETTE['bluish-green'], label='By')
    ax.plot(tt, bb[:,1], color=BLIND_PALETTE['reddish-purple'], linewidth=4, linestyle='dashed')
    ax.plot(t, b[:,2], color=BLIND_PALETTE['blue'], label='Bz')
    ax.plot(tt, bb[:,2], color=BLIND_PALETTE['reddish-purple'], linewidth=4, linestyle='dashed', label='FRi3D')
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.legend()
    ax.set_ylabel('$B$ $[nT]$')


    ax.xaxis.set_major_locator(major)
    ax.xaxis.set_major_formatter(majorFormat)
    ax.xaxis.set_minor_locator(minor)
    ax.yaxis.set_label_coords(-0.08, 0.5)
    
    ax = fig.add_subplot(gs[1])
    im = ax.imshow(
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
    Bbox = transforms.Bbox.from_bounds(1.02, 0, 0.03, 1)
    trans = ax.transAxes+fig.transFigure.inverted()
    l, b, w, h = transforms.TransformedBbox(Bbox, trans).bounds
    cax = fig.add_axes([l, b, w, h])
    cb = plt.colorbar(im, cax = cax)
    # cb.locator = plt.MaxNLocator(5)
    cb.update_ticks()
    axc = ax
    axc.yaxis.set_label_coords(-0.08, 0.5)
    ax.set_ylabel('$PA$ $[^\circ]$')
    
    ax.xaxis.set_major_locator(major)
    ax.xaxis.set_major_formatter(majorFormat)
    ax.xaxis.set_minor_locator(minor)
    ax.yaxis.set_label_coords(-0.08, 0.5)

    ax = fig.add_subplot(gs[2])
    m = data['proton_bulk_speed'] >= 0.0
    ax.plot(data['epoch'][m], data['proton_bulk_speed'][m], 'k')
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.set_ylabel('$V_p$ $[km/s]$')

    ax.xaxis.set_major_locator(major)
    ax.xaxis.set_major_formatter(majorFormat)
    ax.xaxis.set_minor_locator(minor)
    ax.yaxis.set_label_coords(-0.08, 0.5)
    
    ax = fig.add_subplot(gs[3])
    m = data['proton_number_density'] >= 0.0
    ax.plot(data['epoch'][m], data['proton_number_density'][m], 'k')
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.set_ylabel('$N_p$ $[cm^{-3}]$')

    ax.xaxis.set_major_locator(major)
    ax.xaxis.set_major_formatter(majorFormat)
    ax.xaxis.set_minor_locator(minor)
    ax.yaxis.set_label_coords(-0.08, 0.5)
    
    ax = fig.add_subplot(gs[4])
    m = data['proton_temperature'] >= 0.0
    ax.plot(data['epoch'][m], data['proton_temperature'][m]/1e6, 'k')
    ax.set_ylabel('$T_p$ $[MK]$')

    ax.xaxis.set_major_locator(major)
    ax.xaxis.set_major_formatter(majorFormat)
    ax.xaxis.set_minor_locator(minor)
    ax.yaxis.set_label_coords(-0.08, 0.5)

    plt.show()

def demo_map(
    t0=1292253600.0+7200.0,
    period=3.0*24.0*3600.0,
    step=600.0,
    latitude=lambda t: -8.35255421e-04,
    longitude=lambda t: 1.04195088e+00,
    toroidal_height=lambda t: np.polyval(
        np.array([
            u.Unit('km/s').to(u.Unit('m/s'), 480.0), 
            u.au.to(u.m, 0.5)
        ]), 
        t
    ),
    poloidal_height=lambda t: 1.55072350e+10,
    half_width=lambda t: 1.16531074e+00,
    tilt=lambda t: 3.09957231e-03,
    flattening=lambda t: 6.16988347e-01,
    pancaking=lambda t: 5.10604056e-01,
    skew=lambda t: 0.0,
    twist=lambda t: 4.21465864e+00,
    flux=lambda t: 4.66728444e+14,
    sigma=lambda t: 2.0,
    polarity=-1.0,
    chirality=1.0,
    spline_s_phi_kind='cubic',
    spline_s_phi_n=500):
    
    d1 = datetime(2010, 12, 15, 12, 20)
    d2 = datetime(2010, 12, 16, 6)
    dd = d2-d1

    t, b, p = getSTA(
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

    dx = np.linspace(-0.2, 0.3, 500)
    dy = np.linspace(-0.65, 0.4, 500)

    # dx = np.linspace(-0.1, 0.12, 500)
    # dy = np.linspace(-0.07, 0.07, 500)

    # p, _ = evo.impact(
    #     tt, 
    #     x=np.mean(p[:,0]),
    #     y=np.mean(p[:,1]),
    #     z=np.mean(p[:,2])
    # )
    # print(u.m.to(u.au, p))
    # 0.123843505132

    bm = evo.map(
        tt, 
        x=np.mean(p[:,0]),
        y=np.mean(p[:,1]),
        z=np.mean(p[:,2]),
        dx=u.au.to(u.m, dx),
        dy=u.au.to(u.m, dy)
    )

    # dx = -dx+0.12

    plt.contourf(
        dx, 
        dy, 
        u.T.to(u.nT, bm), 
        int(u.T.to(u.nT, np.nanmax(bm)-np.nanmin(bm))/0.1)
    )
    cb = plt.colorbar()
    cb.locator = plt.MaxNLocator(14)
    cb.update_ticks()
    plt.xlabel('X [AU]')
    plt.ylabel('Y [AU]')
    cb.set_label('Bz [nT]')

    # axes projection
    ddx = (max(dx)-min(dx))/10;
    ddy = (max(dy)-min(dy))/10;
    ddx = ddy
    
    xx = np.array([0.41223196,-0.89931677,0.14592516])
    yy = np.array([0.0608059,-0.13265291,-0.98929563])
    zz = np.array([0.90904755,0.41669239,0.0])

    xxx=[np.dot([1,0,0],xx), np.dot([1,0,0],yy), np.dot([1,0,0],zz)];
    yyy=[np.dot([0,1,0],xx), np.dot([0,1,0],yy), np.dot([0,1,0],zz)];
    zzz=[np.dot([0,0,1],xx), np.dot([0,0,1],yy), np.dot([0,0,1],zz)];
    plt.plot((1+np.array([0,xxx[0]]))*ddx, (max(dy)/ddy-1+np.array([0,xxx[1]]))*ddy, '-c', linewidth=3)
    plt.plot((1+np.array([0,yyy[0]]))*ddx, (max(dy)/ddy-1+np.array([0,yyy[1]]))*ddy, '-m', linewidth=3)
    plt.plot((1+np.array([0,zzz[0]]))*ddx, (max(dy)/ddy-1+np.array([0,zzz[1]]))*ddy, '-y', linewidth=3)

    plt.axis('equal')
    plt.show()
    


# demo_fit2remote()
demo_fit2insitu()
