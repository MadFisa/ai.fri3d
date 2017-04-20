
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

# theta       20.0
# phi         125.0
# Rt          7.0
# Rp          2.0
# thetaHW     30.0
# gamma       35.0
# n           0.4
# thetaP      25.0

# def demo_fit2remote():
#     fit2remote(
#         cor2a=True,
#         cor2a_img='data/cor2a_20110604_222400.png',
#         cor2a_aov=u.deg.to(u.rad, 4.0),
#         cor2a_xc=-1.16802e7,
#         cor2a_yc=-2.43284e7,
#         sta_r=u.au.to(u.m, 0.957613),
#         sta_lon=u.deg.to(u.rad, 94.666),
#         sta_lat=u.deg.to(u.rad, 7.333),
#         sta_datetime=datetime(2011,6,4,22,24),

#         cor2b=True,
#         cor2b_img='data/cor2b_20110604_222400.png',
#         cor2b_aov=u.deg.to(u.rad, 4.0),
#         cor2b_xc=-3.85197e7,
#         cor2b_yc=9.39991e7,
#         stb_r=u.au.to(u.m, 1.007948),
#         stb_lon=u.deg.to(u.rad, -93.155),
#         stb_lat=u.deg.to(u.rad, -7.245),
#         stb_datetime=datetime(2011,6,4,22,24),
        
#         c3=True,
#         c3_img='data/c3_20110604_222604.png',
#         c3_fov=u.R_sun.to(u.m, 30.0),
#         c3_xc=-2.07815e8,
#         c3_yc=-8.50331e8,
#         soho_r=u.au.to(u.m, 1.0),
#         soho_lat=u.deg.to(u.rad, 0.0),
#         soho_lon=u.deg.to(u.rad, 0.0),
#         soho_datetime=datetime(2011,6,4,22,26,4),

#         latitude=u.deg.to(u.rad, 20.0),
#         longitude=u.deg.to(u.rad, 125.0),
#         toroidal_height=u.R_sun.to(u.m, 7.0),
#         poloidal_height=u.R_sun.to(u.m, 3.0),
#         half_width=u.deg.to(u.rad, 30.0),
#         tilt=u.deg.to(u.rad, 35.0),
#         flattening=0.4,
#         pancaking=u.deg.to(u.rad, 25.0),
#         skew=u.deg.to(u.rad, 0.0),
        
#         spline_s_phi_kind='cubic',
#         spline_s_phi_n=500)

# theta       22.0
# phi         125.0
# Rt          12.0
# Rp          4.0
# thetaHW     35.0
# gamma       35.0
# n           0.4
# thetaP      30.0

def demo_fit2remote():
    fit2remote(
        cor2a=True,
        cor2a_img='data/cor2a_20110604_225400.png',
        cor2a_aov=u.deg.to(u.rad, 4.0),
        cor2a_xc=-1.16802e7,
        cor2a_yc=-2.43284e7,
        sta_r=u.au.to(u.m, 0.957613),
        sta_lon=u.deg.to(u.rad, 94.666),
        sta_lat=u.deg.to(u.rad, 7.333),
        sta_datetime=datetime(2011,6,4,22,54),

        cor2b=True,
        cor2b_img='data/cor2b_20110604_225400.png',
        cor2b_aov=u.deg.to(u.rad, 4.0),
        cor2b_xc=-3.85197e7,
        cor2b_yc=9.39991e7,
        stb_r=u.au.to(u.m, 1.007948),
        stb_lon=u.deg.to(u.rad, -93.155),
        stb_lat=u.deg.to(u.rad, -7.245),
        stb_datetime=datetime(2011,6,4,22,54),
        
        c3=True,
        c3_img='data/c3_20110604_225534.png',
        c3_fov=u.R_sun.to(u.m, 30.0),
        c3_xc=-2.07815e8,
        c3_yc=-8.50331e8,
        soho_r=u.au.to(u.m, 1.0),
        soho_lat=u.deg.to(u.rad, 0.0),
        soho_lon=u.deg.to(u.rad, 0.0),
        soho_datetime=datetime(2011,6,4,22,55,34),

        latitude=u.deg.to(u.rad, 25.0),
        longitude=u.deg.to(u.rad, 125.0),
        toroidal_height=u.R_sun.to(u.m, 12.0),
        poloidal_height=u.R_sun.to(u.m, 4.0),
        half_width=u.deg.to(u.rad, 35.0),
        tilt=u.deg.to(u.rad, 35.0),
        flattening=0.4,
        pancaking=u.deg.to(u.rad, 30.0),
        skew=u.deg.to(u.rad, 0.0),
        
        spline_s_phi_kind='cubic',
        spline_s_phi_n=500)

# 0.275564042979 1307287200.0 [  3.63941118e-01   2.09495164e+00  -2.06517175e-01   1.19034509e+06
#    7.47989354e+10   1.52249315e+10   7.49200809e-01   3.93149080e-01
#    4.33257718e-01   4.30047827e-01   1.50294339e+00   1.33753703e+14]

# 0.255779532995 1307286600.0 [  3.93140412e-01   2.09770388e+00  -6.48433921e-02   1.19999540e+06
#    7.47989354e+10   1.42774310e+10   7.60668810e-01   4.70469688e-01
#    4.40465792e-01   4.65410526e-01   1.44487636e+00   1.36128083e+14]

# 0.247121901797 1307286600.0 [  2.89251104e-01   2.06836731e+00  -2.33888431e-01   1.23500195e+06
#    7.47989354e+10   1.21881663e+10   6.55987062e-01   4.18358884e-01
#    4.25792852e-01   3.35969551e-01   1.75980424e+00   8.48948181e+13]

# 0.238447116448 1307287200.0 [  2.92256175e-01   2.13357924e+00  -3.85845146e-01   1.29818672e+06
#    7.47989354e+10   1.09761581e+10   7.50550756e-01   4.72097023e-01
#    4.70745851e-01   4.78821828e-01   1.04345531e+00   1.09514269e+14]

# theta       16.74504536
# phi         122.24508571
# at          -0.385845146
# ! Vt        1298.18672
# Rp          0.07337108508724248
# thetaHW     43.00339063
# gamma       27.04916694
# n           0.470745851
# thetaP      27.434469883139027
# tau         1.04345531
# F           1.09514269e+14

def demo_fit2insitu():
    t, b, _, p = getSTA(
        datetime(2011, 6, 6, 16, 30),
        datetime(2011, 6, 7, 1)
    )
    cdas.set_cache(True, './data')
    data = cdas.get_data(
        'sp_phys', 
        'STA_L2_PLA_1DMAX_1MIN', 
        datetime(2011, 6, 6, 16, 30),
        datetime(2011, 6, 7, 1),
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
        period=4.0*24.0*3600.0,
        step_coarse=3600.0,
        step_fine=600.0,
        latitude=np.array([
            u.deg.to(u.rad, [15.0, 35.0])
        ]),
        longitude=np.array([
            u.deg.to(u.rad, [100.0, 140.0])
        ]), 
        toroidal_height=np.array([
            [-5.0, 0.0],
            u.Unit('km/s').to(u.Unit('m/s'), [900.0, 1300.0]), 
            u.au.to(u.m, [0.5, 0.5])
        ]),
        poloidal_height=np.array([
            u.au.to(u.m, [0.02, 0.2])
        ]), 
        half_width=np.array([
            u.deg.to(u.rad, [30.0, 60.0])
        ]), 
        tilt=np.array([
            u.deg.to(u.rad, [15.0, 35.0])
        ]), 
        flattening=np.array([
            [0.4, 0.6]
        ]), 
        pancaking=np.array([
            u.deg.to(u.rad, [15.0, 35.0])
        ]), 
        skew=u.deg.to(u.rad, 0.0),
        twist=np.array([
            [0.0, 3.0]
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
        max_post_time=4.0*3600.0,
        verbose=True,
        timestamp_mask=None)

# 0.238447116448 1307287200.0 [  2.92256175e-01   2.13357924e+00  -3.85845146e-01   1.29818672e+06
#    7.47989354e+10   1.09761581e+10   7.50550756e-01   4.72097023e-01
#    4.70745851e-01   4.78821828e-01   1.04345531e+00   1.09514269e+14]

def demo_insitu(
    t0=1307287200.0+7200.0+3600,
    period=3.0*24.0*3600.0,
    step=600.0,
    latitude=lambda t: 2.92256175e-01,
    longitude=lambda t: 2.13357924e+00,
    toroidal_height=lambda t: np.polyval(
        np.array([
            -3.85845146e-01,
            1.29818672e+06, 
            u.au.to(u.m, 0.5)
        ]), 
        t
    ),
    poloidal_height=lambda t: 1.09761581e+10,
    half_width=lambda t: 7.50550756e-01,
    tilt=lambda t: 4.72097023e-01,
    flattening=lambda t: 4.70745851e-01,
    pancaking=lambda t: 4.78821828e-01,
    skew=lambda t: 0.0,
    twist=lambda t: 1.04345531e+00,
    flux=lambda t: 1.09514269e+14,
    sigma=lambda t: 2.0,
    polarity=1.0,
    chirality=1.0,
    spline_s_phi_kind='cubic',
    spline_s_phi_n=500):

    # d1 = datetime(2010, 12, 15, 10, 20)
    # d2 = datetime(2010, 12, 16, 4)
    d1 = datetime(2011, 6, 6, 16, 30)
    d2 = datetime(2011, 6, 7, 1)
    dd = d2-d1

    print(d1-dd*0.8, d2+dd*0.8)

    t, b, _, p = getSTA(
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

    bb, vv = evo.insitu(
        tt, 
        x=np.mean(p[:,0]),
        y=np.mean(p[:,1]),
        z=np.mean(p[:,2])
    )

    nonzero_indices = np.nonzero(np.sqrt(bb[:,0]**2+bb[:,1]**2+bb[:,2]**2))[0]

    if nonzero_indices.size >= 2:
        tt = tt[nonzero_indices[0]:nonzero_indices[-1]+1]
        bb = bb[nonzero_indices[0]:nonzero_indices[-1]+1,:]
        vv = vv[nonzero_indices[0]:nonzero_indices[-1]+1]

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
        ascii_.read('data/STA_L2_SWEA_PAD_20110606_V04.cef',data_start=129),
        ascii_.read('data/STA_L2_SWEA_PAD_20110607_V04.cef',data_start=129),
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
    ax.plot(tt, vv/1e3, color=BLIND_PALETTE['reddish-purple'], linewidth=4, linestyle='dashed', label='FRi3D')
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
    


# demo_insitu()
demo_fit2remote()
# demo_fit2insitu()
