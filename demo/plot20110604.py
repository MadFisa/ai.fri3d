
import numpy as np
from scipy.interpolate import interp1d

from matplotlib import pyplot as plt
from matplotlib import dates as mdates
from matplotlib import gridspec
from matplotlib.colors import LogNorm
from matplotlib import transforms

import calendar
from datetime import datetime, timedelta

from astropy import units as u
from astropy.io import ascii as ascii_
from astropy import table

from ai.fri3d import Evolution
from ai.shared.data import getMES, getVEX, getSTA
from ai.shared.color import BLIND_PALETTE
import ai.cdas as cdas

step = 600

delta_mes = 6*3600
delta_vex = 6*3600
delta_sta = 6*3600

u.nT = u.def_unit('nT', 1e-9*u.T)

# CME1

# COR
cme1_latitude_cor = u.deg.to(u.rad, 30.0)
cme1_longitude_cor = u.deg.to(u.rad, 110.0)
cme1_toroidal_height_cor = u.R_sun.to(u.m, 12.5)
cme1_poloidal_height_cor = u.R_sun.to(u.m, 3.5)
cme1_half_width_cor = u.deg.to(u.rad, 40.0)
cme1_tilt_cor = u.deg.to(u.rad, 37.0)
cme1_flattening_cor = 0.4
cme1_pancaking_cor = u.deg.to(u.rad, 25.0)
cme1_skew_cor = 0.0
cme1_polarity = -1.0
cme1_chirality = 1.0
cme1_d0_cor = datetime(2011, 6, 4, 8, 54)
cme1_t0_cor = calendar.timegm(cme1_d0_cor.timetuple())
# MES
cme1_d0_mes = datetime(2011, 6, 4, 17, 9)
cme1_d1_mes = datetime(2011, 6, 4, 17, 10)
cme1_t0_mes = calendar.timegm(cme1_d0_mes.timetuple())
cme1_t1_mes = calendar.timegm(cme1_d1_mes.timetuple())
cme1_delta_mes = 10*3600
# VEX
cme1_d0_vex = datetime(2011, 6, 5, 8, 45)
cme1_d1_vex = datetime(2011, 6, 5, 11, 50)
cme1_t0_vex = calendar.timegm(cme1_d0_vex.timetuple())
cme1_t1_vex = calendar.timegm(cme1_d1_vex.timetuple())
cme1_delta_vex = 10*3600
cme1_p_vex = np.array([
    0.00000000e+00, 0.00000000e+00, 1.11498287e+06, 0.00000000e+00,
    1.82344644e+00, 9.32055287e-02, 2.60248888e+14, 8.12884830e-02,
    2.57463174e+00, 1.49408743e+10, 6.98131701e-01, 8.33229899e-01,
    2.06464233e-01, 4.36332313e-01, 2.30014363e+14, 0.00000000e+00,
    0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
    0.00000000e+00, 0.00000000e+00, 0.00000000e+00
])

# CME2

# COR
cme2_latitude_cor = u.deg.to(u.rad, 22.0)
cme2_longitude_cor = u.deg.to(u.rad, 125.0)
cme2_toroidal_height_cor = u.R_sun.to(u.m, 12.0)
cme2_poloidal_height_cor = u.R_sun.to(u.m, 4.0)
cme2_half_width_cor = u.deg.to(u.rad, 35.0)
cme2_tilt_cor = u.deg.to(u.rad, 35.0)
cme2_flattening_cor = 0.4
cme2_pancaking_cor = u.deg.to(u.rad, 30.0)
cme2_skew_cor = 0.0
cme2_polarity = 1.0
cme2_chirality = 1.0
cme2_d0_cor = datetime(2011, 6, 4, 22, 54)
cme2_t0_cor = calendar.timegm(cme2_d0_cor.timetuple())
# MES
cme2_d0_mes = datetime(2011, 6, 5, 4, 40)
cme2_d1_mes = datetime(2011, 6, 5, 9, 29)
cme2_t0_mes = calendar.timegm(cme2_d0_mes.timetuple())
cme2_t1_mes = calendar.timegm(cme2_d1_mes.timetuple())
cme2_delta_mes = 10*3600
# VEX
cme2_d0_vex = datetime(2011, 6, 5, 15, 30)
cme2_d1_vex = datetime(2011, 6, 5, 22, 30)
cme2_t0_vex = calendar.timegm(cme2_d0_vex.timetuple())
cme2_t1_vex = calendar.timegm(cme2_d1_vex.timetuple())
cme2_delta_vex = 10*3600
# cme2_p_vex = np.array([
#     0.00000000e+00, 0.00000000e+00, 1.63103908e+06, 1.68411547e+06,
#     1.95001830e+00, 9.71007874e-01, 5.86563348e+14, 2.13230550e-01,
#     2.11464106e+00, 1.33347207e+10, 6.10865238e-01,-1.86526317e-01,
#     7.83222407e-01, 5.23598776e-01, 4.15008461e+14, 0.00000000e+00,
#     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
#     0.00000000e+00, 0.00000000e+00, 1.00000000e+14
# ])
cme2_p_vex = np.array([
    0.00000000e+00, 0.00000000e+00, 2.46619507e+06, 1.59906971e+06,
    1.71849118e+00, 5.24861205e-01, 1.00000000e+14,-2.71443628e-01,
    1.97068675e+00, 2.78203200e+09, 6.10865238e-01, 4.49940736e-01,
    4.78756335e-01, 5.23598776e-01, 1.86889165e+14, 0.00000000e+00,
    0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
    0.00000000e+00, 0.00000000e+00, 1.00000000e+14, 1.19216084e+05
])
# cme2_p_vex[2] = 1.85e6
# cme2_p_vex[3] = 1.2e6
# cme2_p_vex[9] = u.au.to(u.m, 0.13)

# STA
cme2_d0_sta = datetime(2011, 6, 6, 12, 25)
cme2_d1_sta = datetime(2011, 6, 6, 14, 10)
cme2_t0_sta = calendar.timegm(cme2_d0_sta.timetuple())
cme2_t1_sta = calendar.timegm(cme2_d1_sta.timetuple())
cme2_delta_sta = 10*3600
cme2_p_sta = np.array([
    0.00000000e+00, 0.00000000e+00, 1.06958189e+06, 9.67161777e+05,
    1.86696962e+00, 4.25264445e-01, 1.00000000e+14, 0.00000000e+00,
    0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
    0.00000000e+00, 0.00000000e+00, 1.00000000e+14,-1.49068210e-01,
    1.42215207e+00, 4.30114526e+09, 6.10865238e-01,-1.22030388e+00,
    2.20147966e-01, 5.16012211e-01, 1.95690355e+14
])

# CME 3

# COR
cme3_latitude_cor = u.deg.to(u.rad, -2.0)
cme3_longitude_cor = u.deg.to(u.rad, 92.0)
cme3_toroidal_height_cor = u.R_sun.to(u.m, 16.5)
cme3_poloidal_height_cor = u.R_sun.to(u.m, 4.5)
cme3_half_width_cor = u.deg.to(u.rad, 30.0)
cme3_tilt_cor = u.deg.to(u.rad, 65.0)
cme3_flattening_cor = 0.4
cme3_pancaking_cor = u.deg.to(u.rad, 25.0)
cme3_skew_cor = 0.0
cme3_polarity = 1.0
cme3_chirality = 1.0
cme3_d0_cor = datetime(2011, 6, 4, 23, 56)
cme3_t0_cor = calendar.timegm(cme3_d0_cor.timetuple())
# STA
cme3_d0_sta = datetime(2011, 6, 6, 16, 30)
cme3_d1_sta = datetime(2011, 6, 7, 1)
cme3_t0_sta = calendar.timegm(cme3_d0_sta.timetuple())
cme3_t1_sta = calendar.timegm(cme3_d1_sta.timetuple())
cme3_delta_sta = 10*3600
cme3_p_sta = np.array([
    0.00000000e+00, 0.00000000e+00, 9.95107574e+05, 1.08908611e+06,
    1.90629120e+00, 9.69237318e-01, 0.00000000e+00, 0.00000000e+00,
    0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
    0.00000000e+00, 0.00000000e+00, 1.00000000e+14, 2.80081525e-01,
    2.02642087e+00, 9.42706753e+09, 5.23598776e-01, 5.06115484e-01,
    2.08874976e-01, 4.36332313e-01, 8.61869149e+13
])

di = datetime(2011, 6, 5, 11, 30)
ti = calendar.timegm(di.timetuple())

def insitu_mes():
    d_mes, b_mes, _, p_mes = getMES(
        cme1_d0_mes-timedelta(seconds=delta_mes), 
        cme2_d1_mes+timedelta(seconds=delta_mes)
    )
    t_mes = np.array([calendar.timegm(x.timetuple()) for x in d_mes])
    b_mes = u.T.to(u.nT, b_mes)
    bt_mes = np.sqrt(b_mes[:,0]**2+b_mes[:,1]**2+b_mes[:,2]**2)
    fx_mes = interp1d(
        t_mes, 
        p_mes[:,0], 
        kind='linear', 
        axis=0, 
        fill_value='extrapolate'
    )
    fy_mes = interp1d(
        t_mes, 
        p_mes[:,1], 
        kind='linear', 
        axis=0, 
        fill_value='extrapolate'
    )
    fz_mes = interp1d(
        t_mes, 
        p_mes[:,2], 
        kind='linear', 
        axis=0, 
        fill_value='extrapolate'
    )

    # CME 1

    evo = Evolution()
    p = cme1_p_vex
    evo.toroidal_height = lambda t: p[2]*(t-cme1_t0_cor)+cme1_toroidal_height_cor
    evo.sigma = lambda t: p[4]
    evo.twist = lambda t: p[5]
    evo.skew = lambda t: cme1_skew_cor
    evo.polarity = cme1_polarity
    evo.chirality = cme1_chirality
    evo.latitude = lambda t: p[7]
    evo.longitude = lambda t: p[8]
    evo.poloidal_height = lambda t: p[9]/2.0
    evo.half_width = lambda t: p[10]
    evo.tilt = lambda t: p[11]
    evo.flattening = lambda t: p[12]
    evo.pancaking = lambda t: p[13]
    evo.flux = lambda t: p[6]
    
    tm_mes = np.arange(
        t_mes[0]-cme1_delta_mes, 
        t_mes[-1]+cme1_delta_mes, 
        step, 
        dtype=np.int
    )
    bm_mes, vtm_mes = evo.insitu(
        tm_mes, 
        fx_mes, 
        fy_mes, 
        fz_mes
    )
    bm_mes = u.T.to(u.nT, bm_mes)
    btm_mes = np.sqrt(bm_mes[:,0]**2+bm_mes[:,1]**2+bm_mes[:,2]**2)
    vtm_mes = u.Unit('m/s').to(u.Unit('km/s'), vtm_mes)
    nzi_mes = np.where(np.isfinite(btm_mes))[0]

    if nzi_mes.size > 1 and nzi_mes[0] != 0:
        cme1_tm_mes = tm_mes = tm_mes[nzi_mes]
        cme1_dm_mes = dm_mes = np.array([datetime.utcfromtimestamp(x) for x in tm_mes])
        cme1_bm_mes = bm_mes = bm_mes[nzi_mes,:]
        cme1_btm_mes = btm_mes = btm_mes[nzi_mes]
        cme1_vtm_mes = vtm_mes = vtm_mes[nzi_mes]
    else:
        return 0

    # CME 2

    evo = Evolution()
    p = cme2_p_vex
    # p[2] = 3e6
    # p[9] = u.au.to(u.m, 0.02)
    evo.toroidal_height = lambda t: (
        p[2]*(t-cme2_t0_cor)+cme2_toroidal_height_cor
        if t <= ti else
        p[2]*(ti-cme2_t0_cor)+cme2_toroidal_height_cor+p[3]*(t-ti)
    )
    evo.sigma = lambda t: p[4]
    evo.twist = lambda t: p[5]
    evo.skew = lambda t: cme2_skew_cor
    evo.polarity = cme2_polarity
    evo.chirality = cme2_chirality
    evo.latitude = lambda t: p[7]
    evo.longitude = lambda t: p[8]
    # evo.poloidal_height = lambda t: p[9]/2.0
    evo.poloidal_height = lambda t: cme2_poloidal_height_cor+p[23]*(t-cme2_t0_cor)
    evo.half_width = lambda t: p[10]
    evo.tilt = lambda t: p[11]
    evo.flattening = lambda t: p[12]
    evo.pancaking = lambda t: p[13]
    evo.flux = lambda t: p[6]
    
    tm_mes = np.arange(
        t_mes[0]-cme2_delta_mes, 
        t_mes[-1]+cme2_delta_mes, 
        step, 
        dtype=np.int
    )
    bm_mes, vtm_mes = evo.insitu(
        tm_mes, 
        fx_mes, 
        fy_mes, 
        fz_mes
    )
    bm_mes = u.T.to(u.nT, bm_mes)
    btm_mes = np.sqrt(bm_mes[:,0]**2+bm_mes[:,1]**2+bm_mes[:,2]**2)
    vtm_mes = u.Unit('m/s').to(u.Unit('km/s'), vtm_mes)
    nzi_mes = np.where(np.isfinite(btm_mes))[0]

    if nzi_mes.size > 1 and nzi_mes[0] != 0:
        cme2_tm_mes = tm_mes = tm_mes[nzi_mes]
        cme2_dm_mes = dm_mes = np.array([datetime.utcfromtimestamp(x) for x in tm_mes])
        cme2_bm_mes = bm_mes = bm_mes[nzi_mes,:]
        cme2_btm_mes = btm_mes = btm_mes[nzi_mes]
        cme2_vtm_mes = vtm_mes = vtm_mes[nzi_mes]
    else:
        return 0

    m = np.logical_or.reduce((
        np.logical_and(
            t_mes >= calendar.timegm(datetime(2011, 6, 4, 12, 29).timetuple()),
            t_mes < calendar.timegm(datetime(2011, 6, 4, 17, 14).timetuple()),
        ),
        np.logical_and(
            t_mes >= calendar.timegm(datetime(2011, 6, 5, 0, 44).timetuple()),
            t_mes < calendar.timegm(datetime(2011, 6, 5, 8, 29).timetuple())
        ),
        t_mes >= calendar.timegm(datetime(2011, 6, 5, 12, 43).timetuple())
    ))
    b_mes[np.logical_not(m),:] = np.nan
    bt_mes[np.logical_not(m)] = np.nan

    major = mdates.HourLocator(byhour=(0, 12))
    minor = mdates.HourLocator()
    majorFormat = mdates.DateFormatter('%Y-%m-%d %H:%M')

    fig = plt.figure(figsize=[8,3.33])

    gs = gridspec.GridSpec(1, 1, height_ratios=[2])

    plt.subplots_adjust(hspace=0.001)
    
    ax = fig.add_subplot(gs[0])
    
    ax.plot(d_mes, bt_mes, 'k', label='B')
    ax.plot(
        d_mes, 
        b_mes[:,0], 
        color=BLIND_PALETTE['vermillion'], 
        label='Bx'
    )
    ax.plot(
        d_mes, 
        b_mes[:,1], 
        color=BLIND_PALETTE['bluish-green'], 
        label='By'
    )
    ax.plot(
        d_mes, 
        b_mes[:,2], 
        color=BLIND_PALETTE['blue'], 
        label='Bz'
    )
    ax.axvline(
        datetime(2011, 6, 4, 15, 11),
        color='k',
        linewidth=3,
        linestyle='dashdot'
    )
    ax.axvline(
        datetime(2011, 6, 4, 16, 30),
        color=BLIND_PALETTE['vermillion'],
        linewidth=3,
        linestyle='dotted'
    )
    ax.axvline(
        cme1_dm_mes[0],
        color=BLIND_PALETTE['reddish-purple'], 
        linewidth=3, 
        linestyle='dashed',
        label='FRi3D'
    )
    # ax.plot(
    #     cme1_dm_mes, 
    #     cme1_btm_mes, 
    #     color=BLIND_PALETTE['reddish-purple'], 
    #     linewidth=4, 
    #     linestyle='dashed'
    # )
    # ax.plot(
    #     cme1_dm_mes, 
    #     cme1_bm_mes[:,0], 
    #     color=BLIND_PALETTE['reddish-purple'], 
    #     linewidth=4, 
    #     linestyle='dashed'
    # )
    # ax.plot(
    #     cme1_dm_mes, 
    #     cme1_bm_mes[:,1], 
    #     color=BLIND_PALETTE['reddish-purple'], 
    #     linewidth=4, 
    #     linestyle='dashed'
    # )
    # ax.plot(
    #     cme1_dm_mes, 
    #     cme1_bm_mes[:,2], 
    #     color=BLIND_PALETTE['reddish-purple'], 
    #     linewidth=4, 
    #     linestyle='dashed'
    # )
    ax.axvline(
        datetime(2011, 6, 5, 3, 36),
        color='k',
        linewidth=3,
        linestyle='dashdot'
    )
    ax.axvline(
        datetime(2011, 6, 5, 6, 27),
        color=BLIND_PALETTE['bluish-green'],
        linewidth=3,
        linestyle='dotted'
    )
    ax.axvline(
        cme2_dm_mes[0],
        color=BLIND_PALETTE['reddish-purple'], 
        linewidth=3, 
        linestyle='dashed'
    )
    # ax.plot(
    #     cme2_dm_mes, 
    #     cme2_btm_mes, 
    #     color=BLIND_PALETTE['reddish-purple'], 
    #     linewidth=4, 
    #     linestyle='dashed'
    # )
    # ax.plot(
    #     cme2_dm_mes, 
    #     cme2_bm_mes[:,0], 
    #     color=BLIND_PALETTE['reddish-purple'], 
    #     linewidth=4, 
    #     linestyle='dashed'
    # )
    # ax.plot(
    #     cme2_dm_mes, 
    #     cme2_bm_mes[:,1], 
    #     color=BLIND_PALETTE['reddish-purple'], 
    #     linewidth=4, 
    #     linestyle='dashed'
    # )
    # ax.plot(
    #     cme2_dm_mes, 
    #     cme2_bm_mes[:,2], 
    #     color=BLIND_PALETTE['reddish-purple'], 
    #     linewidth=4, 
    #     linestyle='dashed'
    # )
    

    ax.legend()
    ax.set_ylabel('$B$ $[nT]$')

    ax.xaxis.set_major_locator(major)
    ax.xaxis.set_major_formatter(majorFormat)
    ax.xaxis.set_minor_locator(minor)
    ax.yaxis.set_label_coords(-0.06, 0.5)
    ax.autoscale(enable=True, axis='x', tight=True)
    ax.tick_params(
        which='both', 
        direction='in',
        top=True, 
        right=True
    )

    ax.annotate(
        'CME#1',
        xy=(.236, .932), 
        xycoords='figure fraction',
        horizontalalignment='left', 
        verticalalignment='top',
        fontsize=12,
        color=BLIND_PALETTE['vermillion']
    )

    ax.annotate(
        'CME#2',
        xy=(.638, .932), 
        xycoords='figure fraction',
        horizontalalignment='left', 
        verticalalignment='top',
        fontsize=12,
        color=BLIND_PALETTE['bluish-green']
    )

    # ax = fig.add_subplot(gs[1])

    # ax.plot(
    #     cme1_dm_mes, 
    #     cme1_vtm_mes, 
    #     color=BLIND_PALETTE['reddish-purple'], 
    #     linewidth=4, 
    #     linestyle='dashed'
    # )
    # ax.plot(
    #     cme2_dm_mes, 
    #     cme2_vtm_mes, 
    #     color=BLIND_PALETTE['reddish-purple'], 
    #     linewidth=4, 
    #     linestyle='dashed'
    # )
    
    # ax.xaxis.set_major_locator(major)
    # ax.xaxis.set_major_formatter(majorFormat)
    # ax.xaxis.set_minor_locator(minor)
    # ax.yaxis.set_label_coords(-0.08, 0.5)
    # ax.autoscale(enable=True, axis='x', tight=True)
    # plt.xlim([d_mes[0], d_mes[-1]])

    plt.show()

def insitu_vex():

    d_vex, b_vex, _, p_vex = getVEX(
        cme1_d0_vex-timedelta(seconds=delta_vex), 
        cme2_d1_vex+timedelta(seconds=delta_vex*3)
    )
    t_vex = np.array([calendar.timegm(x.timetuple()) for x in d_vex])
    b_vex = u.T.to(u.nT, b_vex)
    bt_vex = np.sqrt(b_vex[:,0]**2+b_vex[:,1]**2+b_vex[:,2]**2)
    fx_vex = interp1d(
        t_vex, 
        p_vex[:,0], 
        kind='linear', 
        axis=0, 
        fill_value='extrapolate'
    )
    fy_vex = interp1d(
        t_vex, 
        p_vex[:,1], 
        kind='linear', 
        axis=0, 
        fill_value='extrapolate'
    )
    fz_vex = interp1d(
        t_vex, 
        p_vex[:,2], 
        kind='linear', 
        axis=0, 
        fill_value='extrapolate'
    )

    # CME 1

    evo = Evolution()
    p = cme1_p_vex
    evo.toroidal_height = lambda t: p[2]*(t-cme1_t0_cor)+cme1_toroidal_height_cor
    evo.sigma = lambda t: p[4]
    evo.twist = lambda t: p[5]
    evo.skew = lambda t: cme1_skew_cor
    evo.polarity = cme1_polarity
    evo.chirality = cme1_chirality
    evo.latitude = lambda t: p[7]
    evo.longitude = lambda t: p[8]
    evo.poloidal_height = lambda t: p[9]/2.0
    evo.half_width = lambda t: p[10]
    evo.tilt = lambda t: p[11]
    evo.flattening = lambda t: p[12]
    evo.pancaking = lambda t: p[13]
    evo.flux = lambda t: p[14]
    
    tm_vex = np.arange(
        t_vex[0]-cme1_delta_vex, 
        t_vex[-1]+cme1_delta_vex, 
        step, 
        dtype=np.int
    )
    bm_vex, vtm_vex = evo.insitu(
        tm_vex, 
        fx_vex, 
        fy_vex, 
        fz_vex
    )
    bm_vex = u.T.to(u.nT, bm_vex)
    btm_vex = np.sqrt(bm_vex[:,0]**2+bm_vex[:,1]**2+bm_vex[:,2]**2)
    vtm_vex = u.Unit('m/s').to(u.Unit('km/s'), vtm_vex)
    nzi_vex = np.where(np.isfinite(btm_vex))[0]

    if nzi_vex.size > 1 and nzi_vex[0] != 0:
        cme1_tm_vex = tm_vex = tm_vex[nzi_vex]
        cme1_dm_vex = dm_vex = np.array([datetime.utcfromtimestamp(x) for x in tm_vex])
        cme1_bm_vex = bm_vex = bm_vex[nzi_vex,:]
        cme1_btm_vex = btm_vex = btm_vex[nzi_vex]
        cme1_vtm_vex = vtm_vex = vtm_vex[nzi_vex]
    else:
        return 0

    # CME 2

    evo = Evolution()
    p = cme2_p_vex
    evo.toroidal_height = lambda t: (
        p[2]*(t-cme2_t0_cor)+cme2_toroidal_height_cor
        if t <= ti else
        p[2]*(ti-cme2_t0_cor)+cme2_toroidal_height_cor+p[3]*(t-ti)
    )
    evo.sigma = lambda t: p[4]
    evo.twist = lambda t: p[5]
    evo.skew = lambda t: cme2_skew_cor
    evo.polarity = cme2_polarity
    evo.chirality = cme2_chirality
    evo.latitude = lambda t: p[7]
    evo.longitude = lambda t: p[8]
    # evo.poloidal_height = lambda t: p[9]
    evo.poloidal_height = lambda t: cme2_poloidal_height_cor+p[23]*(t-cme2_t0_cor)
    evo.half_width = lambda t: p[10]
    evo.tilt = lambda t: p[11]
    evo.flattening = lambda t: p[12]
    evo.pancaking = lambda t: p[13]
    evo.flux = lambda t: p[14]
    
    tm_vex = np.arange(
        t_vex[0]-cme2_delta_vex, 
        t_vex[-1]+cme2_delta_vex, 
        step, 
        dtype=np.int
    )
    bm_vex, vtm_vex = evo.insitu(
        tm_vex, 
        fx_vex, 
        fy_vex, 
        fz_vex
    )
    bm_vex = u.T.to(u.nT, bm_vex)
    btm_vex = np.sqrt(bm_vex[:,0]**2+bm_vex[:,1]**2+bm_vex[:,2]**2)
    vtm_vex = u.Unit('m/s').to(u.Unit('km/s'), vtm_vex)
    nzi_vex = np.where(np.isfinite(btm_vex))[0]

    if nzi_vex.size > 1 and nzi_vex[0] != 0:
        cme2_tm_vex = tm_vex = tm_vex[nzi_vex]
        cme2_dm_vex = dm_vex = np.array([datetime.utcfromtimestamp(x) for x in tm_vex])
        cme2_bm_vex = bm_vex = bm_vex[nzi_vex,:]
        cme2_btm_vex = btm_vex = btm_vex[nzi_vex]
        cme2_vtm_vex = vtm_vex = vtm_vex[nzi_vex]
    else:
        return 0

    m = np.logical_or(
        t_vex < calendar.timegm(datetime(2011, 6, 6, 0, 44).timetuple()),
        t_vex > calendar.timegm(datetime(2011, 6, 6, 2, 58).timetuple())
    )
    b_vex[np.logical_not(m),:] = np.nan
    bt_vex[np.logical_not(m)] = np.nan

    major = mdates.HourLocator(byhour=(0, 12))
    minor = mdates.HourLocator()
    majorFormat = mdates.DateFormatter('%Y-%m-%d %H:%M')

    fig = plt.figure(figsize=[8,3.33*1.5])

    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])

    plt.subplots_adjust(hspace=0.001)
    
    ax = fig.add_subplot(gs[0])
    
    ax.plot(d_vex, bt_vex, 'k', label='B')
    ax.plot(
        d_vex, 
        b_vex[:,0], 
        color=BLIND_PALETTE['vermillion'], 
        label='Bx'
    )
    ax.plot(
        d_vex, 
        b_vex[:,1], 
        color=BLIND_PALETTE['bluish-green'], 
        label='By'
    )
    ax.plot(
        d_vex, 
        b_vex[:,2], 
        color=BLIND_PALETTE['blue'], 
        label='Bz'
    )

    ax.axvline(
        datetime(2011, 6, 5, 5, 25),
        color='k',
        linewidth=3,
        linestyle='dashdot'
    )
    ax.axvline(
        datetime(2011, 6, 5, 8, 39),
        color=BLIND_PALETTE['vermillion'],
        linewidth=3,
        linestyle='dotted'
    )
    ax.axvline(
        datetime(2011, 6, 5, 11, 50),
        color=BLIND_PALETTE['vermillion'],
        linewidth=3,
        linestyle='dotted'
    )
    ax.plot(
        cme1_dm_vex, 
        cme1_btm_vex, 
        # 'k', 
        color=BLIND_PALETTE['reddish-purple'], 
        linewidth=3, 
        linestyle='dashed',
        label='FRi3D'
    )
    ax.plot(
        cme1_dm_vex, 
        cme1_bm_vex[:,0], 
        # color=BLIND_PALETTE['vermillion'], 
        color=BLIND_PALETTE['reddish-purple'], 
        linewidth=3, 
        linestyle='dashed'
    )
    ax.plot(
        cme1_dm_vex, 
        cme1_bm_vex[:,1], 
        # color=BLIND_PALETTE['bluish-green'], 
        color=BLIND_PALETTE['reddish-purple'], 
        linewidth=3, 
        linestyle='dashed'
    )
    ax.plot(
        cme1_dm_vex, 
        cme1_bm_vex[:,2], 
        # color=BLIND_PALETTE['blue'], 
        color=BLIND_PALETTE['reddish-purple'], 
        linewidth=3, 
        linestyle='dashed'
    )

    ax.axvline(
        datetime(2011, 6, 5, 12, 7),
        color='k',
        linewidth=3,
        linestyle='dashdot'
    )
    ax.axvline(
        datetime(2011, 6, 5, 15, 13),
        color=BLIND_PALETTE['bluish-green'],
        linewidth=3,
        linestyle='dotted'
    )
    ax.axvline(
        datetime(2011, 6, 6, 6, 35),
        color=BLIND_PALETTE['bluish-green'],
        linewidth=3,
        linestyle='dotted'
    )
    ax.plot(
        cme2_dm_vex, 
        cme2_btm_vex, 
        color=BLIND_PALETTE['reddish-purple'], 
        linewidth=3, 
        linestyle='dashed'
    )
    ax.plot(
        cme2_dm_vex, 
        cme2_bm_vex[:,0], 
        color=BLIND_PALETTE['reddish-purple'], 
        linewidth=3, 
        linestyle='dashed'
    )
    ax.plot(
        cme2_dm_vex, 
        cme2_bm_vex[:,1], 
        color=BLIND_PALETTE['reddish-purple'], 
        linewidth=3, 
        linestyle='dashed'
    )
    ax.plot(
        cme2_dm_vex, 
        cme2_bm_vex[:,2], 
        color=BLIND_PALETTE['reddish-purple'], 
        linewidth=3, 
        linestyle='dashed'
    )

    ax.legend()
    ax.set_ylabel('$B$ $[nT]$')

    ax.xaxis.set_major_locator(major)
    ax.xaxis.set_major_formatter(majorFormat)
    ax.xaxis.set_minor_locator(minor)
    ax.yaxis.set_label_coords(-0.06, 0.5)
    ax.autoscale(enable=True, axis='x', tight=True)
    ax.tick_params(
        which='both', 
        direction='in',
        top=True, 
        right=True
    )

    ax = fig.add_subplot(gs[1])

    ax.axvline(
        datetime(2011, 6, 5, 5, 25),
        color='k',
        linewidth=3,
        linestyle='dashdot'
    )
    ax.axvline(
        datetime(2011, 6, 5, 8, 39),
        color=BLIND_PALETTE['vermillion'],
        linewidth=3,
        linestyle='dotted'
    )
    ax.axvline(
        datetime(2011, 6, 5, 11, 50),
        color=BLIND_PALETTE['vermillion'],
        linewidth=3,
        linestyle='dotted'
    )
    ax.plot(
        cme1_dm_vex, 
        cme1_vtm_vex, 
        color=BLIND_PALETTE['reddish-purple'], 
        linewidth=3, 
        linestyle='dashed'
    )

    ax.axvline(
        datetime(2011, 6, 5, 12, 7),
        color='k',
        linewidth=3,
        linestyle='dashdot'
    )
    ax.axvline(
        datetime(2011, 6, 5, 15, 13),
        color=BLIND_PALETTE['bluish-green'],
        linewidth=3,
        linestyle='dotted'
    )
    ax.axvline(
        datetime(2011, 6, 6, 6, 35),
        color=BLIND_PALETTE['bluish-green'],
        linewidth=3,
        linestyle='dotted'
    )
    ax.plot(
        cme2_dm_vex, 
        cme2_vtm_vex, 
        color=BLIND_PALETTE['reddish-purple'], 
        linewidth=3, 
        linestyle='dashed'
    )

    ax.set_ylabel('$V_p$ $[km/s]$')
    
    ax.xaxis.set_major_locator(major)
    ax.xaxis.set_major_formatter(majorFormat)
    ax.xaxis.set_minor_locator(minor)
    ax.yaxis.set_label_coords(-0.06, 0.5)
    ax.autoscale(enable=True, axis='x', tight=True)
    plt.xlim([d_vex[0], d_vex[-1]])
    ax.tick_params(
        which='both', 
        direction='in',
        top=True, 
        right=True
    )

    ax.annotate(
        'CME#1',
        xy=(.236, .914), 
        xycoords='figure fraction',
        horizontalalignment='left', 
        verticalalignment='top',
        fontsize=12,
        color=BLIND_PALETTE['vermillion']
    )

    ax.annotate(
        'CME#2',
        xy=(.496, .914), 
        xycoords='figure fraction',
        horizontalalignment='left', 
        verticalalignment='top',
        fontsize=12,
        color=BLIND_PALETTE['bluish-green']
    )

    # ax = fig.add_subplot(gs[2])

    # ax.plot(
    #     d_vex,
    #     u.rad.to(u.deg, np.arctan(b_vex[:,1]/b_vex[:,0]))
    # )
    # ax.plot(
    #     d_vex,
    #     u.rad.to(u.deg, np.arctan(b_vex[:,2]/np.sqrt(b_vex[:,0]**2+b_vex[:,1]**2)))
    # )

    # ax.xaxis.set_major_locator(major)
    # ax.xaxis.set_major_formatter(majorFormat)
    # ax.xaxis.set_minor_locator(minor)
    # ax.yaxis.set_label_coords(-0.08, 0.5)
    # ax.autoscale(enable=True, axis='x', tight=True)

    plt.show()

def insitu_sta():

    d_sta, b_sta, _, p_sta = getSTA(
        cme2_d0_sta-timedelta(seconds=delta_sta*3), 
        cme3_d1_sta+timedelta(seconds=delta_sta)
    )
    t_sta = np.array([calendar.timegm(x.timetuple()) for x in d_sta])
    b_sta = u.T.to(u.nT, b_sta)
    bt_sta = np.sqrt(b_sta[:,0]**2+b_sta[:,1]**2+b_sta[:,2]**2)
    
    cdas.set_cache(True, 'data')
    data = cdas.get_data(
        'istp_public', 
        'STA_L2_PLA_1DMAX_1MIN', 
        cme2_d0_sta-timedelta(seconds=delta_sta*3), 
        cme3_d1_sta+timedelta(seconds=delta_sta), 
        ['proton_number_density', 'proton_bulk_speed', 'proton_temperature'],
        cdf=True
    )
    data['epoch'] = np.array(data['epoch'])

    pa = table.vstack([
        ascii_.read('data/STA_L2_SWEA_PAD_20110605_V04.cef', data_start=152),
        ascii_.read('data/STA_L2_SWEA_PAD_20110606_V04.cef', data_start=129),
        ascii_.read('data/STA_L2_SWEA_PAD_20110607_V04.cef', data_start=129),
    ])
    pa_time = np.array(pa.columns[0])
    pa_time = np.array([datetime.strptime(t, "%Y-%m-%dT%H:%M:%S.%fZ") for t in pa_time])
    pa_angles = np.array([7.50, 22.50, 37.50, 52.50, 67.50, 82.50, 97.50, 112.50, 127.50, 142.50, 157.50, 172.50])
    pa_energy = table.Table(pa.columns[4:20])
    pa = np.array(table.Table(pa.columns[52:244]))
    pa = np.array([np.array(list(pa[i])) for i in range(pa.size)])
    m = np.logical_and(
        pa_time >= cme2_d0_sta-timedelta(seconds=delta_sta*3), 
        pa_time <= cme3_d1_sta+timedelta(seconds=delta_sta)
    )
    pa_time = pa_time[m]
    pa = pa[m,:]
    pa = (
        # pa[:,3::16]+
        pa[:,4::16]+
        pa[:,5::16]
        # pa[:,6::16]
    )
    pa = np.transpose(pa)

    fx_sta = interp1d(
        t_sta, 
        p_sta[:,0], 
        kind='linear', 
        axis=0, 
        fill_value='extrapolate'
    )
    fy_sta = interp1d(
        t_sta, 
        p_sta[:,1], 
        kind='linear', 
        axis=0, 
        fill_value='extrapolate'
    )
    fz_sta = interp1d(
        t_sta, 
        p_sta[:,2], 
        kind='linear', 
        axis=0, 
        fill_value='extrapolate'
    )

    # CME 2

    evo = Evolution()
    p = cme2_p_sta
    evo.toroidal_height = lambda t: (
        p[2]*(t-cme2_t0_cor)+cme2_toroidal_height_cor
        if t <= ti else
        p[2]*(ti-cme2_t0_cor)+cme2_toroidal_height_cor+p[3]*(t-ti)
    )
    evo.sigma = lambda t: p[4]
    evo.twist = lambda t: p[5]
    evo.skew = lambda t: cme2_skew_cor
    evo.polarity = cme2_polarity
    evo.chirality = cme2_chirality
    evo.latitude = lambda t: p[15]
    evo.longitude = lambda t: p[16]
    evo.poloidal_height = lambda t: p[17]
    evo.half_width = lambda t: p[18]
    evo.tilt = lambda t: p[19]
    evo.flattening = lambda t: p[20]
    evo.pancaking = lambda t: p[21]
    evo.flux = lambda t: p[22]
    
    tm_sta = np.arange(
        t_sta[0]-cme2_delta_sta, 
        t_sta[-1]+cme2_delta_sta, 
        step, 
        dtype=np.int
    )
    bm_sta, vtm_sta = evo.insitu(
        tm_sta, 
        fx_sta, 
        fy_sta, 
        fz_sta
    )
    bm_sta = u.T.to(u.nT, bm_sta)
    btm_sta = np.sqrt(bm_sta[:,0]**2+bm_sta[:,1]**2+bm_sta[:,2]**2)
    vtm_sta = u.Unit('m/s').to(u.Unit('km/s'), vtm_sta)
    nzi_sta = np.where(np.isfinite(btm_sta))[0]
    
    if nzi_sta.size > 1 and nzi_sta[0] != 0:
        cme2_tm_sta = tm_sta = tm_sta[nzi_sta]
        cme2_dm_sta = dm_sta = np.array([datetime.utcfromtimestamp(x) for x in tm_sta])
        cme2_bm_sta = bm_sta = bm_sta[nzi_sta,:]
        cme2_btm_sta = btm_sta = btm_sta[nzi_sta]
        cme2_vtm_sta = vtm_sta = vtm_sta[nzi_sta]
    else:
        return 0

    # CME 3

    evo = Evolution()
    p = cme3_p_sta
    # evo.toroidal_height = lambda t: p[2]*(t-cme3_t0_cor)+cme3_toroidal_height_cor
    cme3_di = datetime(2011, 6, 6, 15)
    cme3_ti = calendar.timegm(cme3_di.timetuple())
    evo.toroidal_height = lambda t: (
        p[2]*(t-cme3_t0_cor)+cme3_toroidal_height_cor
        if t <= cme3_ti else
        p[2]*(cme3_ti-cme3_t0_cor)+cme3_toroidal_height_cor+p[3]*(t-cme3_ti)
    )
    evo.sigma = lambda t: p[4]
    evo.twist = lambda t: p[5]
    evo.skew = lambda t: cme3_skew_cor
    evo.polarity = cme3_polarity
    evo.chirality = cme3_chirality
    evo.latitude = lambda t: p[15]
    evo.longitude = lambda t: p[16]
    evo.poloidal_height = lambda t: p[17]
    evo.half_width = lambda t: p[18]
    evo.tilt = lambda t: p[19]
    evo.flattening = lambda t: p[20]
    evo.pancaking = lambda t: p[21]
    evo.flux = lambda t: p[22]
    
    tm_sta = np.arange(
        t_sta[0]-cme3_delta_sta, 
        t_sta[-1]+cme3_delta_sta, 
        step, 
        dtype=np.int
    )
    bm_sta, vtm_sta = evo.insitu(
        tm_sta, 
        fx_sta, 
        fy_sta, 
        fz_sta
    )
    bm_sta = u.T.to(u.nT, bm_sta)
    btm_sta = np.sqrt(bm_sta[:,0]**2+bm_sta[:,1]**2+bm_sta[:,2]**2)
    vtm_sta = u.Unit('m/s').to(u.Unit('km/s'), vtm_sta)
    nzi_sta = np.where(np.isfinite(btm_sta))[0]

    if nzi_sta.size > 1 and nzi_sta[0] != 0:
        cme3_tm_sta = tm_sta = tm_sta[nzi_sta]
        cme3_dm_sta = dm_sta = np.array([datetime.utcfromtimestamp(x) for x in tm_sta])
        cme3_bm_sta = bm_sta = bm_sta[nzi_sta,:]
        cme3_btm_sta = btm_sta = btm_sta[nzi_sta]
        cme3_vtm_sta = vtm_sta = vtm_sta[nzi_sta]
    else:
        return 0

    major = mdates.HourLocator(byhour=(0, 12))
    minor = mdates.HourLocator()
    majorFormat = mdates.DateFormatter('%Y-%m-%d %H:%M')

    fig = plt.figure(figsize=[8,10])

    gs = gridspec.GridSpec(5, 1, height_ratios=[2, 1, 1, 1, 1])

    plt.subplots_adjust(hspace=0.001)
    
    ax = fig.add_subplot(gs[0])
    
    ax.plot(d_sta, bt_sta, 'k', label='B')
    ax.plot(
        d_sta, 
        b_sta[:,0], 
        color=BLIND_PALETTE['vermillion'], 
        label='Bx'
    )
    ax.plot(
        d_sta, 
        b_sta[:,1], 
        color=BLIND_PALETTE['bluish-green'], 
        label='By'
    )
    ax.plot(
        d_sta, 
        b_sta[:,2], 
        color=BLIND_PALETTE['blue'], 
        label='Bz'
    )

    ax.axvline(
        datetime(2011, 6, 5, 18, 58),
        color='k',
        linewidth=3,
        linestyle='dashdot'
    )
    ax.axvline(
        datetime(2011, 6, 6, 12, 23),
        color=BLIND_PALETTE['bluish-green'],
        linewidth=3,
        linestyle='dotted'
    )
    ax.axvline(
        datetime(2011, 6, 6, 14, 15),
        color=BLIND_PALETTE['bluish-green'],
        linewidth=3,
        linestyle='dotted'
    )
    ax.plot(
        cme2_dm_sta, 
        cme2_btm_sta, 
        color=BLIND_PALETTE['reddish-purple'], 
        linewidth=3, 
        linestyle='dashed',
        label='FRi3D'
    )
    ax.plot(
        cme2_dm_sta, 
        cme2_bm_sta[:,0], 
        color=BLIND_PALETTE['reddish-purple'], 
        linewidth=3, 
        linestyle='dashed'
    )
    ax.plot(
        cme2_dm_sta, 
        cme2_bm_sta[:,1], 
        color=BLIND_PALETTE['reddish-purple'], 
        linewidth=3, 
        linestyle='dashed'
    )
    ax.plot(
        cme2_dm_sta, 
        cme2_bm_sta[:,2], 
        color=BLIND_PALETTE['reddish-purple'], 
        linewidth=3, 
        linestyle='dashed'
    )

    ax.axvline(
        datetime(2011, 6, 6, 17, 7),
        color=BLIND_PALETTE['blue'],
        linewidth=3,
        linestyle='dotted'
    )
    ax.axvline(
        datetime(2011, 6, 7, 1, 35),
        color=BLIND_PALETTE['blue'],
        linewidth=3,
        linestyle='dotted'
    )
    ax.plot(
        cme3_dm_sta, 
        cme3_btm_sta, 
        color=BLIND_PALETTE['reddish-purple'], 
        linewidth=3, 
        linestyle='dashed'
    )
    ax.plot(
        cme3_dm_sta, 
        cme3_bm_sta[:,0], 
        color=BLIND_PALETTE['reddish-purple'], 
        linewidth=3, 
        linestyle='dashed'
    )
    ax.plot(
        cme3_dm_sta, 
        cme3_bm_sta[:,1], 
        color=BLIND_PALETTE['reddish-purple'], 
        linewidth=3, 
        linestyle='dashed'
    )
    ax.plot(
        cme3_dm_sta, 
        cme3_bm_sta[:,2], 
        color=BLIND_PALETTE['reddish-purple'], 
        linewidth=3, 
        linestyle='dashed'
    )

    ax.legend()
    ax.set_ylabel('$B$ $[nT]$')

    ax.xaxis.set_major_locator(major)
    ax.xaxis.set_major_formatter(majorFormat)
    ax.xaxis.set_minor_locator(minor)
    ax.yaxis.set_label_coords(-0.06, 0.5)
    ax.autoscale(enable=True, axis='x', tight=True)
    ax.tick_params(
        which='both', 
        direction='in',
        top=True, 
        right=True
    )

    ax = fig.add_subplot(gs[1])
    im = ax.imshow(
        pa, 
        aspect='auto', 
        cmap='RdBu_r', 
        norm=LogNorm(), 
        extent=[
            mdates.date2num(cme2_d0_sta-timedelta(seconds=delta_sta*3)),
            mdates.date2num(cme3_d1_sta+timedelta(seconds=delta_sta)),
            0.0, 180.0
        ],
        interpolation='bilinear'
    )
    ax.xaxis_date()
    plt.setp(ax.get_xticklabels(), visible=False)
    Bbox = transforms.Bbox.from_bounds(1.02, 0, 0.03, 1)
    trans = ax.transAxes+fig.transFigure.inverted()
    l, b, w, h = transforms.TransformedBbox(Bbox, trans).bounds
    cax = fig.add_axes([l*0.99, b, w, h])
    cb = plt.colorbar(im, cax = cax)
    # cb.locator = plt.MaxNLocator(5)
    cb.update_ticks()
    cb.ax.set_ylabel('PSD $[s^3km^{-6}]$', rotation=270, labelpad=14)
    # cb.ax.set_title('PSD $[s^3km^{-6}]$')
    axc = ax
    axc.yaxis.set_label_coords(-0.06, 0.5)
    ax.set_ylabel('$PA$ $[^\circ]$')
    ax.axvline(
        datetime(2011, 6, 5, 18, 58),
        color='k',
        linewidth=3,
        linestyle='dashdot'
    )
    ax.axvline(
        datetime(2011, 6, 6, 12, 23),
        color=BLIND_PALETTE['bluish-green'],
        linewidth=3,
        linestyle='dotted'
    )
    ax.axvline(
        datetime(2011, 6, 6, 14, 15),
        color=BLIND_PALETTE['bluish-green'],
        linewidth=3,
        linestyle='dotted'
    )
    ax.axvline(
        datetime(2011, 6, 6, 17, 7),
        color=BLIND_PALETTE['blue'],
        linewidth=3,
        linestyle='dotted'
    )
    ax.axvline(
        datetime(2011, 6, 7, 1, 35),
        color=BLIND_PALETTE['blue'],
        linewidth=3,
        linestyle='dotted'
    )
    
    ax.xaxis.set_major_locator(major)
    ax.xaxis.set_major_formatter(majorFormat)
    ax.xaxis.set_minor_locator(minor)
    ax.yaxis.set_label_coords(-0.06, 0.5)
    ax.autoscale(enable=True, axis='x', tight=True)
    ax.tick_params(
        which='both', 
        direction='in',
        top=True, 
        right=True
    )

    ax = fig.add_subplot(gs[2])
    m = data['proton_bulk_speed'] >= 0.0
    data['proton_bulk_speed'][np.logical_not(m)] = np.nan
    ax.plot(data['epoch'], data['proton_bulk_speed'], 'k')
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.set_ylabel('$V_p$ $[km/s]$')
    ax.axvline(
        datetime(2011, 6, 5, 18, 58),
        color='k',
        linewidth=3,
        linestyle='dashdot'
    )
    ax.axvline(
        datetime(2011, 6, 6, 12, 23),
        color=BLIND_PALETTE['bluish-green'],
        linewidth=3,
        linestyle='dotted'
    )
    ax.axvline(
        datetime(2011, 6, 6, 14, 15),
        color=BLIND_PALETTE['bluish-green'],
        linewidth=3,
        linestyle='dotted'
    )
    ax.axvline(
        datetime(2011, 6, 6, 17, 7),
        color=BLIND_PALETTE['blue'],
        linewidth=3,
        linestyle='dotted'
    )
    ax.axvline(
        datetime(2011, 6, 7, 1, 35),
        color=BLIND_PALETTE['blue'],
        linewidth=3,
        linestyle='dotted'
    )

    ax.plot(
        cme2_dm_sta, 
        cme2_vtm_sta, 
        color=BLIND_PALETTE['reddish-purple'], 
        linewidth=3, 
        linestyle='dashed',
        label='FRi3D'
    )
    ax.plot(
        cme3_dm_sta, 
        cme3_vtm_sta, 
        color=BLIND_PALETTE['reddish-purple'], 
        linewidth=3, 
        linestyle='dashed'
    )

    ax.legend()
    ax.xaxis.set_major_locator(major)
    ax.xaxis.set_major_formatter(majorFormat)
    ax.xaxis.set_minor_locator(minor)
    ax.yaxis.set_label_coords(-0.06, 0.5)
    ax.autoscale(enable=True, axis='x', tight=True)
    plt.xlim([d_sta[0], d_sta[-1]])
    ax.tick_params(
        which='both', 
        direction='in',
        top=True, 
        right=True
    )
    
    ax = fig.add_subplot(gs[3])
    m = data['proton_number_density'] >= 0.0
    data['proton_number_density'][np.logical_not(m)] = np.nan
    ax.plot(data['epoch'], data['proton_number_density'], 'k')
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.set_ylabel('$N_p$ $[cm^{-3}]$')
    ax.axvline(
        datetime(2011, 6, 5, 18, 58),
        color='k',
        linewidth=3,
        linestyle='dashdot'
    )
    ax.axvline(
        datetime(2011, 6, 6, 12, 23),
        color=BLIND_PALETTE['bluish-green'],
        linewidth=3,
        linestyle='dotted'
    )
    ax.axvline(
        datetime(2011, 6, 6, 14, 15),
        color=BLIND_PALETTE['bluish-green'],
        linewidth=3,
        linestyle='dotted'
    )
    ax.axvline(
        datetime(2011, 6, 6, 17, 7),
        color=BLIND_PALETTE['blue'],
        linewidth=3,
        linestyle='dotted'
    )
    ax.axvline(
        datetime(2011, 6, 7, 1, 35),
        color=BLIND_PALETTE['blue'],
        linewidth=3,
        linestyle='dotted'
    )

    ax.xaxis.set_major_locator(major)
    ax.xaxis.set_major_formatter(majorFormat)
    ax.xaxis.set_minor_locator(minor)
    ax.yaxis.set_label_coords(-0.06, 0.5)
    ax.autoscale(enable=True, axis='x', tight=True)
    plt.xlim([d_sta[0], d_sta[-1]])
    ax.tick_params(
        which='both', 
        direction='in',
        top=True, 
        right=True
    )
    
    ax = fig.add_subplot(gs[4])
    m = data['proton_temperature'] >= 0.0
    data['proton_temperature'][np.logical_not(m)] = np.nan
    ax.plot(data['epoch'], data['proton_temperature']/1e6, 'k')
    ax.set_ylabel('$T_p$ $[MK]$')
    ax.axvline(
        datetime(2011, 6, 5, 18, 58),
        color='k',
        linewidth=3,
        linestyle='dashdot'
    )
    ax.axvline(
        datetime(2011, 6, 6, 12, 23),
        color=BLIND_PALETTE['bluish-green'],
        linewidth=3,
        linestyle='dotted'
    )
    ax.axvline(
        datetime(2011, 6, 6, 14, 15),
        color=BLIND_PALETTE['bluish-green'],
        linewidth=3,
        linestyle='dotted'
    )
    ax.axvline(
        datetime(2011, 6, 6, 17, 7),
        color=BLIND_PALETTE['blue'],
        linewidth=3,
        linestyle='dotted'
    )
    ax.axvline(
        datetime(2011, 6, 7, 1, 35),
        color=BLIND_PALETTE['blue'],
        linewidth=3,
        linestyle='dotted'
    )

    ax.xaxis.set_major_locator(major)
    ax.xaxis.set_major_formatter(majorFormat)
    ax.xaxis.set_minor_locator(minor)
    ax.yaxis.set_label_coords(-0.06, 0.5)
    ax.autoscale(enable=True, axis='x', tight=True)
    plt.xlim([d_sta[0], d_sta[-1]])
    ax.tick_params(
        which='both', 
        direction='in',
        top=True, 
        right=True
    )

    ax.annotate(
        'CME#2',
        xy=(.486, .897), 
        xycoords='figure fraction',
        horizontalalignment='left', 
        verticalalignment='top',
        fontsize=12,
        color=BLIND_PALETTE['bluish-green']
    )

    ax.annotate(
        'CME#3',
        xy=(.658, .897), 
        xycoords='figure fraction',
        horizontalalignment='left', 
        verticalalignment='top',
        fontsize=12,
        color=BLIND_PALETTE['blue']
    )

    plt.show()

insitu_mes()

insitu_vex()

insitu_sta()
