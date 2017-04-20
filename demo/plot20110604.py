
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
cme1_d0 = datetime(2011, 6, 4, 8, 54)
cme1_t0 = calendar.timegm(cme1_d0.timetuple())
# MESSENGER
cme1_d0_mes = datetime(2011, 6, 4, 17, 9)
cme1_d1_mes = datetime(2011, 6, 4, 17, 10)
cme1_t0_mes = calendar.timegm(cme1_d0_mes.timetuple())
cme1_t1_mes = calendar.timegm(cme1_d1_mes.timetuple())
# VEX
cme1_d0_vex = datetime(2011, 6, 5, 8, 45)
cme1_d1_vex = datetime(2011, 6, 5, 11, 50)
cme1_t0_vex = calendar.timegm(cme1_d0_vex.timetuple())
cme1_t1_vex = calendar.timegm(cme1_d1_vex.timetuple())
cme1_delta_vex = 6*3600
cme1_latitude_vex = 3.71535281e-01
cme1_longitude_vex = 2.09158975e+00
cme1_toroidal_height_decay_vex = 4.98432646e-04
cme1_toroidal_height_speed1_vex = 1.74057191e+06
cme1_toroidal_height_speed2_vex = 1.08406323e+06
cme1_poloidal_height_vex = 1.26309213e+10
cme1_tilt_vex = 6.92166768e-01
cme1_twist_vex = 1.08690976e-01
cme1_flux_vex = 4.86949810e+14
#STA
cme1_d0_sta = datetime(2011, 6, 6, 12, 25)
cme1_d1_sta = datetime(2011, 6, 6, 14, 10)
cme1_t0_sta = calendar.timegm(cme1_d0_sta.timetuple())
cme1_t1_sta = calendar.timegm(cme1_d1_sta.timetuple())
cme1_delta_sta = 6*3600
cme1_latitude_sta = 4.18129374e-01
cme1_longitude_sta = 2.13883564e+00
cme1_toroidal_height_decay_sta = 2.46584756e-04
cme1_toroidal_height_speed1_sta = 1.30702623e+06
cme1_toroidal_height_speed2_sta = 9.04242047e+05
cme1_poloidal_height_sta = 3.69006508e+09
cme1_tilt_sta = 1.29144281e+00
cme1_flattening_sta = 4.99389039e-01
cme1_twist_sta = 1.25584618e-01
cme1_flux_sta = 1.40522930e+14

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
cme2_d0 = datetime(2011, 6, 4, 22, 54)
cme2_t0 = calendar.timegm(cme2_d0.timetuple())
# MESSENGER
cme2_d0_mes = datetime(2011, 6, 5, 4, 40)
cme2_d1_mes = datetime(2011, 6, 5, 9, 29)
cme2_t0_mes = calendar.timegm(cme2_d0_mes.timetuple())
cme2_t1_mes = calendar.timegm(cme2_d1_mes.timetuple())
# VEX
cme2_d0_vex = datetime(2011, 6, 5, 15, 30)
cme2_d1_vex = datetime(2011, 6, 5, 22, 30)
cme2_t0_vex = calendar.timegm(cme2_d0_vex.timetuple())
cme2_t1_vex = calendar.timegm(cme2_d1_vex.timetuple())
cme2_delta_vex = 6*3600
cme2_latitude_vex = 2.89279612e-01
cme2_longitude_vex = 2.19422308e+00
cme2_toroidal_height_decay_vex = 6.75817828e-05
cme2_toroidal_height_speed1_vex = 2.77230795e+06
cme2_toroidal_height_speed2_vex = 1.03521318e+06
cme2_poloidal_height_vex = 1.47186256e+10
cme2_tilt_vex = 6.13536756e-01
cme2_twist_vex = 1.22860516e+00
cme2_flux_vex = 5.50755944e+14
#STA
cme2_d0_sta = datetime(2011, 6, 6, 16, 30)
cme2_d1_sta = datetime(2011, 6, 7, 1)
cme2_t0_sta = calendar.timegm(cme2_d0_sta.timetuple())
cme2_t1_sta = calendar.timegm(cme2_d1_sta.timetuple())
cme2_delta_sta = 6*3600
cme2_latitude_sta = 3.02279517e-01
cme2_longitude_sta = 2.17905229e+00
cme2_toroidal_height_decay_sta = 4.89516179e-04
cme2_toroidal_height_speed1_sta = 1.92546287e+06
cme2_toroidal_height_speed2_sta = 1.43053865e+06
cme2_poloidal_height_sta = 5.65102795e+09
cme2_tilt_sta = 6.35232725e-01
cme2_twist_sta = 6.43469704e-01
cme2_flux_sta = 5.07487712e+13

def insitu_mes():
    d_mes, b_mes, _, p_mes = getMES(
        cme1_d0_mes-timedelta(seconds=delta_mes), 
        cme2_d1_mes+timedelta(seconds=delta_mes)
    )
    # d_mes += timedelta(hours=2)
    b_mes = u.T.to(u.nT, b_mes)
    t_mes = np.array([calendar.timegm(x.timetuple()) for x in d_mes])
    bt_mes = np.sqrt(b_mes[:,0]**2+b_mes[:,1]**2+b_mes[:,2]**2)

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

    ax.legend()
    ax.set_ylabel('$B$ $[nT]$')

    ax.xaxis.set_major_locator(major)
    ax.xaxis.set_major_formatter(majorFormat)
    ax.xaxis.set_minor_locator(minor)
    ax.yaxis.set_label_coords(-0.08, 0.5)
    ax.autoscale(enable=True, axis='x', tight=True)

    plt.show()

def insitu_vex():

    cme1_evo = Evolution()
    cme1_evo.latitude = lambda t: cme1_latitude_vex
    cme1_evo.longitude = lambda t: cme1_longitude_vex
    cme1_evo.toroidal_height = lambda t: (
        (cme1_toroidal_height_speed1_vex-cme1_toroidal_height_speed2_vex)/
        cme1_toroidal_height_decay_vex*
        (1.0-np.exp(-cme1_toroidal_height_decay_vex*(t-cme1_t0)))+
        cme1_toroidal_height_speed2_vex*(t-cme1_t0)+cme1_toroidal_height_cor
    )
    cme1_evo.poloidal_height = lambda t: cme1_poloidal_height_vex
    cme1_evo.half_width = lambda t: cme1_half_width_cor
    cme1_evo.tilt = lambda t: cme1_tilt_vex
    cme1_evo.flattening = lambda t: cme1_flattening_cor
    cme1_evo.pancaking = lambda t: cme1_pancaking_cor
    cme1_evo.skew = lambda t: cme1_skew_cor
    cme1_evo.twist = lambda t: cme1_twist_vex
    cme1_evo.flux = lambda t: cme1_flux_vex
    cme1_evo.sigma = lambda t: 2.0
    cme1_evo.polarity = cme1_polarity
    cme1_evo.chirality = cme1_chirality

    cme2_evo = Evolution()
    cme2_evo.latitude = lambda t: cme2_latitude_vex
    cme2_evo.longitude = lambda t: cme2_longitude_vex
    cme2_evo.toroidal_height = lambda t: (
        (cme2_toroidal_height_speed1_vex-cme2_toroidal_height_speed2_vex)/
        cme2_toroidal_height_decay_vex*
        (1.0-np.exp(-cme2_toroidal_height_decay_vex*(t-cme2_t0)))+
        cme2_toroidal_height_speed2_vex*(t-cme2_t0)+cme2_toroidal_height_cor
    )
    cme2_evo.poloidal_height = lambda t: cme2_poloidal_height_vex
    cme2_evo.half_width = lambda t: cme2_half_width_cor
    cme2_evo.tilt = lambda t: cme2_tilt_vex
    cme2_evo.flattening = lambda t: cme2_flattening_cor
    cme2_evo.pancaking = lambda t: cme2_pancaking_cor
    cme2_evo.skew = lambda t: cme2_skew_cor
    cme2_evo.twist = lambda t: cme2_twist_vex
    cme2_evo.flux = lambda t: cme2_flux_vex
    cme2_evo.sigma = lambda t: 2.0
    cme2_evo.polarity = cme2_polarity
    cme2_evo.chirality = cme2_chirality


    d_vex, b_vex, _, p_vex = getVEX(
        cme1_d0_vex-timedelta(seconds=delta_vex), 
        cme2_d1_vex+timedelta(seconds=delta_vex)
    )
    b_vex = u.T.to(u.nT, b_vex)
    t_vex = np.array([calendar.timegm(x.timetuple()) for x in d_vex])
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

    cme1_tm_vex = np.arange(
        cme1_t0_vex-cme1_delta_vex, 
        cme1_t1_vex+cme1_delta_vex, 
        step, 
        dtype=np.int
    )
    cme1_bm_vex, _ = cme1_evo.insitu(
        cme1_tm_vex, 
        fx_vex, 
        fy_vex, 
        fz_vex
    )
    cme1_bm_vex = u.T.to(u.nT, cme1_bm_vex)
    cme1_btm_vex = np.sqrt(cme1_bm_vex[:,0]**2+cme1_bm_vex[:,1]**2+cme1_bm_vex[:,2]**2)
    cme1_dm_vex = np.array([datetime.utcfromtimestamp(x) for x in cme1_tm_vex])

    cme2_tm_vex = np.arange(
        cme2_t0_vex-cme2_delta_vex, 
        cme2_t1_vex+cme2_delta_vex, 
        step, 
        dtype=np.int
    )
    cme2_bm_vex, _ = cme2_evo.insitu(
        cme2_tm_vex, 
        fx_vex, 
        fy_vex, 
        fz_vex
    )
    cme2_bm_vex = u.T.to(u.nT, cme2_bm_vex)
    cme2_btm_vex = np.sqrt(cme2_bm_vex[:,0]**2+cme2_bm_vex[:,1]**2+cme2_bm_vex[:,2]**2)
    cme2_dm_vex = np.array([datetime.utcfromtimestamp(x) for x in cme2_tm_vex])

    major = mdates.HourLocator(byhour=(0, 12))
    minor = mdates.HourLocator()
    majorFormat = mdates.DateFormatter('%Y-%m-%d %H:%M')

    fig = plt.figure(figsize=[8,3.33])

    gs = gridspec.GridSpec(1, 1, height_ratios=[2])

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

    ax.plot(
        cme1_dm_vex, 
        cme1_btm_vex, 
        color=BLIND_PALETTE['reddish-purple'], 
        linewidth=4, 
        linestyle='dashed'
    )
    ax.plot(
        cme1_dm_vex, 
        cme1_bm_vex[:,0], 
        color=BLIND_PALETTE['reddish-purple'], 
        linewidth=4, 
        linestyle='dashed'
    )
    ax.plot(
        cme1_dm_vex, 
        cme1_bm_vex[:,1], 
        color=BLIND_PALETTE['reddish-purple'], 
        linewidth=4, 
        linestyle='dashed'
    )
    ax.plot(
        cme1_dm_vex, 
        cme1_bm_vex[:,2], 
        color=BLIND_PALETTE['reddish-purple'], 
        linewidth=4, 
        linestyle='dashed'
    )

    ax.plot(
        cme2_dm_vex, 
        cme2_btm_vex, 
        color=BLIND_PALETTE['reddish-purple'], 
        linewidth=4, 
        linestyle='dashed'
    )
    ax.plot(
        cme2_dm_vex, 
        cme2_bm_vex[:,0], 
        color=BLIND_PALETTE['reddish-purple'], 
        linewidth=4, 
        linestyle='dashed'
    )
    ax.plot(
        cme2_dm_vex, 
        cme2_bm_vex[:,1], 
        color=BLIND_PALETTE['reddish-purple'], 
        linewidth=4, 
        linestyle='dashed'
    )
    ax.plot(
        cme2_dm_vex, 
        cme2_bm_vex[:,2], 
        color=BLIND_PALETTE['reddish-purple'], 
        linewidth=4, 
        linestyle='dashed', 
        label='FRi3D'
    )

    ax.legend()
    ax.set_ylabel('$B$ $[nT]$')

    ax.xaxis.set_major_locator(major)
    ax.xaxis.set_major_formatter(majorFormat)
    ax.xaxis.set_minor_locator(minor)
    ax.yaxis.set_label_coords(-0.08, 0.5)
    ax.autoscale(enable=True, axis='x', tight=True)

    plt.show()
    
def insitu_sta():

    cme1_evo = Evolution()
    cme1_evo.latitude = lambda t: cme1_latitude_sta
    cme1_evo.longitude = lambda t: cme1_longitude_sta
    cme1_evo.toroidal_height = lambda t: (
        (cme1_toroidal_height_speed1_sta-cme1_toroidal_height_speed2_sta)/
        cme1_toroidal_height_decay_sta*
        (1.0-np.exp(-cme1_toroidal_height_decay_sta*(t-cme1_t0)))+
        cme1_toroidal_height_speed2_sta*(t-cme1_t0)+cme1_toroidal_height_cor
    )
    cme1_evo.poloidal_height = lambda t: cme1_poloidal_height_sta
    cme1_evo.half_width = lambda t: cme1_half_width_cor
    cme1_evo.tilt = lambda t: cme1_tilt_sta
    cme1_evo.flattening = lambda t: cme1_flattening_sta
    cme1_evo.pancaking = lambda t: cme1_pancaking_cor
    cme1_evo.skew = lambda t: cme1_skew_cor
    cme1_evo.twist = lambda t: cme1_twist_sta
    cme1_evo.flux = lambda t: cme1_flux_sta
    cme1_evo.sigma = lambda t: 2.0
    cme1_evo.polarity = cme1_polarity
    cme1_evo.chirality = cme1_chirality

    cme2_evo = Evolution()
    cme2_evo.latitude = lambda t: cme2_latitude_sta
    cme2_evo.longitude = lambda t: cme2_longitude_sta
    cme2_evo.toroidal_height = lambda t: (
        (cme2_toroidal_height_speed1_sta-cme2_toroidal_height_speed2_sta)/
        cme2_toroidal_height_decay_sta*
        (1.0-np.exp(-cme2_toroidal_height_decay_sta*(t-cme2_t0)))+
        cme2_toroidal_height_speed2_sta*(t-cme2_t0)+cme2_toroidal_height_cor
    )
    cme2_evo.poloidal_height = lambda t: cme2_poloidal_height_sta
    cme2_evo.half_width = lambda t: cme2_half_width_cor
    cme2_evo.tilt = lambda t: cme2_tilt_sta
    cme2_evo.flattening = lambda t: cme2_flattening_cor
    cme2_evo.pancaking = lambda t: cme2_pancaking_cor
    cme2_evo.skew = lambda t: cme2_skew_cor
    cme2_evo.twist = lambda t: cme2_twist_sta
    cme2_evo.flux = lambda t: cme2_flux_sta
    cme2_evo.sigma = lambda t: 2.0
    cme2_evo.polarity = cme2_polarity
    cme2_evo.chirality = cme2_chirality


    d_sta, b_sta, _, p_sta = getSTA(
        cme1_d0_sta-timedelta(seconds=delta_sta), 
        cme2_d1_sta+timedelta(seconds=delta_sta)
    )
    b_sta = u.T.to(u.nT, b_sta)
    t_sta = np.array([calendar.timegm(x.timetuple()) for x in d_sta])
    bt_sta = np.sqrt(b_sta[:,0]**2+b_sta[:,1]**2+b_sta[:,2]**2)

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

    cme1_tm_sta = np.arange(
        cme1_t0_sta-cme1_delta_sta, 
        cme1_t1_sta+cme1_delta_sta, 
        step, 
        dtype=np.int
    )
    cme1_bm_sta, cme1_vtm_sta = cme1_evo.insitu(
        cme1_tm_sta, 
        fx_sta, 
        fy_sta, 
        fz_sta
    )
    cme1_bm_sta = u.T.to(u.nT, cme1_bm_sta)
    cme1_vtm_sta = u.Unit('m/s').to(u.Unit('km/s'), cme1_vtm_sta)
    cme1_btm_sta = np.sqrt(cme1_bm_sta[:,0]**2+cme1_bm_sta[:,1]**2+cme1_bm_sta[:,2]**2)
    cme1_dm_sta = np.array([datetime.utcfromtimestamp(x) for x in cme1_tm_sta])

    cme2_tm_sta = np.arange(
        cme2_t0_sta-cme2_delta_sta, 
        cme2_t1_sta+cme2_delta_sta, 
        step, 
        dtype=np.int
    )
    cme2_bm_sta, cme2_vtm_sta = cme2_evo.insitu(
        cme2_tm_sta, 
        fx_sta, 
        fy_sta, 
        fz_sta
    )
    cme2_bm_sta = u.T.to(u.nT, cme2_bm_sta)
    cme2_vtm_sta = u.Unit('m/s').to(u.Unit('km/s'), cme2_vtm_sta)
    cme2_btm_sta = np.sqrt(cme2_bm_sta[:,0]**2+cme2_bm_sta[:,1]**2+cme2_bm_sta[:,2]**2)
    cme2_dm_sta = np.array([datetime.utcfromtimestamp(x) for x in cme2_tm_sta])


    cdas.set_cache(True, 'data')
    data = cdas.get_data(
        'istp_public', 
        'STA_L2_PLA_1DMAX_1MIN', 
        cme1_d0_sta-timedelta(seconds=delta_sta), 
        cme2_d1_sta+timedelta(seconds=delta_sta), 
        ['proton_number_density', 'proton_bulk_speed', 'proton_temperature'],
        cdf=True
    )
    data['epoch'] = np.array(data['epoch'])

    pa = table.vstack([
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
        pa_time >= cme1_d0_sta-timedelta(seconds=delta_sta), 
        pa_time <= cme2_d1_sta+timedelta(seconds=delta_sta)
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

    ax.plot(
        cme1_dm_sta, 
        cme1_btm_sta, 
        color=BLIND_PALETTE['reddish-purple'], 
        linewidth=4, 
        linestyle='dashed'
    )
    ax.plot(
        cme1_dm_sta, 
        cme1_bm_sta[:,0], 
        color=BLIND_PALETTE['reddish-purple'], 
        linewidth=4, 
        linestyle='dashed'
    )
    ax.plot(
        cme1_dm_sta, 
        cme1_bm_sta[:,1], 
        color=BLIND_PALETTE['reddish-purple'], 
        linewidth=4, 
        linestyle='dashed'
    )
    ax.plot(
        cme1_dm_sta, 
        cme1_bm_sta[:,2], 
        color=BLIND_PALETTE['reddish-purple'], 
        linewidth=4, 
        linestyle='dashed'
    )

    ax.plot(
        cme2_dm_sta, 
        cme2_btm_sta, 
        color=BLIND_PALETTE['reddish-purple'], 
        linewidth=4, 
        linestyle='dashed'
    )
    ax.plot(
        cme2_dm_sta, 
        cme2_bm_sta[:,0], 
        color=BLIND_PALETTE['reddish-purple'], 
        linewidth=4, 
        linestyle='dashed'
    )
    ax.plot(
        cme2_dm_sta, 
        cme2_bm_sta[:,1], 
        color=BLIND_PALETTE['reddish-purple'], 
        linewidth=4, 
        linestyle='dashed'
    )
    ax.plot(
        cme2_dm_sta, 
        cme2_bm_sta[:,2], 
        color=BLIND_PALETTE['reddish-purple'], 
        linewidth=4, 
        linestyle='dashed', 
        label='FRi3D'
    )
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.legend()
    ax.set_ylabel('$B$ $[nT]$')

    ax.xaxis.set_major_locator(major)
    ax.xaxis.set_major_formatter(majorFormat)
    ax.xaxis.set_minor_locator(minor)
    ax.yaxis.set_label_coords(-0.08, 0.5)
    ax.autoscale(enable=True, axis='x', tight=True)

    ax = fig.add_subplot(gs[1])
    im = ax.imshow(
        pa, 
        aspect='auto', 
        cmap='RdBu_r', 
        norm=LogNorm(), 
        extent=[
            mdates.date2num(cme1_d0_sta-timedelta(seconds=delta_sta)),
            mdates.date2num(cme2_d1_sta+timedelta(seconds=delta_sta)),
            0.0, 180.0
        ],
        interpolation='bilinear'
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
    ax.autoscale(enable=True, axis='x', tight=True)

    ax = fig.add_subplot(gs[2])
    m = data['proton_bulk_speed'] >= 0.0
    ax.plot(data['epoch'][m], data['proton_bulk_speed'][m], 'k')
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.set_ylabel('$V_p$ $[km/s]$')

    # ax.plot(
    #     cme1_dm_sta, 
    #     cme1_vtm_sta, 
    #     color=BLIND_PALETTE['reddish-purple'], 
    #     linewidth=4, 
    #     linestyle='dashed'
    # )
    ax.plot(
        cme2_dm_sta, 
        cme2_vtm_sta, 
        color=BLIND_PALETTE['reddish-purple'], 
        linewidth=4, 
        linestyle='dashed'
    )

    ax.xaxis.set_major_locator(major)
    ax.xaxis.set_major_formatter(majorFormat)
    ax.xaxis.set_minor_locator(minor)
    ax.yaxis.set_label_coords(-0.08, 0.5)
    ax.autoscale(enable=True, axis='x', tight=True)
    
    ax = fig.add_subplot(gs[3])
    m = data['proton_number_density'] >= 0.0
    ax.plot(data['epoch'][m], data['proton_number_density'][m], 'k')
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.set_ylabel('$N_p$ $[cm^{-3}]$')

    ax.xaxis.set_major_locator(major)
    ax.xaxis.set_major_formatter(majorFormat)
    ax.xaxis.set_minor_locator(minor)
    ax.yaxis.set_label_coords(-0.08, 0.5)
    ax.autoscale(enable=True, axis='x', tight=True)
    
    ax = fig.add_subplot(gs[4])
    m = data['proton_temperature'] >= 0.0
    ax.plot(data['epoch'][m], data['proton_temperature'][m]/1e6, 'k')
    ax.set_ylabel('$T_p$ $[MK]$')

    ax.xaxis.set_major_locator(major)
    ax.xaxis.set_major_formatter(majorFormat)
    ax.xaxis.set_minor_locator(minor)
    ax.yaxis.set_label_coords(-0.08, 0.5)
    ax.autoscale(enable=True, axis='x', tight=True)

    plt.show()

insitu_mes()

# insitu_vex()

# insitu_sta()
