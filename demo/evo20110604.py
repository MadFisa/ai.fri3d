
import numpy as np
from scipy.interpolate import interp1d

from matplotlib import pyplot as plt
from matplotlib import dates as mdates
from matplotlib import gridspec
from matplotlib.colors import LogNorm
from matplotlib import transforms
from mpl_toolkits.mplot3d import proj3d

import calendar
from datetime import datetime, timedelta

from astropy import units as u
from astropy.io import ascii as ascii_
from astropy import table

from ai.fri3d import FRi3D, Evolution
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

def toroidal_height():
    cme1_toroidal_height_vex = lambda t: (
        (cme1_toroidal_height_speed1_vex-cme1_toroidal_height_speed2_vex)/
        cme1_toroidal_height_decay_vex*
        (1.0-np.exp(-cme1_toroidal_height_decay_vex*(t-cme1_t0)))+
        cme1_toroidal_height_speed2_vex*(t-cme1_t0)+cme1_toroidal_height_cor
    )

    cme2_toroidal_height_vex = lambda t: (
        (cme2_toroidal_height_speed1_vex-cme2_toroidal_height_speed2_vex)/
        cme2_toroidal_height_decay_vex*
        (1.0-np.exp(-cme2_toroidal_height_decay_vex*(t-cme2_t0)))+
        cme2_toroidal_height_speed2_vex*(t-cme2_t0)+cme2_toroidal_height_cor
    )

    cme1_toroidal_height_sta = lambda t: (
        (cme1_toroidal_height_speed1_sta-cme1_toroidal_height_speed2_sta)/
        cme1_toroidal_height_decay_sta*
        (1.0-np.exp(-cme1_toroidal_height_decay_sta*(t-cme1_t0)))+
        cme1_toroidal_height_speed2_sta*(t-cme1_t0)+cme1_toroidal_height_cor
    )

    cme2_toroidal_height_sta = lambda t: (
        (cme2_toroidal_height_speed1_sta-cme2_toroidal_height_speed2_sta)/
        cme2_toroidal_height_decay_sta*
        (1.0-np.exp(-cme2_toroidal_height_decay_sta*(t-cme2_t0)))+
        cme2_toroidal_height_speed2_sta*(t-cme2_t0)+cme2_toroidal_height_cor
    )

    major = mdates.HourLocator(byhour=(0, 12))
    minor = mdates.HourLocator()
    majorFormat = mdates.DateFormatter('%Y-%m-%d %H:%M')

    fig = plt.figure(figsize=[8,8])

    gs = gridspec.GridSpec(1, 1, height_ratios=[2])

    plt.subplots_adjust(hspace=0.001)
    ax = fig.add_subplot(gs[0])
    
    ax.plot(
        cme1_d0, 
        cme1_toroidal_height_cor, 
        '+', 
        color=BLIND_PALETTE['vermillion']
    )
    ax.plot(
        [cme1_d0_vex, cme1_d1_vex], 
        [
            cme1_toroidal_height_vex(cme1_t0_vex),
            cme1_toroidal_height_vex(cme1_t1_vex)
        ],
        '+',
        color=BLIND_PALETTE['vermillion']
    )
    ax.plot(
        [cme1_d0_sta, cme1_d1_sta], 
        [
            cme1_toroidal_height_sta(cme1_t0_sta),
            cme1_toroidal_height_sta(cme1_t1_sta)
        ],
        '+',
        color=BLIND_PALETTE['vermillion']
    )

    ax.plot(
        cme2_d0, 
        cme2_toroidal_height_cor, 
        '+',
        color=BLIND_PALETTE['bluish-green']
    )
    ax.plot(
        [cme2_d0_vex, cme2_d1_vex], 
        [
            cme2_toroidal_height_vex(cme2_t0_vex),
            cme2_toroidal_height_vex(cme2_t1_vex)
        ],
        '+',
        color=BLIND_PALETTE['bluish-green']
    )
    ax.plot(
        [cme2_d0_sta, cme2_d1_sta], 
        [
            cme2_toroidal_height_sta(cme2_t0_sta),
            cme2_toroidal_height_sta(cme2_t1_sta)
        ],
        '+',
        color=BLIND_PALETTE['bluish-green']
    )

    ax.legend()
    # ax.set_ylabel('$B$ $[nT]$')

    ax.xaxis.set_major_locator(major)
    ax.xaxis.set_major_formatter(majorFormat)
    ax.xaxis.set_minor_locator(minor)
    ax.yaxis.set_label_coords(-0.08, 0.5)
    ax.autoscale(enable=True, axis='x', tight=True)

    plt.show()

def snapshot_mes():
    cme1_toroidal_height_vex = lambda t: (
        (cme1_toroidal_height_speed1_vex-cme1_toroidal_height_speed2_vex)/
        cme1_toroidal_height_decay_vex*
        (1.0-np.exp(-cme1_toroidal_height_decay_vex*(t-cme1_t0)))+
        cme1_toroidal_height_speed2_vex*(t-cme1_t0)+cme1_toroidal_height_cor
    )

    cme2_toroidal_height_vex = lambda t: (
        (cme2_toroidal_height_speed1_vex-cme2_toroidal_height_speed2_vex)/
        cme2_toroidal_height_decay_vex*
        (1.0-np.exp(-cme2_toroidal_height_decay_vex*(t-cme2_t0)))+
        cme2_toroidal_height_speed2_vex*(t-cme2_t0)+cme2_toroidal_height_cor
    )

    cme1_toroidal_height_sta = lambda t: (
        (cme1_toroidal_height_speed1_sta-cme1_toroidal_height_speed2_sta)/
        cme1_toroidal_height_decay_sta*
        (1.0-np.exp(-cme1_toroidal_height_decay_sta*(t-cme1_t0)))+
        cme1_toroidal_height_speed2_sta*(t-cme1_t0)+cme1_toroidal_height_cor
    )

    cme2_toroidal_height_sta = lambda t: (
        (cme2_toroidal_height_speed1_sta-cme2_toroidal_height_speed2_sta)/
        cme2_toroidal_height_decay_sta*
        (1.0-np.exp(-cme2_toroidal_height_decay_sta*(t-cme2_t0)))+
        cme2_toroidal_height_speed2_sta*(t-cme2_t0)+cme2_toroidal_height_cor
    )

    t = cme2_t0_mes

    fr1 = FRi3D(
        latitude=cme1_latitude_cor,
        longitude=cme1_longitude_vex,
        toroidal_height=cme1_toroidal_height_vex(t),
        poloidal_height=(cme1_poloidal_height_vex+cme1_poloidal_height_cor)/2.0,
        half_width=cme1_half_width_cor,
        tilt=cme1_tilt_cor,
        flattening=cme1_flattening_cor,
        pancaking=cme1_pancaking_cor,
        skew=cme1_skew_cor
    )
    
    fr2 = FRi3D(
        latitude=cme2_latitude_cor,
        longitude=cme2_longitude_vex,
        toroidal_height=cme2_toroidal_height_vex(t),
        poloidal_height=(cme2_poloidal_height_vex+cme2_poloidal_height_cor)/2.0,
        half_width=cme2_half_width_cor,
        tilt=cme2_tilt_cor,
        flattening=cme2_flattening_cor,
        pancaking=cme2_pancaking_cor,
        skew=cme2_skew_cor
    )
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d', adjustable='box', aspect=1.0)
    ax.set_xlabel('X [AU]')
    ax.set_zlabel('Z [AU]')
    ax.set_ylabel('Y [AU]')
    ax.set_xlim(-1.5, 0.5)
    ax.set_ylim(0.0, 2.0)
    ax.set_zlim(-1.0, 1.0)
    ax.view_init(90.0, 0.0)

    x, y, z = fr1.shell()
    x *= u.m.to(u.au)
    y *= u.m.to(u.au)
    z *= u.m.to(u.au)
    ax.plot_wireframe(x, y, z, color=BLIND_PALETTE['blue'], alpha=0.4, label='CME 1')
    
    x, y, z = fr2.shell()
    x *= u.m.to(u.au)
    y *= u.m.to(u.au)
    z *= u.m.to(u.au)
    ax.plot_wireframe(x, y, z, color=BLIND_PALETTE['vermillion'], alpha=0.4, label='CME 2')
    
    plt.show()

def snapshot_vex():
    cme1_toroidal_height_vex = lambda t: (
        (cme1_toroidal_height_speed1_vex-cme1_toroidal_height_speed2_vex)/
        cme1_toroidal_height_decay_vex*
        (1.0-np.exp(-cme1_toroidal_height_decay_vex*(t-cme1_t0)))+
        cme1_toroidal_height_speed2_vex*(t-cme1_t0)+cme1_toroidal_height_cor
    )

    cme2_toroidal_height_vex = lambda t: (
        (cme2_toroidal_height_speed1_vex-cme2_toroidal_height_speed2_vex)/
        cme2_toroidal_height_decay_vex*
        (1.0-np.exp(-cme2_toroidal_height_decay_vex*(t-cme2_t0)))+
        cme2_toroidal_height_speed2_vex*(t-cme2_t0)+cme2_toroidal_height_cor
    )

    cme1_toroidal_height_sta = lambda t: (
        (cme1_toroidal_height_speed1_sta-cme1_toroidal_height_speed2_sta)/
        cme1_toroidal_height_decay_sta*
        (1.0-np.exp(-cme1_toroidal_height_decay_sta*(t-cme1_t0)))+
        cme1_toroidal_height_speed2_sta*(t-cme1_t0)+cme1_toroidal_height_cor
    )

    cme2_toroidal_height_sta = lambda t: (
        (cme2_toroidal_height_speed1_sta-cme2_toroidal_height_speed2_sta)/
        cme2_toroidal_height_decay_sta*
        (1.0-np.exp(-cme2_toroidal_height_decay_sta*(t-cme2_t0)))+
        cme2_toroidal_height_speed2_sta*(t-cme2_t0)+cme2_toroidal_height_cor
    )

    t = cme1_t0_sta

    fr1 = FRi3D(
        latitude=cme1_latitude_vex,
        longitude=cme1_longitude_vex,
        toroidal_height=cme1_toroidal_height_vex(t),
        poloidal_height=cme1_poloidal_height_vex,
        half_width=cme1_half_width_cor,
        tilt=cme1_tilt_vex,
        flattening=cme1_flattening_cor,
        pancaking=cme1_pancaking_cor,
        skew=cme1_skew_cor
    )
    
    fr2 = FRi3D(
        latitude=cme2_latitude_vex,
        longitude=cme2_longitude_vex,
        toroidal_height=cme2_toroidal_height_vex(t),
        poloidal_height=cme2_poloidal_height_vex,
        half_width=cme2_half_width_cor,
        tilt=cme2_tilt_vex,
        flattening=cme2_flattening_cor,
        pancaking=cme2_pancaking_cor,
        skew=cme2_skew_cor
    )
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d', adjustable='box', aspect=1.0)
    ax.set_xlabel('X [AU]')
    ax.set_zlabel('Z [AU]')
    ax.set_ylabel('Y [AU]')
    ax.set_xlim(-1.5, 0.5)
    ax.set_ylim(0.0, 2.0)
    ax.set_zlim(-1.0, 1.0)
    ax.view_init(90.0, 0.0)

    x, y, z = fr1.shell()
    x *= u.m.to(u.au)
    y *= u.m.to(u.au)
    z *= u.m.to(u.au)
    ax.plot_wireframe(x, y, z, color=BLIND_PALETTE['blue'], alpha=0.4)
    
    x, y, z = fr2.shell()
    x *= u.m.to(u.au)
    y *= u.m.to(u.au)
    z *= u.m.to(u.au)
    ax.plot_wireframe(x, y, z, color=BLIND_PALETTE['vermillion'], alpha=0.4)
    
    plt.show()

def snapshot_sta():
    cme1_toroidal_height_vex = lambda t: (
        (cme1_toroidal_height_speed1_vex-cme1_toroidal_height_speed2_vex)/
        cme1_toroidal_height_decay_vex*
        (1.0-np.exp(-cme1_toroidal_height_decay_vex*(t-cme1_t0)))+
        cme1_toroidal_height_speed2_vex*(t-cme1_t0)+cme1_toroidal_height_cor
    )

    cme2_toroidal_height_vex = lambda t: (
        (cme2_toroidal_height_speed1_vex-cme2_toroidal_height_speed2_vex)/
        cme2_toroidal_height_decay_vex*
        (1.0-np.exp(-cme2_toroidal_height_decay_vex*(t-cme2_t0)))+
        cme2_toroidal_height_speed2_vex*(t-cme2_t0)+cme2_toroidal_height_cor
    )

    cme1_toroidal_height_sta = lambda t: (
        (cme1_toroidal_height_speed1_sta-cme1_toroidal_height_speed2_sta)/
        cme1_toroidal_height_decay_sta*
        (1.0-np.exp(-cme1_toroidal_height_decay_sta*(t-cme1_t0)))+
        cme1_toroidal_height_speed2_sta*(t-cme1_t0)+cme1_toroidal_height_cor
    )

    cme2_toroidal_height_sta = lambda t: (
        (cme2_toroidal_height_speed1_sta-cme2_toroidal_height_speed2_sta)/
        cme2_toroidal_height_decay_sta*
        (1.0-np.exp(-cme2_toroidal_height_decay_sta*(t-cme2_t0)))+
        cme2_toroidal_height_speed2_sta*(t-cme2_t0)+cme2_toroidal_height_cor
    )

    t = cme1_t0_sta

    fr1 = FRi3D(
        latitude=cme1_latitude_sta,
        longitude=cme1_longitude_sta,
        toroidal_height=cme1_toroidal_height_vex(t),
        poloidal_height=cme1_poloidal_height_sta,
        half_width=cme1_half_width_cor,
        tilt=cme1_tilt_sta,
        flattening=cme1_flattening_sta,
        pancaking=cme1_pancaking_cor,
        skew=cme1_skew_cor,
        polarity=-1.0,
        chirality=1.0
    )
    
    fr2 = FRi3D(
        latitude=cme2_latitude_sta,
        longitude=cme2_longitude_sta,
        toroidal_height=cme2_toroidal_height_sta(t),
        poloidal_height=cme2_poloidal_height_sta,
        half_width=cme2_half_width_cor,
        tilt=cme2_tilt_sta,
        flattening=cme2_flattening_cor,
        pancaking=cme2_pancaking_cor,
        skew=cme2_skew_cor,
        polarity=1.0,
        chirality=1.0
    )
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d', adjustable='box', aspect=1.0)
    ax.set_xlabel('X [AU]')
    ax.set_zlabel('Z [AU]')
    ax.set_ylabel('Y [AU]')
    ax.set_xlim(-1.5, 0.5)
    ax.set_ylim(0.0, 2.0)
    ax.set_zlim(-1.0, 1.0)
    ax.view_init(90.0, 0.0)

    x, y, z = fr1.shell()
    x *= u.m.to(u.au)
    y *= u.m.to(u.au)
    z *= u.m.to(u.au)
    ax.plot_wireframe(x, y, z, color=BLIND_PALETTE['blue'], alpha=0.4)
    
    x, y, z = fr2.shell()
    x *= u.m.to(u.au)
    y *= u.m.to(u.au)
    z *= u.m.to(u.au)
    ax.plot_wireframe(x, y, z, color=BLIND_PALETTE['vermillion'], alpha=0.4)
    
    plt.show()

# toroidal_height()

snapshot_mes()

snapshot_vex()

snapshot_sta()
