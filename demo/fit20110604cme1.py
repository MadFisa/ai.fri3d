
# from ai.fri3d.optimize import fit2remote, fit2insitu
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
import calendar
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def fit2insitu():

    step = 600

    # COR & initial
    latitude0 = u.deg.to(u.rad, 34.0)
    longitude0 = u.deg.to(u.rad, 130.0)
    toroidal_height0 = u.R_sun.to(u.m, 12.0)
    poloidal_height0 = u.R_sun.to(u.m, 3.0)
    half_width0 = u.deg.to(u.rad, 44.0)
    tilt0 = u.deg.to(u.rad, -35.0)
    flattening0 = 0.3
    pancaking0 = u.deg.to(u.rad, 18.0)
    skew = 0.0
    twist = np.array([0.1, 2.0])
    flux = np.array([1e13, 1e15])
    polarity = 1.0
    chirality = 1.0
    d0 = datetime(2011, 6, 4, 8, 54)
    t0 = calendar.timegm(d0.timetuple())

    spline_s_phi_kind = 'cubic',
    spline_s_phi_n = 500

    # MESSENGER
    d0_mes = datetime(2011, 6, 4, 17, 9)
    t0_mes = calendar.timegm(d0_mes.timetuple())
    bt_mes = u.nT.to(u.T, 80.0)
    delta_mes = 2*3600

    # VEX
    d0_vex = datetime(2011, 6, 5, 8, 45)
    d1_vex = datetime(2011, 6, 5, 11, 50)
    t0_vex = calendar.timegm(d0_vex.timetuple())
    t1_vex = calendar.timegm(d1_vex.timetuple())
    d_vex, b_vex, _, p_vex = getVEX(d0_vex, d1_vex)
    bx_vex = b_vex[:,0]
    by_vex = b_vex[:,1]
    bz_vex = b_vex[:,2]
    bt_vex = np.sqrt(bx_vex**+by_vex**2+bz_vex**2)
    delta_vex = 2*3600

    # STA
    d0_sta = datetime(2011, 6, 6, 12, 25)
    d1_sta = datetime(2011, 6, 6, 14, 10)
    t0_sta = calendar.timegm(d0_sta.timetuple())
    t1_sta = calendar.timegm(d1_sta.timetuple())
    t_sta, b_sta, v_sta, p_sta = getSTA(d0_sta, d1_sta)
    bx_sta = b_sta[:,0]
    by_sta = b_sta[:,1]
    bz_sta = b_sta[:,2]
    bt_sta = np.sqrt(bx_sta**2+by_sta**2+bz_sta**2)
    delta_sta = 2*3600

    di = datetime(2011, 6, 5, 11, 30)
    ti = calendar.timegm(di.timetuple())

    def F(p):
        evo = Evolution()

        # p[0] - latitude final
        # p[1] - latitude decay
        evo.latitude = lambda t: (latitude0-p[0])*np.exp(-p[1]*(t-t0))+p[0]

        evo.longitude = lambda t: longitude0

        # p[2] - speed before interaction
        # p[3] - speed after interaction
        evo.toroidal_height = lambda t: p[2]*t+toroidal_height0 if t <= ti else p[3]*t+p[2]*ti+toroidal_height0

        # p[4] - poloidal height final
        # p[5] - poloidal height decay
        evo.poloidal_height = lambda t: (poloidal_height0-p[4])*np.exp(-p[5]*(t-t0))+p[4]

        evo.half_width = lambda t: half_width0

        # p[6] - tilt speed
        evo.tilt = lambda t: p[6]*t+tilt0

        evo.flattening = lambda t: flattening0

        evo.pancaking = lambda t: pancaking0

        evo.skew = lambda t: skew

        # p[7] - twist
        evo.twist = lambda t: p[7]

        # p[8] - flux
        evo.flux = lambda t: p[8]

        evo.polarity = polarity
        evo.chirality = chirality
        evo.spline_s_phi_kind = spline_s_phi_kind
        evo.spline_s_phi_n = spline_s_phi_n

        tm_mes = np.arange(
            t0_mes-delta_mes, 
            t0_mes+delta_mes, 
            step, 
            dtype=np.int
        )
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
        bm_mes, vm_mes = evo.insitu(
            tm_mes, 
            fx_mes(tm_mes), 
            fy_mes(tm_mes), 
            fz_mes(tm_mes)
        )
        btm_mes = np.sqrt(bm_mes[:,0]**2+bm_mes[:,1]**2+bm_mes[:,2]**2)
        nzi_mes = np.where(np.isfinite(btm_mes))[0]
        pre_delta_mes = np.nan
        bt_delta_mes = np.nan
        if nzi_mes.size > 0 and nzi_mes[0] != 0:
            pre_delta_mes = np.abs(tm_mes[nzi_mes[0]]-t0_mes)
            bt_delta_mes = np.abs(np.mean(btm_mes)-bt_mes)
        else:
            return np.nan

        t_vex_m = np.arange(
            t0_vex-delta_vex, 
            t1_vex+delta_vex, 
            step, 
            dtype=np.int
        )
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
        bm_vex, vm_vex = evo.insitu(
            tm_vex, 
            fx_vex(tm_vex), 
            fy_vex(tm_vex), 
            fz_vex(tm_vex)
        )
        btm_vex = np.sqrt(bm_vex[:,0]**2+bm_vex[:,1]**2+bm_vex[:,2]**2)
        nzi_vex = np.where(np.isfinite(btm_vex))[0]
        pre_delta_vex = post_delta_vex = np.nan
        b_delta_vex = np.nan
        if (nzi_vex.size > 0 and 
                nzi_vex[0] != 0 and 
                nzi_vex[-1] != tm_vex.size-1):
            pre_delta_vex = np.abs(tm_vex[nzi_vex[0]]-t0_vex)
            post_delta_vex = np.abs(tm_vex[nzi_vex[-1]]-t1_vex)
            b_delta_vex = fastdtw(
                bm_vex[nzi_vex,:], 
                b_vex, 
                dist=euclidean
            )
        else:
            return np.nan

        t_sta_m = np.arange(
            t0_sta-delta_sta, 
            t1_sta+delta_sta, 
            step, 
            dtype=np.int
        )
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
        bm_sta, vm_sta = evo.insitu(
            tm_sta, 
            fx_sta(tm_sta), 
            fy_sta(tm_sta), 
            fz_sta(tm_sta)
        )
        btm_sta = np.sqrt(bm_sta[:,0]**2+bm_sta[:,1]**2+bm_sta[:,2]**2)
        nzi_sta = np.where(np.isfinite(btm_sta))[0]
        pre_delta_sta = post_delta_sta = np.nan
        b_delta_sta = np.nan
        if (nzi_sta.size > 0 and 
                nzi_sta[0] != 0 and 
                nzi_sta[-1] != tm_sta.size-1):
            pre_delta_sta = np.abs(tm_sta[nzi_sta[0]]-t0_sta)
            post_delta_sta = np.abs(tm_sta[nzi_sta[-1]]-t1_sta)
            b_delta_sta = fastdtw(
                bm_sta[nzi_sta,:], 
                b_sta, 
                dist=euclidean
            )
        else:
            return np.nan

        print(
            pre_delta_mes, bt_delta_mes,
            pre_delta_vex, post_delta_vex, b_delta_vex,
            pre_delta_sta, post_delta_sta, b_delta_sta
        )

        # return 

    bounds = [
        u.deg.to(u.rad, (-5.0, 10.0)),
        (),
        (700.0, 1400.0),
        (700.0, 1400.0),
        u.au.to(u.m, (0.05, 0.25)),
        (),
        (),
        (0.5, 2.5),
        (1e13, 1e15),
    ]

    res = differential_evolution(F, bounds=bounds)

    print(res.x)

    return res

def fitshell():
    theta0 = u.deg.to(u.rad, 34.0)
    phi0 = u.deg.to(u.rad, 130.0)
    Rt0 = u.R_sun.to(u.m, 12.0)
    Rp0 = u.R_sun.to(u.m, 3.0)
    thetaHW0 = u.deg.to(u.rad, 44.0)
    gamma0 = u.deg.to(u.rad, -35.0)
    n0 = 0.3
    thetaP0 = u.deg.to(u.rad, 18.0)

    polarity = 1.0
    chirality = 1.0
    spline_s_phi_kind='cubic',
    spline_s_phi_n=500

    t0 = datetime(2011, 6, 4, 8, 54)

    theta = lambda t: (theta0-theta1)*np.exp(-atheta*(t-t0))+theta1
    gamma = lambda t: agamma*t+gamma0
    Rp = lambda t: (Rp0-Rp1)*np.exp(-aRp*(t-t0))+Rp1
    Rt = lambda t: aRt*t+Rt0 if t <= ti else bRt*t+aRt*ti+Rt0

    # To fit
    # theta1
    # atheta
    # agamma
    # Rp1
    # aRp
    # aRt
    # bRt

    # flux
    # twist
    # sigma

    step = 600

    t1 = datetime(2011, 6, 7, 18)

    t = np.arange(
        calendar.timegm(t0.timetuple()),
        calendar.timegm(t1.timetuple()),
        step
    )

    fr = FRi3D()
    fr.polarity = polarity
    fr.chirality = chirality
    fr.spline_s_phi_kind = spline_s_phi_kind
    fr.spline_s_phi_n = spline_s_phi_n
    b = []
    v = []

    for i, t in enumerate(t):
        fr.latitude = self.latitude(t)
        fr.longitude = self.longitude(t)
        fr.toroidal_height = self.toroidal_height(t)
        fr.poloidal_height = self.poloidal_height(t)
        fr.half_width = self.half_width(t)
        fr.tilt = self.tilt(t)
        fr.flattening = self.flattening(t)
        fr.pancaking = self.pancaking(t)
        fr.skew = self.skew(t)
        fr.twist = self.twist(t)
        fr.flux = self.flux(t)
        fr.sigma = self.sigma(t)
        if i == 0:
            fr.toroidal_height = 1.0
            fr.init()
            fr._unit_spline_initial_axis_s_phi = \
                fr._spline_initial_axis_s_phi
            fr.toroidal_height = self.toroidal_height(t)
            fr.init()
        # valid if flattening, half width and flux stay constant
        fr._spline_initial_axis_s_phi = lambda s: \
            fr._unit_spline_initial_axis_s_phi(s/fr.toroidal_height)
        b_, c_ = fr.data(x, y, z)
        if b_.size == 0:
            b_ = np.array([0.0, 0.0, 0.0])
        if c_.size == 0:
            c_ = np.array([0.0, 0.0])
        b.append(b_.ravel())
        v.append(
            c_[0]*(self.toroidal_height(t)-self.toroidal_height(t-1))+
            c_[1]*(self.poloidal_height(t)-self.poloidal_height(t-1))
        )