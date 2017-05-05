
# from ai.fri3d.optimize import fit2remote, fit2insitu
from ai.fri3d import Evolution
from astropy import units as u
from astropy import constants as c
from datetime import datetime, timedelta
import numpy as np
from ai.shared.data import getMES, getVEX, getSTA
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
from scipy.optimize import differential_evolution, fmin_l_bfgs_b, basinhopping, minimize
import time
import calendar
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn import preprocessing
from scipy.signal import savgol_filter

from functools import wraps
import errno
import os
import signal

from skopt import gp_minimize

res_prev = np.inf
num_eval = 0

class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)

def fit2insitu():

    step = 600

    # COR & initial
    latitude_cor = u.deg.to(u.rad, 30.0)
    longitude_cor = u.deg.to(u.rad, 110.0)
    toroidal_height_cor = u.R_sun.to(u.m, 12.5)
    poloidal_height_cor = u.R_sun.to(u.m, 3.5)
    half_width_cor = u.deg.to(u.rad, 40.0)
    tilt_cor = u.deg.to(u.rad, 37.0)
    flattening_cor = 0.4
    pancaking_cor = u.deg.to(u.rad, 25.0)
    skew = 0.0
    polarity = -1.0
    chirality = 1.0
    d0 = datetime(2011, 6, 4, 8, 54)
    t0 = calendar.timegm(d0.timetuple())

    spline_s_phi_kind = 'cubic',
    spline_s_phi_n = 500

    # MESSENGER
    d0_mes = datetime(2011, 6, 4, 17, 9)
    d1_mes = datetime(2011, 6, 4, 17, 10)
    t0_mes = calendar.timegm(d0_mes.timetuple())
    t1_mes = calendar.timegm(d1_mes.timetuple())
    d_mes, b_mes, _, p_mes = getMES(d0_mes, d1_mes)
    t_mes = np.array([calendar.timegm(x.timetuple()) for x in d_mes])
    bt_mes = np.mean(np.sqrt(b_mes[:,0]**2+b_mes[:,1]**2+b_mes[:,2]**2))
    ta_mes = t0_mes+3600
    pa_mes = np.mean(p_mes, axis=0)
    delta_mes = 40*3600

    # VEX
    d0_vex = datetime(2011, 6, 5, 8, 45)
    d1_vex = datetime(2011, 6, 5, 11, 50)
    t0_vex = calendar.timegm(d0_vex.timetuple())
    t1_vex = calendar.timegm(d1_vex.timetuple())
    d_vex, b_vex, _, p_vex = getVEX(d0_vex, d1_vex)
    b_vex = savgol_filter(b_vex, int(d_vex.size/2.0//2*2+1), 3, axis=0)
    t_vex = np.array([calendar.timegm(x.timetuple()) for x in d_vex])
    p = np.polynomial.polynomial.polyfit(t_vex, b_vex, 1)
    b_vex = np.polynomial.polynomial.polyval(t_vex, p).T
    bt_vex = np.sqrt(b_vex[:,0]**2+b_vex[:,1]**2+b_vex[:,2]**2)
    ta_vex = np.mean(t_vex)
    pa_vex = np.mean(p_vex, axis=0)
    delta_vex = 40*3600

    # STA
    d0_sta = datetime(2011, 6, 6, 12, 25)
    d1_sta = datetime(2011, 6, 6, 14, 10)
    t0_sta = calendar.timegm(d0_sta.timetuple())
    t1_sta = calendar.timegm(d1_sta.timetuple())
    d_sta, b_sta, _, p_sta = getSTA(d0_sta, d1_sta)
    b_sta = savgol_filter(b_sta, int(d_sta.size/2.0//2*2+1), 3, axis=0)
    t_sta = np.array([calendar.timegm(x.timetuple()) for x in d_sta])
    p = np.polynomial.polynomial.polyfit(t_sta, b_sta, 1)
    b_sta = np.polynomial.polynomial.polyval(t_sta, p).T
    bt_sta = np.sqrt(b_sta[:,0]**2+b_sta[:,1]**2+b_sta[:,2]**2)
    ta_sta = np.mean(t_sta)
    pa_sta = np.mean(p_sta, axis=0)
    
    cdas.set_cache(True, './data')
    data = cdas.get_data(
        'sp_phys', 
        'STA_L2_PLA_1DMAX_1MIN', 
        d0_sta,
        d1_sta,
        ['proton_bulk_speed'],
        cdf=True
    )
    mask = data['proton_bulk_speed'] > 0.0
    f = interp1d(
        np.array([calendar.timegm(x.timetuple()) for x in data['epoch'][mask]]), 
        data['proton_bulk_speed'][mask], 
        kind='linear',
        fill_value='extrapolate'
    )
    v_sta = u.Unit('km/s').to(u.Unit('m/s'), f(t_sta))
    v_sta = savgol_filter(v_sta, int(v_sta.size/2.0//2*2+1), 3)
    p = np.polyfit(t_sta, v_sta, 1)
    v_sta = np.polyval(p, t_sta)

    delta_sta = 40*3600

    di = datetime(2011, 6, 5, 11, 30)
    ti = calendar.timegm(di.timetuple())

    scaler = preprocessing.MinMaxScaler()
    
    def constraint_cross(p):
        p = scaler.inverse_transform(np.array([p]))[0].tolist()
        p[0] = np.exp(p[0])
        evo = Evolution()
        if p[1] < p[2]:
            print('CONSTRAINT FAILED')
            return -1
        evo.toroidal_height = lambda t: (
            (p[1]-p[2])/p[0]*(1.0-np.exp(-p[0]*(t-t0)))+p[2]*(t-t0)+
            toroidal_height_cor
            if t <= ti else
            (p[1]-p[2])/p[0]*(1.0-np.exp(-p[0]*(ti-t0)))+p[2]*(ti-t0)+
            toroidal_height_cor+
            p[3]*(t-ti)
        )
        evo.sigma = lambda t: p[4]
        evo.twist = lambda t: p[5]
        evo.skew = lambda t: skew
        evo.polarity = polarity
        evo.chirality = chirality
        evo.latitude = lambda t: p[7]
        evo.longitude = lambda t: p[8]
        evo.poloidal_height = lambda t: p[9]
        evo.half_width = lambda t: p[10]
        evo.tilt = lambda t: p[11]
        evo.flattening = lambda t: p[12]
        evo.pancaking = lambda t: p[13]
        evo.flux = lambda t: p[6]
        bm_mes, _ = evo.insitu([ta_mes], pa_mes[0], pa_mes[1], pa_mes[2])
        btm_mes = np.sqrt(bm_mes[:,0]**2+bm_mes[:,1]**2+bm_mes[:,2]**2)
        nzi_mes = np.where(np.isfinite(btm_mes))[0]
        
        if nzi_mes.size == 0:
            constraint_mes = -1
        else:
            constraint_mes = 1

        evo.flux = lambda t: p[14]
        bm_vex, _ = evo.insitu([ta_vex], pa_vex[0], pa_vex[1], pa_vex[2])
        btm_vex = np.sqrt(bm_vex[:,0]**2+bm_vex[:,1]**2+bm_vex[:,2]**2)
        nzi_vex = np.where(np.isfinite(btm_vex))[0]
        if nzi_vex.size == 0:
            constraint_vex = -1
        else:
            constraint_vex = 1

        evo.latitude = lambda t: p[15]
        evo.longitude = lambda t: p[16]
        evo.poloidal_height = lambda t: p[17]
        evo.half_width = lambda t: p[18]
        evo.tilt = lambda t: p[19]
        evo.flattening = lambda t: p[20]
        evo.pancaking = lambda t: p[21]
        evo.flux = lambda t: p[22]
        bm_sta, _ = evo.insitu([ta_sta], pa_sta[0], pa_sta[1], pa_sta[2])
        btm_sta = np.sqrt(bm_sta[:,0]**2+bm_sta[:,1]**2+bm_sta[:,2]**2)
        nzi_sta = np.where(np.isfinite(btm_sta))[0]
        if nzi_sta.size == 0:
            constraint_sta = -1
        else:
            constraint_sta = 1

        constraint = constraint_mes+constraint_vex+constraint_sta-2
        print('CONSTRAINT = ', constraint)
        return constraint

    # scales = np.array([
    #     1e4,
    #     1e-1/u.Unit('km/s').to(u.Unit('m/s'), 1.0),
    #     1e-1/u.Unit('km/s').to(u.Unit('m/s'), 1.0),
    #     1e-1/u.Unit('km/s').to(u.Unit('m/s'), 1.0),
    #     1e1,
    #     1e1,
    #     1e-13,
    #     1.0/u.deg.to(u.rad, 1.0),
    #     1.0/u.deg.to(u.rad, 1.0),
    #     1e3/u.au.to(u.m, 1.0),
    #     1.0/u.deg.to(u.rad, 1.0),
    #     1.0/u.deg.to(u.rad, 1.0),
    #     1e1,
    #     1.0/u.deg.to(u.rad, 1.0),
    #     1e-13,
    #     1.0/u.deg.to(u.rad, 1.0),
    #     1.0/u.deg.to(u.rad, 1.0),
    #     1e3/u.au.to(u.m, 1.0),
    #     1.0/u.deg.to(u.rad, 1.0),
    #     1.0/u.deg.to(u.rad, 1.0),
    #     1e1,
    #     1.0/u.deg.to(u.rad, 1.0),
    #     1e-13
    # ])

    scales = np.array([
        1e4,
        1e-1/u.Unit('km/s').to(u.Unit('m/s'), 1.0)/5.0,
        1e-1/u.Unit('km/s').to(u.Unit('m/s'), 1.0)/5.0,
        1e-1/u.Unit('km/s').to(u.Unit('m/s'), 1.0)/5.0,
        1e1,
        1e1,
        1e-13,
        1.0/u.deg.to(u.rad, 1.0)/5.0,
        1.0/u.deg.to(u.rad, 1.0)/5.0,
        1e3/u.au.to(u.m, 1.0)/5.0,
        1.0/u.deg.to(u.rad, 1.0)/5.0,
        1.0/u.deg.to(u.rad, 1.0)/5.0,
        1e1,
        1.0/u.deg.to(u.rad, 1.0)/5.0,
        1e-13,
        1.0/u.deg.to(u.rad, 1.0)/5.0,
        1.0/u.deg.to(u.rad, 1.0)/5.0,
        1e3/u.au.to(u.m, 1.0)/5.0,
        1.0/u.deg.to(u.rad, 1.0)/5.0,
        1.0/u.deg.to(u.rad, 1.0)/5.0,
        1e1,
        1.0/u.deg.to(u.rad, 1.0)/5.0,
        1e-13
    ])

    weights = np.array([
        1.5, 1.0,
        2.0, 1.5, 1.0,
        2.0, 1.5, 1.0, 1.0,
    ])

    def F(p):
        global res_prev
        global num_eval
        num_eval += 1
        # print('NUMBER OF EVALUATIONS = ', num_eval)
        if num_eval%100 == 0:
            print('NUMBER OF EVALUATIONS = ', num_eval)
        p = p/scales
        # p = scaler.inverse_transform(np.array([p]))[0].tolist()
        # p[0] = np.exp(p[0])
        """
        SHARED
        0, 1, 2, 3: toroidal_height
        4: sigma
        5: twist
        MES
        6: flux
        MES & VEX
        7: latitude
        8: longitude
        9: poloidal_height
        10: half_width
        11: tilt
        12: flattening
        13: pancaking
        VEX
        14: flux
        STA
        15: latitude
        16: longitude
        17: poloidal_height
        18: half_width
        19: tilt
        20: flattening
        21: pancaking
        22: flux
        """
        evo = Evolution()
        if p[1] < p[2]:
            # res = np.inf
            res = 10.0
            print(res)
            return res
            # return np.nan
            # return 1.0
        evo.toroidal_height = lambda t: (
            (p[1]-p[2])/p[0]*(1.0-np.exp(-p[0]*(t-t0)))+p[2]*(t-t0)+
            toroidal_height_cor
            if t <= ti else
            (p[1]-p[2])/p[0]*(1.0-np.exp(-p[0]*(ti-t0)))+p[2]*(ti-t0)+
            toroidal_height_cor+
            p[3]*(t-ti)
        )
        evo.sigma = lambda t: p[4]
        evo.twist = lambda t: p[5]
        evo.skew = lambda t: skew
        evo.polarity = polarity
        evo.chirality = chirality

        evo.latitude = lambda t: p[7]
        evo.longitude = lambda t: p[8]
        evo.poloidal_height = lambda t: p[9]
        evo.half_width = lambda t: p[10]
        evo.tilt = lambda t: p[11]
        evo.flattening = lambda t: p[12]
        evo.pancaking = lambda t: p[13]
        evo.flux = lambda t: p[6]
        
        tm_mes = np.arange(
            t_mes[0]-delta_mes, 
            t_mes[0]+delta_mes, 
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
        bm_mes, _ = evo.insitu(
            tm_mes, 
            fx_mes, 
            fy_mes, 
            fz_mes
        )
        btm_mes = np.sqrt(bm_mes[:,0]**2+bm_mes[:,1]**2+bm_mes[:,2]**2)
        nzi_mes = np.where(np.isfinite(btm_mes))[0]

        fit_t_mes = np.inf
        fit_bt_mes = np.inf

        if nzi_mes.size > 1 and nzi_mes[0] != 0:
            tm_mes = tm_mes[nzi_mes]
            bm_mes = bm_mes[nzi_mes,:]
            btm_mes = btm_mes[nzi_mes]

            fit_t_mes = np.abs(tm_mes[0]-t_mes[0])/(t_vex[-1]-t_vex[0])
            fit_bt_mes = (
                np.abs(np.median(btm_mes)-bt_mes)/
                np.median(btm_mes)
            )
        else:
            # res = np.inf
            res = 10.0
            print(res)
            return res
            # return np.nan
            # return 1.0

        evo.flux = lambda t: p[14]

        tm_vex = np.arange(
            t_vex[0]-delta_vex, 
            t_vex[-1]+delta_vex, 
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
        bm_vex, _ = evo.insitu(
            tm_vex, 
            fx_vex, 
            fy_vex, 
            fz_vex
        )
        btm_vex = np.sqrt(bm_vex[:,0]**2+bm_vex[:,1]**2+bm_vex[:,2]**2)
        nzi_vex = np.where(np.isfinite(btm_vex))[0]
        
        fit_t_vex = np.inf
        fit_b_vex = np.inf
        fit_bt_vex = np.inf

        if (nzi_vex.size > 1 and 
                nzi_vex[0] != 0 and 
                nzi_vex[-1] != tm_vex.size-1):
            
            tm_vex = tm_vex[nzi_vex]
            bm_vex = bm_vex[nzi_vex,:]
            btm_vex = btm_vex[nzi_vex]

            fit_t_vex = (
                (abs(tm_vex[0]-t_vex[0])+abs(tm_vex[-1]-t_vex[-1]))/
                (t_vex[-1]-t_vex[0])
            )

            m = np.logical_and(
                t_vex >= max(t_vex[0], tm_vex[0]), 
                t_vex <= min(t_vex[-1], tm_vex[-1])
            )
            f = interp1d(
                tm_vex, 
                bm_vex, 
                kind='linear', 
                axis=0
            )
            bf_vex = b_vex[m,:]
            btf_vex = np.sqrt(bf_vex[:,0]**2+bf_vex[:,1]**2+bf_vex[:,2]**2)
            bmf_vex = f(t_vex[m])
            btmf_vex = np.sqrt(bmf_vex[:,0]**2+bmf_vex[:,1]**2+bmf_vex[:,2]**2)
            fit_bt_vex = np.median(np.abs(btf_vex-btmf_vex)/btf_vex)
            fit_b_vex = np.median(
                [np.abs(
                    np.arccos(
                        np.dot(bf_vex[i,:], bmf_vex[i,:])/
                        btf_vex[i]/
                        btmf_vex[i]
                    )
                ) for i in np.arange(bf_vex.shape[0])]
            )/np.pi/2.0

            if not np.isfinite(fit_b_vex):
                fit_b_vex = 1.0
            if not np.isfinite(fit_bt_vex):
                fit_bt_vex = np.abs(np.median(bt_vex)-np.median(btm_vex))/np.median(bt_vex)
        else:
            # res = np.inf
            res = 10.0
            print(res)
            return res
            # return np.nan
            # return 1.0
        
        evo.latitude = lambda t: p[15]
        evo.longitude = lambda t: p[16]
        evo.poloidal_height = lambda t: p[17]
        evo.half_width = lambda t: p[18]
        evo.tilt = lambda t: p[19]
        evo.flattening = lambda t: p[20]
        evo.pancaking = lambda t: p[21]
        evo.flux = lambda t: p[22]

        tm_sta = np.arange(
            t_sta[0]-delta_sta, 
            t_sta[-1]+delta_sta, 
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
            fx_sta, 
            fy_sta, 
            fz_sta
        )
        btm_sta = np.sqrt(bm_sta[:,0]**2+bm_sta[:,1]**2+bm_sta[:,2]**2)
        nzi_sta = np.where(np.isfinite(btm_sta))[0]
        
        fit_t_sta = np.inf
        fit_b_sta = np.inf
        fit_bt_sta = np.inf
        fit_vt_sta = np.inf
        if (nzi_sta.size > 1 and 
                nzi_sta[0] != 0 and 
                nzi_sta[-1] != tm_sta.size-1):
            
            tm_sta = tm_sta[nzi_sta]
            bm_sta = bm_sta[nzi_sta,:]
            btm_sta = btm_sta[nzi_sta]
            vm_sta = vm_sta[nzi_sta]

            fit_t_sta = (
                (abs(tm_sta[0]-t_sta[0])+abs(tm_sta[-1]-t_sta[-1]))/
                (t_sta[-1]-t_sta[0])
            )

            m = np.logical_and(
                t_sta >= max(t_sta[0], tm_sta[0]), 
                t_sta <= min(t_sta[-1], tm_sta[-1])
            )
            f = interp1d(
                tm_sta, 
                bm_sta, 
                kind='linear', 
                axis=0
            )
            bf_sta = b_sta[m,:]
            btf_sta = np.sqrt(bf_sta[:,0]**2+bf_sta[:,1]**2+bf_sta[:,2]**2)
            
            bmf_sta = f(t_sta[m])
            btmf_sta = np.sqrt(bmf_sta[:,0]**2+bmf_sta[:,1]**2+bmf_sta[:,2]**2)
            fit_bt_sta = np.median(np.abs(btf_sta-btmf_sta)/btf_sta)
            fit_b_sta = np.median(
                [np.abs(
                    np.arccos(
                        np.dot(bf_sta[i,:], bmf_sta[i,:])/
                        btf_sta[i]/
                        btmf_sta[i]
                    )
                ) for i in np.arange(bf_sta.shape[0])]
            )/np.pi/2.0

            if not np.isfinite(fit_b_sta):
                fit_b_sta = 1.0
            if not np.isfinite(fit_bt_sta):
                fit_bt_sta = np.abs(np.median(bt_sta)-np.median(btm_sta))/np.median(bt_sta)

            f = interp1d(
                tm_sta, 
                vm_sta, 
                kind='linear', 
                axis=0
            )
            vf_sta = v_sta[m]
            vmf_sta = f(t_sta[m])
            fit_vt_sta = np.median(np.abs(vf_sta-vmf_sta)/vf_sta)

            if not np.isfinite(fit_vt_sta):
                fit_vt_sta = np.abs(np.median(v_sta)-np.median(vm_sta))/np.median(v_sta)
        else:
            # return np.inf
            # res = np.inf
            res = 10.0
            print(res)
            return res
            # return 1.0
        
        res = np.mean(np.array([
            fit_t_mes, fit_bt_mes, 
            fit_t_vex, fit_b_vex, fit_bt_vex,
            fit_t_sta, fit_b_sta, fit_bt_sta, fit_vt_sta
        ])*weights)


        # res = np.mean(
        #     [
        #         fit_t_mes, fit_bt_mes, 
        #         fit_t_vex, fit_b_vex, fit_bt_vex,
        #         fit_t_sta, fit_b_sta, fit_bt_sta, fit_vt_sta
        #     ]
        # )

        if not np.isfinite(res):
            # res = np.inf
            res = 10.0

        # if res == np.inf:
            # res = 1.0
            # res = np.nan
            # res = np.inf
        
        if res < res_prev:
            res_prev = res
            fp = open('./cme1_gp.txt', 'w')
            print('MESSENGER: ', fit_t_mes, fit_bt_mes, file=fp)
            print('VEX: ', fit_t_vex, fit_b_vex, fit_bt_vex, file=fp)
            print('STEREO-A: ', fit_t_sta, fit_b_sta, fit_bt_sta, fit_vt_sta, file=fp)
            print('AVERAGE: ', res, file=fp)
            print('SHARED toroidal_height decay = ', p[0], file=fp)
            print(
                'SHARED toroidal_height speed = ', 
                u.Unit('m/s').to(u.Unit('km/s'), p[1]), 
                file=fp
            )
            print(
                'SHARED toroidal_height speed = ', 
                u.Unit('m/s').to(u.Unit('km/s'), p[2]), 
                file=fp
            )
            print(
                'SHARED toroidal_height speed = ', 
                u.Unit('m/s').to(u.Unit('km/s'), p[3]), 
                file=fp
            )
            print('SHARED sigma = ', p[4], file=fp)
            print('SHARED twist = ', p[5], file=fp)
            print('MESSENGER flux = ', p[6], file=fp)
            print('Venus Express latitude = ', u.rad.to(u.deg, p[7]), file=fp)
            print('Venus Express longitude = ', u.rad.to(u.deg, p[8]), file=fp)
            print('Venus Express poloidal_height = ', u.m.to(u.au, p[9]), file=fp)
            print('Venus Express half_width = ', u.rad.to(u.deg, p[10]), file=fp)
            print('Venus Express tilt = ', u.rad.to(u.deg, p[11]), file=fp)
            print('Venus Express flattening = ', p[12], file=fp)
            print('Venus Express pancaking = ', u.rad.to(u.deg, p[13]), file=fp)
            print('Venus Express flux = ', p[14], file=fp)
            print('STEREO-A latitude = ', u.rad.to(u.deg, p[15]), file=fp)
            print('STEREO-A longitude = ', u.rad.to(u.deg, p[16]), file=fp)
            print('STEREO-A poloidal_height = ', u.m.to(u.au, p[17]), file=fp)
            print('STEREO-A half_width = ', u.rad.to(u.deg, p[18]), file=fp)
            print('STEREO-A tilt = ', u.rad.to(u.deg, p[19]), file=fp)
            print('STEREO-A flattening = ', p[20], file=fp)
            print('STEREO-A pancaking = ', u.rad.to(u.deg, p[21]), file=fp)
            print('STEREO-A flux = ', p[22], file=fp)
            print(p, file=fp)
            fp.close()

            d_vex = np.array(
                [datetime.utcfromtimestamp(t) for t in t_vex]
            )
            dm_vex = np.array(
                [datetime.utcfromtimestamp(t) for t in tm_vex]
            )
            d_sta = np.array(
                [datetime.utcfromtimestamp(t) for t in t_sta]
            )
            dm_sta = np.array(
                [datetime.utcfromtimestamp(t) for t in tm_sta]
            )
            plt.close('all')
            fig = plt.figure()
            plt.subplots_adjust(hspace=0.001)
            
            ax1 = fig.add_subplot(211)
            ax1.plot(t_vex, bt_vex, 'k')
            ax1.plot(t_vex, b_vex[:,0], 'r')
            ax1.plot(t_vex, b_vex[:,1], 'g')
            ax1.plot(t_vex, b_vex[:,2], 'b')
            ax1.plot(tm_vex, btm_vex, '--k')
            ax1.plot(tm_vex, bm_vex[:,0], '--r')
            ax1.plot(tm_vex, bm_vex[:,1], '--g')
            ax1.plot(tm_vex, bm_vex[:,2], '--b')
            ax2 = fig.add_subplot(212)
            ax2.plot(t_sta, bt_sta, 'k')
            ax2.plot(t_sta, b_sta[:,0], 'r')
            ax2.plot(t_sta, b_sta[:,1], 'g')
            ax2.plot(t_sta, b_sta[:,2], 'b')
            ax2.plot(tm_sta, btm_sta, '--k')
            ax2.plot(tm_sta, bm_sta[:,0], '--r')
            ax2.plot(tm_sta, bm_sta[:,1], '--g')
            ax2.plot(tm_sta, bm_sta[:,2], '--b')
            plt.setp(ax1.get_xticklabels(), visible=False)
            plt.ion()
            plt.draw()
            plt.pause(0.001)
            plt.show()

        print(res)
        return res
        # except(TimeoutError):
        #     return 1.0

    """
    SHARED
    0, 1, 2, 3: toroidal_height
    4: sigma
    5: twist
    MES
    6: flux
    MES & VEX
    7: latitude
    8: longitude
    9: poloidal_height
    10: half_width
    11: tilt
    12: flattening
    13: pancaking
    VEX
    14: flux
    STA
    15: latitude
    16: longitude
    17: poloidal_height
    18: half_width
    19: tilt
    20: flattening
    21: pancaking
    22: flux
    """
   
    # MESSENGER:  0.0540540540541 0.00396832829118
    # VEX:  0.72972972973 0.120462610021 0.0127583915335
    # STEREO-A:  0.047619047619 0.331832696083 0.0242011712684 0.251033732211
    # AVERAGE:  0.175073306757
    # SHARED toroidal_height decay =  0.0036167026341954902
    # SHARED toroidal_height speed =  1983.2473331639947
    # SHARED toroidal_height speed =  1244.1623531914918
    # SHARED toroidal_height speed =  1197.8985359569697
    # SHARED sigma =  1.5970727570465675
    # SHARED twist =  0.8034845459864268
    # MESSENGER flux =  411053472957630.6
    # Venus Express latitude =  7.49620333226335
    # Venus Express longitude =  109.55111175791984
    # Venus Express poloidal_height =  0.10030943078702129
    # Venus Express half_width =  36.299242230990714
    # Venus Express tilt =  47.60650881350853
    # Venus Express flattening =  0.4653042831929942
    # Venus Express pancaking =  26.514054163558335
    # Venus Express flux =  390981937416770.5
    # STEREO-A latitude =  11.221704979190427
    # STEREO-A longitude =  132.00707745004368
    # STEREO-A poloidal_height =  0.02905006448518758
    # STEREO-A half_width =  40.698862659239545
    # STEREO-A tilt =  36.65146057428148
    # STEREO-A flattening =  0.46648693251728435
    # STEREO-A pancaking =  20.586420655821588
    # STEREO-A flux =  68398197167775.695

    bounds = [
        # SHARED
        (1e-3, 1e-2),
        tuple(u.Unit('km/s').to(u.Unit('m/s'), (1900.0, 2200.0)).tolist()),
        tuple(u.Unit('km/s').to(u.Unit('m/s'), (1000.0, 1200.0)).tolist()),
        tuple(u.Unit('km/s').to(u.Unit('m/s'), (1800.0, 2200.0)).tolist()),
        (1.5, 2.0),
        (0.1, 1.0),
        # MES
        (1e14, 1e15),
        # MES & VEX
        tuple(u.deg.to(u.rad, (0.0, 30.0)).tolist()),
        tuple(u.deg.to(u.rad, (110.0, 140.0)).tolist()),
        tuple(u.au.to(u.m, (0.05, 0.1)).tolist()),
        tuple(u.deg.to(u.rad, (20.0, 50.0)).tolist()),
        tuple(u.deg.to(u.rad, (30.0, 60.0)).tolist()),
        (0.1, 0.5),
        tuple(u.deg.to(u.rad, (20.0, 40.0)).tolist()),
        (1e14, 1e15),
        # STA
        tuple(u.deg.to(u.rad, (0.0, 30.0)).tolist()),
        tuple(u.deg.to(u.rad, (110.0, 140.0)).tolist()),
        tuple(u.au.to(u.m, (0.01, 0.05)).tolist()),
        tuple(u.deg.to(u.rad, (20.0, 50.0)).tolist()),
        tuple(u.deg.to(u.rad, (30.0, 60.0)).tolist()),
        (0.1, 0.9),
        tuple(u.deg.to(u.rad, (20.0, 40.0)).tolist()),
        (1e13, 1e14),
    ]
    
    # scaler.fit(np.array(bounds).T)

    x0 = np.array(
        [  
            3.18596095e-03, 2.02885855e+06, 1.10137755e+06, 1.93483979e+06,
            1.54299268e+00, 5.69685737e-01, 4.15745797e+14, 9.53233363e-02,
            1.95768770e+00, 1.21121608e+10, 7.55540119e-01, 9.60205562e-01,
            1.42825441e-01, 5.47142290e-01, 3.34977230e+14, 2.73275071e-01,
            2.43285868e+00, 5.38356134e+09, 9.87878035e-01, 6.13401379e-01,
            8.62785166e-01, 3.86288785e-01, 6.36962686e+13
        ]
    )
    # res = basinhopping(
    #     F,
    #     x0=x0*scales,
    #     niter=1000000,
    #     T=0.01,
    #     stepsize=1.0,
    #     minimizer_kwargs=dict(
    #         method='Nelder-Mead',
    #         options=dict(
    #             fatol=0.05,
    #             maxiter=300,
    #             maxfev=300,
    #         ),
    #         tol=0.05,
    #         # method='Powell',
    #         # bounds=scaler.transform(np.array(bounds).T).T.tolist()
    #     ),
    #     # callback=callback,
    #     interval=10,
    #     disp=True
    # )

    # print((np.tile(x0*scales, (50,1))+np.random.randn(50, x0.size)).tolist())

    res = gp_minimize(
        F,
        tuple((np.array(bounds)*scales[:,np.newaxis]).tolist()),
        n_calls=500,
        # x0=(np.tile(x0*scales, (50,1))+np.random.randn(50, x0.size)).tolist(),
        n_jobs=4,
    )

    # res = basinhopping(
    #     F,
    #     x0=scaler.transform(np.array([x0]))[0].tolist(),
    #     T=0.01,
    #     stepsize=0.01,
    #     minimizer_kwargs=dict(
    #         method='COBYLA',
    #         constraints=[
    #             dict(
    #                 type='ineq',
    #                 fun=constraint_cross
    #             )
    #         ],
    #         options=dict(
    #             rhobeg=np.ones(23)*0.01,
    #             tol=0.1,
    #             maxiter=500,
    #             disp=True
    #         ),
    #         tol=0.1,
    #     ),
    #     interval=20,
    #     disp=True
    # )

    # res = minimize(
    #     F,
    #     x0=scaler.transform(np.array([x0]))[0].tolist(),
    #     method='Nelder-Mead',
    # )

    # res = differential_evolution(
    #     F, 
    #     bounds=scaler.transform(np.array(bounds).T).T.tolist(),
    #     strategy='rand1bin',
    #     popsize=10,
    #     mutation=(0.5, 1.0),
    #     recombination=0.9,
    #     polish=False
    # )

    # next attempt: basinhopping + cobyla with constraints

    print(res.x)
    print(res.success)
    print(res.message)
    
    return res

fit2insitu()
