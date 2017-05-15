
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
from scipy.interpolate import interp1d
import calendar
from ai.fri3d.differentialevolution import differential_evolution

res_prev = np.inf
num_eval = 0

def fit2insitu():

    step = 600

    # COR & initial
    d0_cor = datetime(2011, 6, 4, 23, 56)
    t0_cor = calendar.timegm(d0_cor.timetuple())
    latitude_cor = u.deg.to(u.rad, -2.0)
    longitude_cor = u.deg.to(u.rad, 92.0)
    toroidal_height_cor = u.R_sun.to(u.m, 16.5)
    poloidal_height_cor = u.R_sun.to(u.m, 4.5)
    half_width_cor = u.deg.to(u.rad, 30.0)
    tilt_cor = u.deg.to(u.rad, 65.0)
    flattening_cor = 0.3
    pancaking_cor = u.deg.to(u.rad, 18.0)
    skew = 0.0
    polarity = 1.0
    chirality = 1.0
    
    spline_s_phi_kind = 'cubic',
    spline_s_phi_n = 500

    # MESSENGER
    d0_mes = datetime(2011, 6, 5, 4, 40)
    d1_mes = datetime(2011, 6, 5, 9, 29)
    t0_mes = calendar.timegm(d0_mes.timetuple())
    t1_mes = calendar.timegm(d1_mes.timetuple())
    d_mes, b_mes, _, p_mes = getMES(d0_mes, d1_mes)
    t_mes = np.array([calendar.timegm(x.timetuple()) for x in d_mes])
    bt_mes = np.mean(np.sqrt(b_mes[:,0]**2+b_mes[:,1]**2+b_mes[:,2]**2))
    delta_mes = 20*3600

    # VEX
    d0_vex = datetime(2011, 6, 5, 15, 30)
    d1_vex = datetime(2011, 6, 5, 22, 30)
    t0_vex = calendar.timegm(d0_vex.timetuple())
    t1_vex = calendar.timegm(d1_vex.timetuple())
    d_vex, b_vex, _, p_vex = getVEX(d0_vex, d1_vex)
    t_vex = np.array([calendar.timegm(x.timetuple()) for x in d_vex])
    bt_vex = np.sqrt(b_vex[:,0]**2+b_vex[:,1]**2+b_vex[:,2]**2)
    delta_vex = 20*3600

    def F(params):
        global res_prev
        global num_eval
        num_eval += 1
        if num_eval%100 == 0:
            print('NUMBER OF EVALUATIONS = ', num_eval)
        p = np.zeros(23)
        p[0] = params[0]
        p[1] = params[1]
        p[2] = params[2]
        p[3] = 0.0
        p[4] = 1.86287442e+00
        p[5] = 5.45612614e-01
        p[6] = 1e14
        p[7] = params[3]
        p[8] = params[4]
        p[9] = params[5]
        p[10] = params[6]
        p[11] = params[7]
        p[12] = params[8]
        p[13] = params[9]
        p[14] = 1e14
        p[15] = 0.0
        p[16] = 0.0
        p[17] = 0.0
        p[18] = 0.0
        p[19] = 0.0
        p[20] = 0.0
        p[21] = 0.0
        p[22] = 1e14
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
            res = np.inf
            return res
        evo.toroidal_height = lambda t: (
            (p[1]-p[2])/p[0]*(1.0-np.exp(-p[0]*(t-t0_cor)))+p[2]*(t-t0_cor)+
            toroidal_height_cor
        )
        
        evo.sigma = lambda t: p[4]
        evo.twist = lambda t: p[5]
        evo.skew = lambda t: skew
        evo.polarity = polarity
        evo.chirality = chirality

        # MESSENGER

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
        
        if nzi_mes.size > 1 and nzi_mes[0] != 0:
            tm_mes = tm_mes[nzi_mes]
            bm_mes = bm_mes[nzi_mes,:]
            btm_mes = btm_mes[nzi_mes]

            fit_t_mes = np.abs(tm_mes[0]-t_mes[0])/(t_vex[-1]-t_vex[0])
            kappa_mes = bt_mes/np.median(btm_mes)
            p[6] *= kappa_mes
            bm_mes *= kappa_mes
            btm_mes *= kappa_mes
        else:
            res = np.inf
            return res

        # VEX

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
                f = interp1d(
                    (tm_vex-tm_vex[0])/(tm_vex[-1]-tm_vex[0])*(t_vex[-1]-t_vex[0])+t_vex[0],
                    bm_vex,
                    kind='linear',
                    axis=0
                )
                bf_vex = b_vex
                btf_vex = np.sqrt(bf_vex[:,0]**2+bf_vex[:,1]**2+bf_vex[:,2]**2)
                bmf_vex = f(t_vex)
                btmf_vex = np.sqrt(bmf_vex[:,0]**2+bmf_vex[:,1]**2+bmf_vex[:,2]**2)
                fit_b_vex = np.median(
                    [np.abs(
                        np.arccos(
                            np.dot(bf_vex[i,:], bmf_vex[i,:])/
                            btf_vex[i]/
                            btmf_vex[i]
                        )
                    ) for i in np.arange(bf_vex.shape[0])]
                )/np.pi/2.0
                # fit_b_vex = 1.0
            


            kappa_vex = np.median(bt_vex)/np.median(btm_vex)
            p[14] *= kappa_vex
            bm_vex *= kappa_vex
            btm_vex *= kappa_vex
        else:
            res = np.inf
            return res

        res = np.mean(np.array([
            fit_t_mes,
            fit_t_vex, fit_b_vex,
        ]))

        if not np.isfinite(res):
            res = np.inf
        
        if res < res_prev:
            res_prev = res
            fp = open('./cme3v3_run3.txt', 'w')
            print('MESSENGER: ', fit_t_mes, file=fp)
            print('VEX: ', fit_t_vex, fit_b_vex, file=fp)
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
            print('SHARED sigma = ', p[4], file=fp)
            print('SHARED twist = ', p[5], file=fp)
            print('MESSENGER flux = ', p[6], file=fp)
            print('VEX latitude = ', u.rad.to(u.deg, p[7]), file=fp)
            print('VEX longitude = ', u.rad.to(u.deg, p[8]), file=fp)
            print('VEX poloidal_height = ', u.m.to(u.au, p[9]), file=fp)
            print('VEX half_width = ', u.rad.to(u.deg, p[10]), file=fp)
            print('VEX tilt = ', u.rad.to(u.deg, p[11]), file=fp)
            print('VEX flattening = ', p[12], file=fp)
            print('VEX pancaking = ', u.rad.to(u.deg, p[13]), file=fp)
            print('VEX flux = ', p[14], file=fp)
            print(p, file=fp)
            fp.close()

            d_vex = np.array(
                [datetime.utcfromtimestamp(t) for t in t_vex]
            )
            dm_vex = np.array(
                [datetime.utcfromtimestamp(t) for t in tm_vex]
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
            plt.ion()
            plt.draw()
            plt.pause(0.001)
            plt.show()

        return res
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
    bounds = [
        # SHARED
        (1e-4, 1e-2),
        tuple(u.Unit('km/s').to(u.Unit('m/s'), (2000.0, 2500.0)).tolist()),
        tuple(u.Unit('km/s').to(u.Unit('m/s'), (1500.0, 2500.0)).tolist()),
        # tuple(u.Unit('km/s').to(u.Unit('m/s'), (800.0, 2000.0)).tolist()),
        # (1.6, 2.0),
        # (0.0, 1.0),
        # MESSENGER
        # (1e14, 1e15),
        # MES & VEX
        tuple(u.deg.to(u.rad, (-5.0, 10.0)).tolist()),
        tuple(u.deg.to(u.rad, (90.0, 125.0)).tolist()),
        tuple(u.au.to(u.m, (0.06, 0.1)).tolist()),
        tuple(u.deg.to(u.rad, (20.0, 50.0)).tolist()),
        tuple(u.deg.to(u.rad, (35.0, 50.0)).tolist()),
        (0.3, 0.8),
        tuple(u.deg.to(u.rad, (30.0, 50.0)).tolist()),
        # (1e14, 1e15),
        # STA
        # tuple(u.deg.to(u.rad, (-10.0, 10.0)).tolist()),
        # tuple(u.deg.to(u.rad, (90.0, 110.0)).tolist()),
        # tuple(u.au.to(u.m, (0.01, 0.08)).tolist()),
        # tuple(u.deg.to(u.rad, (20.0, 50.0)).tolist()),
        # tuple(u.deg.to(u.rad, (30.0, 100.0)).tolist()),
        # (0.1, 0.9),
        # tuple(u.deg.to(u.rad, (20.0, 40.0)).tolist()),
        # (1e13, 1e14),
    ]
    
    res = differential_evolution(
        F, 
        bounds=bounds,
        strategy='best1bin',
        popsize=100,
        mutation=(0.5, 1.0),
        recombination=0.9,
        disp=True,
        polish=False
    )

    print(res.x)
    print(res.success)
    print(res.message)
    
    return res

fit2insitu()
