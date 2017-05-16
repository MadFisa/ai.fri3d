
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

    # COR
    d0_cor = datetime(2011, 6, 4, 22, 54)
    t0_cor = calendar.timegm(d0_cor.timetuple())
    latitude_cor = u.deg.to(u.rad, 22.0)
    longitude_cor = u.deg.to(u.rad, 125.0)
    toroidal_height_cor = u.R_sun.to(u.m, 12.0)
    poloidal_height_cor = u.R_sun.to(u.m, 4.0)
    half_width_cor = u.deg.to(u.rad, 35.0)
    tilt_cor = u.deg.to(u.rad, 35.0)
    flattening_cor = 0.4
    pancaking_cor = u.deg.to(u.rad, 30.0)
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
    delta_mes = 10*3600

    # VEX
    d0_vex = datetime(2011, 6, 5, 15, 30)
    d1_vex = datetime(2011, 6, 5, 22, 30)
    t0_vex = calendar.timegm(d0_vex.timetuple())
    t1_vex = calendar.timegm(d1_vex.timetuple())
    d_vex, b_vex, _, p_vex = getVEX(d0_vex, d1_vex)
    t_vex = np.array([calendar.timegm(x.timetuple()) for x in d_vex])
    bt_vex = np.sqrt(b_vex[:,0]**2+b_vex[:,1]**2+b_vex[:,2]**2)
    # m = t_vex >= calendar.timegm(datetime(2011, 6, 5, 19).timetuple())
    # ba_vex = np.median(bt_vex[m])
    # b_vex[np.logical_not(m),0] *= ba_vex/bt_vex[np.logical_not(m)]
    # b_vex[np.logical_not(m),1] *= ba_vex/bt_vex[np.logical_not(m)]
    # b_vex[np.logical_not(m),2] *= ba_vex/bt_vex[np.logical_not(m)]
    # bt_vex[np.logical_not(m)] *= ba_vex/bt_vex[np.logical_not(m)]
    delta_vex = 10*3600

    phi = u.rad.to(u.deg, np.arctan(b_vex[:,1]/b_vex[:,0]))
    theta = u.rad.to(u.deg, np.arctan(b_vex[:,2]/np.sqrt(b_vex[:,0]**2+b_vex[:,1]**2)))
    plt.figure()
    plt.plot(phi, 'r')
    plt.plot(theta, 'g')
    # plt.show()


    # STA
    d0_sta = datetime(2011, 6, 6, 12, 25)
    d1_sta = datetime(2011, 6, 6, 14, 10)
    t0_sta = calendar.timegm(d0_sta.timetuple())
    t1_sta = calendar.timegm(d1_sta.timetuple())
    d_sta, b_sta, _, p_sta = getSTA(d0_sta, d1_sta)
    t_sta = np.array([calendar.timegm(x.timetuple()) for x in d_sta])
    bt_sta = np.sqrt(b_sta[:,0]**2+b_sta[:,1]**2+b_sta[:,2]**2)

    phi = u.rad.to(u.deg, np.arctan(b_sta[:,1]/b_sta[:,0]))
    theta = u.rad.to(u.deg, np.arctan(b_sta[:,2]/np.sqrt(b_sta[:,0]**2+b_sta[:,1]**2)))
    plt.figure()
    plt.plot(phi, 'r')
    plt.plot(theta, 'g')
    plt.show()
    
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
    vt_sta = u.Unit('km/s').to(u.Unit('m/s'), f(t_sta))
    
    delta_sta = 10*3600

    di = datetime(2011, 6, 5, 11, 30)
    ti = calendar.timegm(di.timetuple())

    def F(params):
        global res_prev
        global num_eval
        num_eval += 1
        if num_eval%100 == 0:
            print('NUMBER OF EVALUATIONS = ', num_eval)
        p = np.zeros(23)
        p[0] = 0.0
        p[1] = 0.0
        p[2] = params[0]
        p[3] = params[1]
        p[4] = params[2]
        p[5] = params[3]
        p[6] = 1e14
        p[7] = params[4]
        p[8] = params[5]
        p[9] = params[6]
        p[10] = params[7]
        p[11] = params[8]
        p[12] = params[9]
        p[13] = params[10]
        p[14] = 1e14
        p[15] = params[11]
        p[16] = params[12]
        p[17] = params[13]
        p[18] = params[14]
        p[19] = params[15]
        p[20] = params[16]
        p[21] = params[17]
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
        evo.toroidal_height = lambda t: (
            p[2]*(t-t0_cor)+toroidal_height_cor
            if t <= ti else
            p[2]*(ti-t0_cor)+toroidal_height_cor+p[3]*(t-ti)
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
            )/np.pi*2.0

            if not np.isfinite(fit_b_vex):
                fit_b_vex = 1.0
            
            kappa_vex = np.median(bt_vex)/np.median(btm_vex)
            p[14] *= kappa_vex
            bm_vex *= kappa_vex
            btm_vex *= kappa_vex
        else:
            res = np.inf
            return res

        # STA
            
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
        bm_sta, vtm_sta = evo.insitu(
            tm_sta, 
            fx_sta, 
            fy_sta, 
            fz_sta
        )
        btm_sta = np.sqrt(bm_sta[:,0]**2+bm_sta[:,1]**2+bm_sta[:,2]**2)
        nzi_sta = np.where(np.isfinite(btm_sta))[0]
        
        fit_t_sta = np.inf
        fit_b_sta = np.inf
        fit_vt_sta = np.inf
        if (nzi_sta.size > 1 and 
                nzi_sta[0] != 0 and 
                nzi_sta[-1] != tm_sta.size-1):
            
            tm_sta = tm_sta[nzi_sta]
            bm_sta = bm_sta[nzi_sta,:]
            btm_sta = btm_sta[nzi_sta]
            vtm_sta = vtm_sta[nzi_sta]

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
            fit_b_sta = np.median(
                [np.abs(
                    np.arccos(
                        np.dot(bf_sta[i,:], bmf_sta[i,:])/
                        btf_sta[i]/
                        btmf_sta[i]
                    )
                ) for i in np.arange(bf_sta.shape[0])]
            )/np.pi*2.0

            if not np.isfinite(fit_b_sta):
                fit_b_sta = 1.0
            
            kappa_sta = np.median(bt_sta)/np.median(btm_sta)
            p[22] *= kappa_sta
            bm_sta *= kappa_sta
            btm_sta *= kappa_sta

            f = interp1d(
                tm_sta, 
                vtm_sta, 
                kind='linear', 
                axis=0
            )
            vtf_sta = vt_sta[m]
            vtmf_sta = f(t_sta[m])
            fit_vt_sta = np.median(np.abs(vtf_sta-vtmf_sta)/vtf_sta)

            if not np.isfinite(fit_vt_sta):
                fit_vt_sta = np.abs(np.median(vt_sta)-np.median(vtm_sta))/np.median(vt_sta)
        else:
            res = np.inf
            return res

        res = np.mean(np.array([
            fit_t_mes, 
            fit_t_vex, fit_b_vex,
            fit_t_sta, fit_b_sta, fit_vt_sta
        ]))

        if not np.isfinite(res):
            res = np.inf

        if res < res_prev:
            res_prev = res
            fp = open('./cme2v2_run2.txt', 'w')
            print('MESSENGER: ', fit_t_mes, file=fp)
            print('VEX: ', fit_t_vex, fit_b_vex, file=fp)
            print('STEREO-A: ', fit_t_sta, fit_b_sta, fit_vt_sta, file=fp)
            print('AVERAGE: ', res, file=fp)
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
            print('VEX latitude = ', u.rad.to(u.deg, p[7]), file=fp)
            print('VEX longitude = ', u.rad.to(u.deg, p[8]), file=fp)
            print('VEX poloidal_height = ', u.m.to(u.au, p[9]), file=fp)
            print('VEX half_width = ', u.rad.to(u.deg, p[10]), file=fp)
            print('VEX tilt = ', u.rad.to(u.deg, p[11]), file=fp)
            print('VEX flattening = ', p[12], file=fp)
            print('VEX pancaking = ', u.rad.to(u.deg, p[13]), file=fp)
            print('VEX flux = ', p[14], file=fp)
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
        # (1e-1, 1e-2),
        # tuple(u.Unit('km/s').to(u.Unit('m/s'), (1800.0, 2400.0)).tolist()),
        tuple(u.Unit('km/s').to(u.Unit('m/s'), (1000.0, 2000.0)).tolist()),
        tuple(u.Unit('km/s').to(u.Unit('m/s'), (1000.0, 1500.0)).tolist()),
        (1.6, 2.0),
        (0.0, 1.0),
        # MES
        # (1e14, 1e15),
        # MES & VEX
        tuple(u.deg.to(u.rad, (-20.0, 20.0)).tolist()),
        tuple(u.deg.to(u.rad, (80.0, 130.0)).tolist()),
        tuple(u.au.to(u.m, (0.01, 0.1)).tolist()),
        tuple(u.deg.to(u.rad, (20.0, 40.0)).tolist()),
        tuple(u.deg.to(u.rad, (-80.0, 80.0)).tolist()),
        (0.2, 0.8),
        tuple(u.deg.to(u.rad, (30.0, 40.0)).tolist()),
        # (1e14, 1e15),
        # STA
        tuple(u.deg.to(u.rad, (-20.0, 20.0)).tolist()),
        tuple(u.deg.to(u.rad, (80.0, 130.0)).tolist()),
        tuple(u.au.to(u.m, (0.01, 0.1)).tolist()),
        tuple(u.deg.to(u.rad, (20.0, 40.0)).tolist()),
        tuple(u.deg.to(u.rad, (-80.0, 0.0)).tolist()),
        (0.2, 0.8),
        tuple(u.deg.to(u.rad, (30.0, 40.0)).tolist()),
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

fit2insitu()

# MESSENGER:  0.0714285714286
# VEX:  0.0238095238095 0.147885690074
# STEREO-A:  0.047619047619 0.256383357026 0.0764026480497
# AVERAGE:  0.103921473001
# SHARED toroidal_height decay =  0.074880278671
# SHARED toroidal_height speed =  2158.73067257
# SHARED toroidal_height speed =  1607.18562046
# SHARED toroidal_height speed =  1168.41863675
# SHARED sigma =  1.78277984837
# SHARED twist =  0.417389768315
# MESSENGER flux =  9.30223125511e+14
# VEX latitude =  15.4165425873
# VEX longitude =  121.314131546
# VEX poloidal_height =  0.0931936834373
# VEX half_width =  31.9491511754
# VEX tilt =  60.2708575138
# VEX flattening =  0.867515533104
# VEX pancaking =  34.030985596
# VEX flux =  4.27330783546e+14
# STEREO-A latitude =  18.2209764426
# STEREO-A longitude =  115.144557025
# STEREO-A poloidal_height =  0.0118936690065
# STEREO-A half_width =  30.3747451069
# STEREO-A tilt =  72.7271727662
# STEREO-A flattening =  0.725633433038
# STEREO-A pancaking =  36.8518912917
# STEREO-A flux =  7.04993499698e+13
# [  7.48802787e-02   2.15873067e+06   1.60718562e+06   1.16841864e+06
#    1.78277985e+00   4.17389768e-01   9.30223126e+14   2.69069427e-01
#    2.11733102e+00   1.39415766e+10   5.57617881e-01   1.05192491e+00
#    8.67515533e-01   5.93952746e-01   4.27330784e+14   3.18016032e-01
#    2.00965164e+00   1.77926756e+09   5.30139312e-01   1.26932862e+00
#    7.25633433e-01   6.43186839e-01   7.04993500e+13]
