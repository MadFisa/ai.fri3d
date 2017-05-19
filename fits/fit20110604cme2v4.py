
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

    # STA
    d0_sta = datetime(2011, 6, 6, 12, 25)
    d1_sta = datetime(2011, 6, 6, 14, 10)
    t0_sta = calendar.timegm(d0_sta.timetuple())
    t1_sta = calendar.timegm(d1_sta.timetuple())
    d_sta, b_sta, _, p_sta = getSTA(d0_sta, d1_sta)
    t_sta = np.array([calendar.timegm(x.timetuple()) for x in d_sta])
    bt_sta = np.sqrt(b_sta[:,0]**2+b_sta[:,1]**2+b_sta[:,2]**2)

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
        p[7] = 0.0
        p[8] = 0.0
        p[9] = 0.0
        p[10] = 0.0
        p[11] = 0.0
        p[12] = 0.0
        p[13] = 0.0
        p[14] = 1e14
        p[15] = params[4]
        p[16] = params[5]
        p[17] = params[6]
        p[18] = half_width_cor
        p[19] = params[7]
        p[20] = params[8]
        p[21] = params[9]
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
            fit_t_sta, fit_b_sta, fit_vt_sta
        ]))

        if not np.isfinite(res):
            res = np.inf

        if res < res_prev:
            res_prev = res
            fp = open('./cme2v4_run2.txt', 'w')
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

            d_sta = np.array(
                [datetime.utcfromtimestamp(t) for t in t_sta]
            )
            dm_sta = np.array(
                [datetime.utcfromtimestamp(t) for t in tm_sta]
            )
            plt.close('all')
            fig = plt.figure()
            plt.subplots_adjust(hspace=0.001)
            
            ax2 = fig.add_subplot(212)
            ax2.plot(t_sta, bt_sta, 'k')
            ax2.plot(t_sta, b_sta[:,0], 'r')
            ax2.plot(t_sta, b_sta[:,1], 'g')
            ax2.plot(t_sta, b_sta[:,2], 'b')
            ax2.plot(tm_sta, btm_sta, '--k')
            ax2.plot(tm_sta, bm_sta[:,0], '--r')
            ax2.plot(tm_sta, bm_sta[:,1], '--g')
            ax2.plot(tm_sta, bm_sta[:,2], '--b')
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
        tuple(u.Unit('km/s').to(u.Unit('m/s'), (1000.0, 1600.0)).tolist()),
        (1.6, 2.0),
        (0.0, 1.0),
        # MES
        # (1e14, 1e15),
        # MES & VEX
        # tuple(u.deg.to(u.rad, (10.0, 20.0)).tolist()),
        # tuple(u.deg.to(u.rad, (110.0, 125.0)).tolist()),
        # tuple(u.au.to(u.m, (0.01, 0.1)).tolist()),
        # tuple(u.deg.to(u.rad, (20.0, 40.0)).tolist()),
        # tuple(u.deg.to(u.rad, (-50.0, 50.0)).tolist()),
        # (0.2, 0.8),
        # tuple(u.deg.to(u.rad, (30.0, 40.0)).tolist()),
        # (1e14, 1e15),
        # STA
        tuple(u.deg.to(u.rad, (-20.0, 20.0)).tolist()),
        tuple(u.deg.to(u.rad, (80.0, 130.0)).tolist()),
        tuple(u.au.to(u.m, (0.005, 0.04)).tolist()),
        # tuple(u.deg.to(u.rad, (20.0, 40.0)).tolist()),
        tuple(u.deg.to(u.rad, (-70.0, -20.0)).tolist()),
        (0.2, 0.4),
        tuple(u.deg.to(u.rad, (10.0, 30.0)).tolist()),
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

# STEREO-A:  0.047619047619 0.0846761294283 0.013111095246
# AVERAGE:  0.0484687574311
# SHARED toroidal_height speed =  1074.13697467
# SHARED toroidal_height speed =  1000.62862142
# SHARED sigma =  1.8473254712
# SHARED twist =  0.351453804492
# STEREO-A latitude =  -18.3088164634
# STEREO-A longitude =  81.6636726523
# STEREO-A poloidal_height =  0.030154291444
# STEREO-A half_width =  39.2969475439
# STEREO-A tilt =  -61.1322776563
# STEREO-A flattening =  0.201437153817
# STEREO-A pancaking =  35.847787406
# STEREO-A flux =  2.40762503817e+14
# [  0.00000000e+00   0.00000000e+00   1.07413697e+06   1.00062862e+06
#    1.84732547e+00   3.51453804e-01   1.00000000e+14   0.00000000e+00
#    0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
#    0.00000000e+00   0.00000000e+00   1.00000000e+14  -3.19549129e-01
#    1.42529997e+00   4.51101779e+09   6.85861121e-01  -1.06695952e+00
#    2.01437154e-01   6.25661920e-01   2.40762504e+14]




# STEREO-A:  0.904761904762 0.371791498672 0.0735345318287
# AVERAGE:  0.450029311754
# SHARED toroidal_height speed =  1405.89612148
# SHARED toroidal_height speed =  1045.99595555
# SHARED sigma =  1.91598495508
# SHARED twist =  0.412730776438
# STEREO-A latitude =  -16.4452299199
# STEREO-A longitude =  87.1759045761
# STEREO-A poloidal_height =  0.0280694521151
# STEREO-A half_width =  29.5725030646
# STEREO-A tilt =  -52.0206710733
# STEREO-A flattening =  0.510711828628
# STEREO-A pancaking =  39.2396705617
# STEREO-A flux =  2.28819532223e+14
# [  0.00000000e+00   0.00000000e+00   1.40589612e+06   1.04599596e+06
#    1.91598496e+00   4.12730776e-01   1.00000000e+14   0.00000000e+00
#    0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
#    0.00000000e+00   0.00000000e+00   1.00000000e+14  -2.87023408e-01
#    1.52150656e+00   4.19913027e+09   5.16137547e-01  -9.07931989e-01
#    5.10711829e-01   6.84861449e-01   2.28819532e+14]