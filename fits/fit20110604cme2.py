
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
from scipy.optimize import differential_evolution, fmin_l_bfgs_b, basinhopping
import time
import calendar
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn import preprocessing
from scipy.signal import savgol_filter

res_prev = np.inf



def fit2insitu():

    step = 600

    # COR & initial
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
    d0 = datetime(2011, 6, 4, 22, 54)
    t0 = calendar.timegm(d0.timetuple())

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
    delta_mes = 30*3600

    # VEX
    d0_vex = datetime(2011, 6, 5, 15, 30)
    d1_vex = datetime(2011, 6, 5, 22, 30)
    t0_vex = calendar.timegm(d0_vex.timetuple())
    t1_vex = calendar.timegm(d1_vex.timetuple())
    d_vex, b_vex, _, p_vex = getVEX(d0_vex, d1_vex)
    b_vex = savgol_filter(b_vex, int(d_vex.size/2.0//2*2+1), 3, axis=0)
    t_vex = np.array([calendar.timegm(x.timetuple()) for x in d_vex])
    bt_vex = np.sqrt(b_vex[:,0]**2+b_vex[:,1]**2+b_vex[:,2]**2)
    delta_vex = 30*3600

    # STA
    d0_sta = datetime(2011, 6, 6, 16, 30)
    d1_sta = datetime(2011, 6, 7, 1)
    t0_sta = calendar.timegm(d0_sta.timetuple())
    t1_sta = calendar.timegm(d1_sta.timetuple())
    d_sta, b_sta, _, p_sta = getSTA(d0_sta, d1_sta)
    b_sta = savgol_filter(b_sta, int(d_sta.size/2.0//2*2+1), 3, axis=0)
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
    v_sta = u.Unit('km/s').to(u.Unit('m/s'), f(t_sta))
    v_sta = savgol_filter(v_sta, int(v_sta.size/2.0//2*2+1), 3)
    
    delta_sta = 30*3600

    di = datetime(2011, 6, 5, 11, 30)
    ti = calendar.timegm(di.timetuple())

    scaler = preprocessing.MinMaxScaler()
    
    def F(p):
        global res_prev
        p = scaler.inverse_transform(np.array([p]))[0].tolist()
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
            # return np.inf
            return np.nan
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
            # return np.inf
            return np.nan
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
        else:
            # return np.inf
            return np.nan
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

            f = interp1d(
                tm_sta, 
                vm_sta, 
                kind='linear', 
                axis=0
            )
            vf_sta = v_sta[m]
            vmf_sta = f(t_sta[m])
            fit_vt_sta = np.median(np.abs(vf_sta-vmf_sta)/vf_sta)
        else:
            # return np.inf
            return np.nan
            # return 1.0
        
        res = np.mean(
            [
                fit_t_mes, fit_bt_mes, 
                fit_t_vex, fit_b_vex, fit_bt_vex,
                fit_t_sta, fit_b_sta, fit_bt_sta, fit_vt_sta
            ]
        )

        if res == np.inf:
            # res = 1.0
            res = np.nan
        
        if res < res_prev:
            res_prev = res
            print('MESSENGER: ', fit_t_mes, fit_bt_mes)
            print('VEX: ', fit_t_vex, fit_b_vex, fit_bt_vex)
            print('STEREO-A: ', fit_t_sta, fit_b_sta, fit_bt_sta, fit_vt_sta)
            print('AVERAGE: ', res)
            print('toroidal_height decay = ', p[0])
            print(
                'SHARED toroidal_height speed = ', 
                u.Unit('m/s').to(u.Unit('km/s'), p[1])
            )
            print(
                'SHARED toroidal_height speed = ', 
                u.Unit('m/s').to(u.Unit('km/s'), p[2])
            )
            print(
                'SHARED toroidal_height speed = ', 
                u.Unit('m/s').to(u.Unit('km/s'), p[3])
            )
            print('SHARED sigma = ', p[4])
            print('SHARED twist = ', p[5])
            print('MESSENGER flux = ', p[6])
            print('Venus Express latitude = ', u.rad.to(u.deg, p[7]))
            print('Venus Express longitude = ', u.rad.to(u.deg, p[8]))
            print('Venus Express poloidal_height = ', u.m.to(u.au, p[9]))
            print('Venus Express half_width = ', u.rad.to(u.deg, p[10]))
            print('Venus Express tilt = ', u.rad.to(u.deg, p[11]))
            print('Venus Express flattening = ', p[12])
            print('Venus Express pancaking = ', u.rad.to(u.deg, p[13]))
            print('Venus Express flux = ', p[14])
            print('STEREO-A latitude = ', u.rad.to(u.deg, p[15]))
            print('STEREO-A longitude = ', u.rad.to(u.deg, p[16]))
            print('STEREO-A poloidal_height = ', u.m.to(u.au, p[17]))
            print('STEREO-A half_width = ', u.rad.to(u.deg, p[18]))
            print('STEREO-A tilt = ', u.rad.to(u.deg, p[19]))
            print('STEREO-A flattening = ', p[20])
            print('STEREO-A pancaking = ', u.rad.to(u.deg, p[21]))
            print('STEREO-A flux = ', p[22])
            
            print(p)

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
        (1e-3, 1e-2),
        tuple(u.Unit('km/s').to(u.Unit('m/s'), (2400.0, 2600.0)).tolist()),
        tuple(u.Unit('km/s').to(u.Unit('m/s'), (1500.0, 1700.0)).tolist()),
        tuple(u.Unit('km/s').to(u.Unit('m/s'), (1100.0, 1300.0)).tolist()),
        (1.6, 2.0),
        (0.9, 1.3),
        # MES
        (1e14, 1e15),
        # MES & VEX
        tuple(u.deg.to(u.rad, (0.0, 20.0)).tolist()),
        tuple(u.deg.to(u.rad, (110.0, 130.0)).tolist()),
        tuple(u.au.to(u.m, (0.03, 0.09)).tolist()),
        tuple(u.deg.to(u.rad, (20.0, 40.0)).tolist()),
        tuple(u.deg.to(u.rad, (30.0, 50.0)).tolist()),
        (0.3, 0.5),
        tuple(u.deg.to(u.rad, (20.0, 40.0)).tolist()),
        (1e14, 1e15),
        # STA
        tuple(u.deg.to(u.rad, (0.0, 20.0)).tolist()),
        tuple(u.deg.to(u.rad, (110.0, 130.0)).tolist()),
        tuple(u.au.to(u.m, (0.02, 0.06)).tolist()),
        tuple(u.deg.to(u.rad, (20.0, 40.0)).tolist()),
        tuple(u.deg.to(u.rad, (30.0, 50.0)).tolist()),
        (0.3, 0.5),
        tuple(u.deg.to(u.rad, (20.0, 40.0)).tolist()),
        (1e13, 1e14),
    ]
    
    scaler.fit(np.array(bounds).T)

    x0 = [0.0012586916416986423, 2444002.7422559243, 1547845.9583524799, 1194746.5296720322, 1.8223714090039305, 1.2384413492572999, 675577937864485.2, 0.07461232452845426, 2.098935897713696, 11722889185.787663, 0.4487002673163215, 0.8594261117035951, 0.3756881370549535, 0.6349368433913544, 549947380479405.1, 0.10282435140382482, 2.2243779886461166, 7345243868.796222, 0.6170123309208321, 0.5744943494439477, 0.4511159612130795, 0.5347397433295806, 52989014798912.96]

    res = differential_evolution(
        F, 
        bounds=scaler.transform(np.array(bounds).T).T.tolist(),
        strategy='best2bin',
        popsize=10,
        mutation=(0.1, 1.0),
        recombination=0.1,
        polish=False
    )
    # res = basinhopping(
    #     F,
    #     x0=scaler.transform(np.array([x0]))[0].tolist(),
    #     T=0.1,
    #     stepsize=0.05,
    #     minimizer_kwargs=dict(
    #         method='L-BFGS-B',
    #         # method='TNC',
    #         bounds=scaler.transform(np.array(bounds).T).T.tolist(),
    #         options=dict(
    #             eps=0.05
    #         )
    #     ),
    #     interval=10
    # )
    
    return res

fit2insitu()
