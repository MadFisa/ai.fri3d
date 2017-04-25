
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
from scipy.optimize import differential_evolution, brute
import time
import calendar
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

res_prev = np.inf

def fit2insitu():

    step = 600

    # COR & initial
    latitude0 = u.deg.to(u.rad, 30.0)
    longitude0 = u.deg.to(u.rad, 110.0)
    toroidal_height0 = u.R_sun.to(u.m, 12.5)
    poloidal_height0 = u.R_sun.to(u.m, 3.5)
    half_width0 = u.deg.to(u.rad, 40.0)
    tilt0 = u.deg.to(u.rad, 37.0)
    flattening0 = 0.4
    pancaking0 = u.deg.to(u.rad, 25.0)
    skew = 0.0
    twist = np.array([0.1, 2.0])
    flux = np.array([1e13, 1e15])
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
    delta_mes = 10*3600

    # VEX
    d0_vex = datetime(2011, 6, 5, 8, 45)
    d1_vex = datetime(2011, 6, 5, 11, 50)
    t0_vex = calendar.timegm(d0_vex.timetuple())
    t1_vex = calendar.timegm(d1_vex.timetuple())
    d_vex, b_vex, _, p_vex = getVEX(d0_vex, d1_vex)
    t_vex = np.array([calendar.timegm(x.timetuple()) for x in d_vex])
    bt_vex = np.sqrt(b_vex[:,0]**2+b_vex[:,1]**2+b_vex[:,2]**2)
    delta_vex = 10*3600

    print(
        np.median(u.rad.to(u.deg, np.arctan(b_vex[:,1]/b_vex[:,0]))),
        np.median(u.rad.to(u.deg, np.arctan(b_vex[:,2]/np.sqrt(b_vex[:,0]**2+b_vex[:,1]**2))))
    )

    # plt.plot(u.rad.to(u.deg, np.arctan(b_vex[:,1]/b_vex[:,0])))
    # plt.plot(u.rad.to(u.deg, np.arctan(b_vex[:,2]/np.sqrt(b_vex[:,0]**2+b_vex[:,1]**2))))
    # plt.show()

    # STA
    d0_sta = datetime(2011, 6, 6, 12, 25)
    d1_sta = datetime(2011, 6, 6, 14, 10)
    t0_sta = calendar.timegm(d0_sta.timetuple())
    t1_sta = calendar.timegm(d1_sta.timetuple())
    d_sta, b_sta, v_sta, p_sta = getSTA(d0_sta, d1_sta)
    t_sta = np.array([calendar.timegm(x.timetuple()) for x in d_sta])
    bt_sta = np.sqrt(b_sta[:,0]**2+b_sta[:,1]**2+b_sta[:,2]**2)
    
    print(
        np.median(u.rad.to(u.deg, np.arctan(b_sta[:,1]/b_sta[:,0]))),
        np.median(u.rad.to(u.deg, np.arctan(b_sta[:,2]/np.sqrt(b_sta[:,0]**2+b_sta[:,1]**2))))
    )
    
    # plt.plot(u.rad.to(u.deg, np.arctan(b_sta[:,1]/b_sta[:,0])))
    # plt.plot(u.rad.to(u.deg, np.arctan(b_sta[:,2]/np.sqrt(b_sta[:,0]**2+b_sta[:,1]**2))))
    # plt.show()

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

    delta_sta = 10*3600

    di = datetime(2011, 6, 5, 11, 30)
    ti = calendar.timegm(di.timetuple())

    def F(p):
        global res_prev
        evo = Evolution()

        # theta0 = latitude0
        # theta1 = p[0]
        # atheta = p[1]
        # theta1 = p[0]
        # atheta = p[1]

        # evo.latitude = lambda t: (
        #     (theta0-theta1)*np.exp(-atheta*(t-t0))+theta1
        # )
        evo.latitude = lambda t: p[0]

        # phi0 = longitude0
        # vphi = p[2]
        # vphi = p[2]

        # evo.longitude = lambda t: phi0+vphi*(t-t0)
        evo.longitude = lambda t: p[1]

        Rt0 = toroidal_height0
        # aRt = p[3]
        # v0Rt = p[4]
        # v1Rt = p[5]
        # aRt = p[3]
        # v0Rt = p[4]
        # v1Rt = p[5]

        aRt = p[2]
        v0Rt = p[3]
        v1Rt = p[4]

        if v0Rt < v1Rt:
            return np.inf

        # evo.toroidal_height = lambda t: (
        #     (1.0-np.exp(-aRt*(t-t0)))*(v0Rt*(t-t0)+(dRt-Rt0))+Rt0
        #     if t <= ti else
        #     (1.0-np.exp(-aRt*(ti-t0)))*(v0Rt*(ti-t0)+(dRt-Rt0))+Rt0+v1Rt*(t-ti)
        # )
        evo.toroidal_height = lambda t: (
            (v0Rt-v1Rt)/aRt*(1.0-np.exp(-aRt*(t-t0)))+v1Rt*(t-t0)+Rt0
        )

        # Rp0 = poloidal_height0
        # Rp1 = p[6]
        # aRp = p[7]
        # Rp1 = p[6]
        # aRp = p[7]
        
        # evo.poloidal_height = lambda t: (Rp0-Rp1)*np.exp(-aRp*(t-t0))+Rp1
        evo.poloidal_height = lambda t: p[5]

        phihw0 = half_width0
        vphihw = 0.0

        evo.half_width = lambda t: vphihw*(t-t0)+phihw0

        # gamma0 = tilt0
        # vgamma = p[8]
        # vgamma = p[8]
        
        # evo.tilt = lambda t: vgamma*(t-t0)+gamma0
        evo.tilt = lambda t: p[6]

        evo.flattening = lambda t: flattening0
        # evo.flattening = lambda t: p[9]

        thetap0 = pancaking0
        # vthetap = p[9]
        # vthetap = args[9]

        # evo.pancaking = lambda t: vthetap*(t-t0)+thetap0
        evo.pancaking = lambda t: thetap0

        evo.skew = lambda t: skew

        # tau_vex = p[10]
        # tau_sta = p[11]
        # evo.twist = lambda t: tau_vex

        # tau = p[9]
        # evo.twist = lambda t: tau
        evo.twist = lambda t: p[7]

        # F_mes = p[12]
        # F_vex = p[13]
        # F_sta = p[14]
        # F_mes = F_vex = F_sta = 5e14
        # F = p[10]
        # evo.flux = lambda t: F
        evo.flux = lambda t: p[8]

        # evo.sigma = lambda t: p[15]
        evo.sigma = lambda t: 2.0

        evo.polarity = polarity
        evo.chirality = chirality
        # evo.spline_s_phi_kind = spline_s_phi_kind
        # evo.spline_s_phi_n = spline_s_phi_n

        """
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
            # pre_delta_mes = np.abs(tm_mes[nzi_mes[0]]-t0_mes)

            tm_mes = tm_mes[nzi_mes]
            bm_mes = bm_mes[nzi_mes,:]
            btm_mes = btm_mes[nzi_mes]

            fit_t_mes = np.abs(tm_mes[0]-t_mes[0])/(t_vex[-1]-t_vex[0])
            fit_bt_mes = (
                np.abs(np.mean(btm_mes)-bt_mes)/
                np.mean(btm_mes)
            )
            # print('MES | ', fit_t_mes, fit_b_mes)

            # m = np.logical_not(np.isnan(btm_mes))
            # bt_delta_mes = np.abs(np.mean(btm_mes[nzi_mes])-bt_mes)
            # bt_rel_mes = bt_delta_mes/np.mean(btm_mes[nzi_mes])
            # print(np.mean(btm_mes[m]), bt_mes)
        else:
            return np.inf
        """
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
            fit_bt_vex = np.median(np.abs(btf_vex-btmf_vex))/np.median(btf_vex)
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
            return np.inf
        """
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
        fit_v_sta = np.inf

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
            fit_bt_sta = np.median(np.abs(btf_sta-btmf_sta))/np.median(btf_sta)
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
            fit_v_sta = np.median([euclidean(
                vf_sta[i],
                vmf_sta[i]
            ) for i in np.arange(vf_sta.shape[0])])/np.median(vf_sta)

        else:
            return np.inf
        """
        
        res = np.mean([fit_t_vex, fit_b_vex, fit_bt_vex])
        # res = np.mean([fit_t_sta, fit_b_sta, fit_bt_sta, fit_v_sta])
        
        if res < res_prev:
            res_prev = res
            # print('MESSENGER: ', fit_t_mes, fit_bt_mes)
            print('VEX: ', fit_t_vex, fit_b_vex, fit_bt_vex)
            # print('STEREO-A: ', fit_t_sta, fit_b_sta, fit_bt_sta, fit_v_sta)
            print('AVERAGE: ', res)
            print(p)
            # print(
            #     u.rad.to(u.deg, evo.latitude(t0_sta)),
            #     u.rad.to(u.deg, evo.longitude(t0_sta)),
            #     u.rad.to(u.deg, evo.tilt(t0_sta))
            # )
            print(
                u.rad.to(u.deg, evo.latitude(t0_vex)),
                u.rad.to(u.deg, evo.longitude(t0_vex)),
                u.rad.to(u.deg, evo.tilt(t0_vex))
            )
            # print(
                # u.rad.to(u.deg, p[0]),
                # p[1],
                # u.rad.to(u.deg, p[2]),
                # p[3],
                # u.Unit('m/s').to(u.Unit('km/s'), p[4]),
                # # u.m.to(u.R_sun, p[5]),
                # u.Unit('m/s').to(u.Unit('km/s'), p[6]),
                # u.m.to(u.au, p[7]),
                # p[8],
                # u.rad.to(u.deg, p[9]),
            #     p[10],
            #     p[11],
            #     p[12],
            # )

            # tm_vex = t_vex[nzi_vex]
            # bm_vex = bm_vex[nzi_vex,:]
            # tm_sta = t_sta[nzi_sta]
            # bm_sta = bm_sta[nzi_sta,:]

            # print(
            #     (np.abs(tm_vex[0]-t_vex[0])+np.abs(tm_vex[-1]-t_vex[-1]))/
            #     (t_vex[-1]-t_vex[0])
            # )

            d_vex = np.array(
                [datetime.utcfromtimestamp(t) for t in t_vex]
            )
            dm_vex = np.array(
                [datetime.utcfromtimestamp(t) for t in tm_vex]
            )
            """
            d_sta = np.array(
                [datetime.utcfromtimestamp(t) for t in t_sta]
            )
            dm_sta = np.array(
                [datetime.utcfromtimestamp(t) for t in tm_sta]
            )
            """
            plt.close('all')
            fig = plt.figure()
            # plt.subplots_adjust(hspace=0.001)
            ax1 = fig.add_subplot(211)
            ax1.plot(t_vex, bt_vex, 'k')
            ax1.plot(t_vex, b_vex[:,0], 'r')
            ax1.plot(t_vex, b_vex[:,1], 'g')
            ax1.plot(t_vex, b_vex[:,2], 'b')
            ax1.plot(tm_vex, btm_vex, '--k')
            ax1.plot(tm_vex, bm_vex[:,0], '--r')
            ax1.plot(tm_vex, bm_vex[:,1], '--g')
            ax1.plot(tm_vex, bm_vex[:,2], '--b')
            """
            ax2 = fig.add_subplot(212)
            ax2.plot(t_sta, bt_sta, 'k')
            ax2.plot(t_sta, b_sta[:,0], 'r')
            ax2.plot(t_sta, b_sta[:,1], 'g')
            ax2.plot(t_sta, b_sta[:,2], 'b')
            ax2.plot(tm_sta, btm_sta, '--k')
            ax2.plot(tm_sta, bm_sta[:,0], '--r')
            ax2.plot(tm_sta, bm_sta[:,1], '--g')
            ax2.plot(tm_sta, bm_sta[:,2], '--b')
            """
            # plt.setp(ax1.get_xticklabels(), visible=False)
            plt.ion()
            plt.draw()
            plt.pause(0.001)
            plt.show()

        return res

    # 0: theta1
    # 1: atheta
    # 2: vphi
    # 3: aRt
    # 4: v0Rt
    # 5: dRt
    # 6: v1Rt
    # 7: Rp1
    # 8: aRp
    # 9: vgamma
    # 10: tau
    # 11: F

    # geometrical fit
    # bounds = [
    #     u.deg.to(u.rad, (-5.0, 5.0)),
    #     (5e-5, 5e-4),
    #     u.deg.to(u.rad, (-10.0, 10.0))/(t0_sta-t0),
    #     (5e-5, 5e-4),
    #     u.Unit('km/s').to(u.Unit('m/s'), (900.0, 2000.0)),
    #     # u.R_sun.to(u.m, (10.0, 100.0)),
    #     u.R_sun.to(u.m, (10.0, 10.0)), # unused
    #     u.Unit('km/s').to(u.Unit('m/s'), (900.0, 2000.0)),
    #     u.au.to(u.m, (0.01, 0.1)),
    #     (5e-5, 5e-4),
    #     u.deg.to(u.rad, (-20.0, 20.0))/(t0_sta-t0),
    #     (0.5, 0.5),
    #     (1e14, 1e14),
    # ]
    
    bounds = [
        u.deg.to(u.rad, (0.0, 30.0)),
        # u.deg.to(u.rad, (60.0, 90.0)),
        u.deg.to(u.rad, (105.0, 125.0)),
        (5e-5, 5e-4),
        u.Unit('km/s').to(u.Unit('m/s'), (900.0, 2000.0)),
        u.Unit('km/s').to(u.Unit('m/s'), (900.0, 2000.0)),
        u.au.to(u.m, (0.01, 0.1)),
        u.deg.to(u.rad, (35.0, 55.0)),
        (0.0, 0.3),
        (1e13, 1e15),
        # (0.3, 0.5),
    ]


    # bounds = [
    #     u.deg.to(u.rad, (-10.0, 30.0)),
    #     (5e-5, 5e-4),
    #     u.deg.to(u.rad, (-20.0, 20.0))/(t0_sta-t0),
    #     (5e-5, 5e-4),
    #     u.Unit('km/s').to(u.Unit('m/s'), (900.0, 2000.0)),
    #     u.Unit('km/s').to(u.Unit('m/s'), (900.0, 2000.0)),
    #     u.au.to(u.m, (0.01, 0.1)),
    #     (5e-5, 5e-4),
    #     u.deg.to(u.rad, (-20.0, 20.0))/(t0_sta-t0),
    #     (0.0, 1.0),
    #     (1e13, 1e15),
    #     # (1.5, 2.5),
    #     # (1e13, 1e15),
    #     # (1e13, 1e15),
    #     # (2.0, 2.0),
    # ]

    """
    fix everything except Rt
    """

    # magnetic fit
    # bounds = [
    #     (-2.04958689e-02, -2.04958689e-02),
    #     (3.06500305e-04, 3.06500305e-04),
    #     (3.46095731e-07, 3.46095731e-07),
    #     (1.42307095e-04, 1.42307095e-04),
    #     (1.55585227e+06, 1.55585227e+06),
    #     (1.05521505e+06, 1.05521505e+06),
    #     (6.72004614e+09, 6.72004614e+09),
    #     (4.03800688e-04, 4.03800688e-04),
    #     (-5.62764276e-07, -5.62764276e-07),
    #     (0.0, 5.0),
    #     (1e13, 1e15),
    #     (2.0, 2.0),
    # ]
    # bounds = [
    #     (0.0, 1.0),
    #     (1.5, 2.5),
    # ]
    # args = [
    #     -2.04958689e-02,
    #     3.06500305e-04,
    #     3.46095731e-07,
    #     1.42307095e-04,
    #     1.55585227e+06,
    #     1.05521505e+06,
    #     6.72004614e+09,
    #     4.03800688e-04,
    #     -5.62764276e-07,
    # ]



    # res = differential_evolution(F, bounds=bounds, args=args, 
    #     popsize=10, mutation=(1.0,1.9), recombination=0.2)
    res = differential_evolution(F, bounds=bounds)
    # res = brute(F, ranges=bounds)

    print(res.x)

    return res

fit2insitu()

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