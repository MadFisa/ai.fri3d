
from ai.fri3d import FRi3D, Evolution
from ai.shared import cs
from ai.shared.color import BLIND_PALETTE
from astropy import units as u
from scipy.spatial.distance import euclidean
from scipy.signal import correlate
from scipy.interpolate import interp1d
from scipy.optimize import differential_evolution
import numpy as np
import time

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime

db_prev = np.inf

def fit2insitu(t, b, 
    x=u.au.to(u.m, 1.0), 
    y=u.au.to(u.m, 0.0), 
    z=u.au.to(u.m, 0.0),
    period=4.0*24.0*3600.0,
    step_coarse=7200.0,
    step_fine=600.0,
    latitude=np.array([
        u.deg.to(u.rad, [-90.0, 90.0])
    ]),
    longitude=np.array([
        u.deg.to(u.rad, [-180.0, 180.0])
    ]), 
    toroidal_height=np.array([
        u.Unit('km/s').to(u.Unit('m/s'), [200.0, 800.0]), 
        u.au.to(u.m, [0.5, 1.0])
    ]),
    poloidal_height=np.array([
        u.au.to(u.m, [0.1, 0.3])
    ]), 
    half_width=np.array([
        u.deg.to(u.rad, [0.0, 90.0])
    ]), 
    tilt=np.array([
        u.deg.to(u.rad, [-180.0, 180.0])
    ]), 
    flattening=np.array([
        [0.3, 0.7]
    ]), 
    pancaking=np.array([
        u.deg.to(u.rad, [10.0, 90.0])
    ]), 
    skew=np.array([
        u.deg.to(u.rad, [0.0, 40.0])
    ]),
    twist=np.array([
        [1.0, 10.0]
    ]), 
    flux=np.array([
        [1e13, 1e15]
    ]),
    sigma=np.array([
        [1.0, 4.0]
    ]),
    polarity=1.0,
    chirality=1.0, 
    spline_s_phi_kind='linear',
    spline_s_phi_n=100,
    max_pre_time=None,
    max_post_time=None,
    verbose=False,
    timestamp_mask=None):

    t = np.array([time.mktime(x.timetuple()) for x in t])
    _, mask = np.unique(t, return_index=True)
    t = t[mask]
    b = b[mask,:]
    f = interp1d(t, b, kind='linear', axis=0)
    t_real_fine_original = np.arange(t[0], t[-1], step_fine)
    b_real_fine_original = f(t_real_fine_original)

    if timestamp_mask is not None:
        mask = timestamp_mask(t_real_fine_original)
        t_real_fine = t_real_fine_original[mask]
        b_real_fine = b_real_fine_original[mask,:]
    else:
        t_real_fine = t_real_fine_original
        b_real_fine = b_real_fine_original

    db_prev = np.inf

    def F(p):
        global db_prev
        evo = Evolution()
        n = 0
        
        if isinstance(latitude, np.ndarray):
            evo.latitude = lambda t, n=n: \
                np.polyval(p[n:n+latitude.shape[0]], t)
            n += latitude.shape[0]
        else:
            evo.latitude = lambda t: latitude

        if isinstance(longitude, np.ndarray):
            evo.longitude = lambda t, n=n: \
                np.polyval(p[n:n+longitude.shape[0]], t)
            n += longitude.shape[0]
        else:
            evo.longitude = lambda t: longitude

        if isinstance(toroidal_height, np.ndarray):
            evo.toroidal_height = lambda t, n=n: \
                np.polyval(p[n:n+toroidal_height.shape[0]], t)
            n += toroidal_height.shape[0]
        else:
            evo.toroidal_height = lambda t: toroidal_height

        if isinstance(poloidal_height, np.ndarray):
            evo.poloidal_height = lambda t, n=n: \
                np.polyval(p[n:n+poloidal_height.shape[0]], t)
            n += poloidal_height.shape[0]
        else:
            evo.poloidal_height = lambda t: poloidal_height

        if isinstance(half_width, np.ndarray):
            evo.half_width = lambda t, n=n: \
                np.polyval(p[n:n+half_width.shape[0]], t)
            n += half_width.shape[0]
        else:
            evo.half_width = lambda t: half_width

        if isinstance(tilt, np.ndarray):
            evo.tilt = lambda t, n=n: np.polyval(p[n:n+tilt.shape[0]], t)
            n += tilt.shape[0]
        else:
            evo.tilt = lambda t: tilt

        if isinstance(flattening, np.ndarray):
            evo.flattening = lambda t, n=n: \
                np.polyval(p[n:n+flattening.shape[0]], t)
            n += flattening.shape[0]
        else:
            evo.flattening = lambda t: flattening

        if isinstance(pancaking, np.ndarray):
            evo.pancaking = lambda t, n=n: \
                np.polyval(p[n:n+pancaking.shape[0]], t)
            n += pancaking.shape[0]
        else:
            evo.pancaking = lambda t: pancaking

        if isinstance(skew, np.ndarray):
            evo.skew = lambda t, n=n: np.polyval(p[n:n+skew.shape[0]], t)
            n += skew.shape[0]
        else:
            evo.skew = lambda t: skew

        if isinstance(twist, np.ndarray):
            evo.twist = lambda t, n=n: np.polyval(p[n:n+twist.shape[0]], t)
            n += twist.shape[0]
        else:
            evo.twist = lambda t: twist

        if isinstance(flux, np.ndarray):
            evo.flux = lambda t, n=n: np.polyval(p[n:n+flux.shape[0]], t)
            n += flux.shape[0]
        else:
            evo.flux = lambda t: flux

        if isinstance(sigma, np.ndarray):
            evo.sigma = lambda t, n=n: np.polyval(p[n:n+sigma.shape[0]], t)
            n += sigma.shape[0]
        else:
            evo.sigma = lambda t: sigma

        evo.polarity = polarity
        evo.chirality = chirality
        evo.spline_s_phi_kind = spline_s_phi_kind
        evo.spline_s_phi_n = spline_s_phi_n

        t_model_coarse = np.arange(0.0, period+step_coarse, step_coarse)

        b_model_coarse = evo.insitu(t_model_coarse, x, y, z)

        t_start = 0.0

        nonzero_indices = np.nonzero(np.sqrt(
            b_model_coarse[:,0]**2+
            b_model_coarse[:,1]**2+
            b_model_coarse[:,2]**2
        ))[0]

        if nonzero_indices.size >= 2:
            t_model_coarse = \
                t_model_coarse[nonzero_indices[0]:nonzero_indices[-1]+1]
            t_model_coarse -= t_model_coarse[0]
            b_model_coarse = \
                b_model_coarse[nonzero_indices[0]:nonzero_indices[-1]+1,:]
            
            t_start -= nonzero_indices[0]*step_coarse

            # coeff = np.mean(np.sqrt(
            #     b_real_fine[:,0]**2+
            #     b_real_fine[:,1]**2+
            #     b_real_fine[:,2]**2
            # ))/np.mean(np.sqrt(
            #     b_model_coarse[:,0]**2+
            #     b_model_coarse[:,1]**2+
            #     b_model_coarse[:,2]**2
            # ))
            # b_model_coarse *= coeff

            f = interp1d(
                t_model_coarse, 
                b_model_coarse, 
                kind='linear', 
                axis=0, 
                fill_value='extrapolate'
            )
            t_model_fine = np.arange(0.0, t_model_coarse[-1], step_fine)
            b_model_fine = f(t_model_fine)

            cor = (
                correlate(b_model_fine[:,0], b_real_fine_original[:,0])+
                correlate(b_model_fine[:,1], b_real_fine_original[:,1])+
                correlate(b_model_fine[:,2], b_real_fine_original[:,2])
            )/3.0
            shift = np.argmax(cor[t_real_fine_original.size-1:])
            t_model_fine += t_real_fine_original[0]-shift*step_fine

            if t_model_fine.size >= t_real_fine_original.size:
                t_start += t_real_fine_original[0]-shift*step_fine

                pre_time = t_real_fine_original[0]-t_model_fine[0]
                post_time = t_model_fine[-1]-t_real_fine_original[-1]

                # print(pre_time, post_time)

                if pre_time < 0.0 or post_time < 0.0:
                    return np.inf

                if (max_pre_time is not None and pre_time > max_pre_time):
                    return np.inf

                if (max_post_time is not None and post_time > max_post_time):
                    return np.inf

                b_model_fine_ = \
                    b_model_fine[shift:shift+t_real_fine_original.size,:]

                if timestamp_mask is not None:
                    mask = timestamp_mask(t_model_fine)
                    t_model_fine = t_model_fine[mask]
                    b_model_fine = b_model_fine[mask,:]

                db = np.mean([euclidean(
                    b_model_fine_[i,:],
                    b_real_fine[i,:]
                ) for i in range(t_real_fine.size)])
                if db < db_prev:
                    db_prev = db
                    
                    if verbose:
                        print(db, t_start, p)

                        d_real_fine = np.array(
                            [datetime.fromtimestamp(t) for t in t_real_fine]
                        )
                        d_model_fine = np.array(
                            [datetime.fromtimestamp(t) for t in t_model_fine]
                        )
                        plt.close('all')
                        fig = plt.figure()
                        plt.plot(
                            d_real_fine, 
                            np.sqrt(
                                b_real_fine[:,0]**2+
                                b_real_fine[:,1]**2+
                                b_real_fine[:,2]**2
                            ), 
                            'k'
                        )
                        plt.plot(
                            d_real_fine,
                            b_real_fine[:,0], 
                            'r'
                        )
                        plt.plot(
                            d_real_fine,
                            b_real_fine[:,1], 
                            'g'
                        )
                        plt.plot(
                            d_real_fine,
                            b_real_fine[:,2], 
                            'b'
                        )
                        plt.plot(
                            d_model_fine, 
                            np.sqrt(
                                b_model_fine[:,0]**2+
                                b_model_fine[:,1]**2+
                                b_model_fine[:,2]**2
                            ), 
                            '--k'
                        )
                        plt.plot(
                            d_model_fine,
                            b_model_fine[:,0],
                            '--r'
                        )
                        plt.plot(
                            d_model_fine,
                            b_model_fine[:,1],
                            '--g'
                        )
                        plt.plot(
                            d_model_fine,
                            b_model_fine[:,2],
                            '--b'
                        )
                        plt.ion()
                        plt.draw()
                        plt.pause(0.001)
                        plt.show()
                return db
        return np.inf

    bounds = []

    if isinstance(latitude, np.ndarray):
        for i in range(latitude.shape[0]):
            bounds.append((latitude[i,0], latitude[i,-1])) 

    if isinstance(longitude, np.ndarray):
        for i in range(longitude.shape[0]):
            bounds.append((longitude[i,0], longitude[i,-1])) 

    if isinstance(toroidal_height, np.ndarray):
        for i in range(toroidal_height.shape[0]):
            bounds.append((toroidal_height[i,0], toroidal_height[i,-1])) 

    if isinstance(poloidal_height, np.ndarray):
        for i in range(poloidal_height.shape[0]):
            bounds.append((poloidal_height[i,0], poloidal_height[i,-1])) 

    if isinstance(half_width, np.ndarray):
        for i in range(half_width.shape[0]):
            bounds.append((half_width[i,0], half_width[i,-1])) 

    if isinstance(tilt, np.ndarray):
        for i in range(tilt.shape[0]):
            bounds.append((tilt[i,0], tilt[i,-1])) 

    if isinstance(flattening, np.ndarray):
        for i in range(flattening.shape[0]):
            bounds.append((flattening[i,0], flattening[i,-1])) 

    if isinstance(pancaking, np.ndarray):
        for i in range(pancaking.shape[0]):
            bounds.append((pancaking[i,0], pancaking[i,-1])) 

    if isinstance(skew, np.ndarray):
        for i in range(skew.shape[0]):
            bounds.append((skew[i,0], skew[i,-1])) 

    if isinstance(twist, np.ndarray):
        for i in range(twist.shape[0]):
            bounds.append((twist[i,0], twist[i,-1])) 

    if isinstance(flux, np.ndarray):
        for i in range(flux.shape[0]):
            bounds.append((flux[i,0], flux[i,-1])) 

    if isinstance(sigma, np.ndarray):
        for i in range(sigma.shape[0]):
            bounds.append((sigma[i,0], sigma[i,-1])) 

    res = differential_evolution(F, bounds=bounds)

    print(res.x)

    return res

def fit2insitu2(
    t1, b1, x1, y1, z1, 
    t2, b2, x2, y2, z2,
    t0,
    period=4.0*24.0*3600.0,
    step_coarse=7200.0,
    step_fine=600.0,
    max_pre_time1=2*3600.0,
    max_post_time1=2*3600.0,
    max_pre_time2=2*3600.0,
    max_post_time2=2*3600.0,
    latitude=np.array([
        u.deg.to(u.rad, [-90.0, 90.0])
    ]),
    longitude=np.array([
        u.deg.to(u.rad, [-180.0, 180.0])
    ]), 
    toroidal_height=np.array([
        [-0.4, 0.4],
        u.Unit('km/s').to(u.Unit('m/s'), [200.0, 800.0]), 
        u.au.to(u.m, [0.5, 1.0])
    ]),
    poloidal_height=np.array([
        [-0.4, 0.0], 
        u.Unit('km/s').to(u.Unit('m/s'), [0.0, 200.0]), 
        u.au.to(u.m, [0.1, 0.3])
    ]), 
    half_width=np.array([
        u.deg.to(u.rad, [0.0, 90.0])
    ]), 
    tilt=np.array([
        u.deg.to(u.rad, [-180.0, 180.0])
    ]), 
    flattening=np.array([
        [0.3, 0.8]
    ]), 
    pancaking=np.array([
        u.deg.to(u.rad, [10.0, 90.0])
    ]), 
    skew=np.array([
        u.deg.to(u.rad, [0.0, 40.0])
    ]),
    twist=np.array([
        [0.5, 10.0]
    ]), 
    flux=np.array([
        [1e13, 1e15]
    ]),
    sigma=np.array([
        [1.0, 4.0]
    ]),
    polarity=1.0,
    chirality=1.0, 
    spline_s_phi_kind='linear',
    spline_s_phi_n=100,
    
    verbose=False,
    timestamp_mask=None):

    db_prev = np.inf

    def F(p):
        global db_prev
        evo = Evolution()
        n = 0
        
        if isinstance(latitude, np.ndarray):
            evo.latitude = lambda t, n=n: \
                np.polyval(p[n:n+latitude.shape[0]], t-t0)
            n += latitude.shape[0]
        else:
            evo.latitude = lambda t: latitude

        if isinstance(longitude, np.ndarray):
            evo.longitude = lambda t, n=n: \
                np.polyval(p[n:n+longitude.shape[0]], t-t0)
            n += longitude.shape[0]
        else:
            evo.longitude = lambda t: longitude

        if isinstance(toroidal_height, np.ndarray):
            evo.toroidal_height = lambda t, n=n: \
                np.polyval(p[n:n+toroidal_height.shape[0]], t-t0)
            n += toroidal_height.shape[0]
        else:
            evo.toroidal_height = lambda t: toroidal_height

        if isinstance(poloidal_height, np.ndarray):
            evo.poloidal_height = lambda t, n=n: \
                np.polyval(p[n:n+poloidal_height.shape[0]], t-t0)
            n += poloidal_height.shape[0]
        else:
            evo.poloidal_height = lambda t: poloidal_height

        if isinstance(half_width, np.ndarray):
            evo.half_width = lambda t, n=n: \
                np.polyval(p[n:n+half_width.shape[0]], t-t0)
            n += half_width.shape[0]
        else:
            evo.half_width = lambda t: half_width

        if isinstance(tilt, np.ndarray):
            evo.tilt = lambda t, n=n: np.polyval(p[n:n+tilt.shape[0]], t-t0)
            n += tilt.shape[0]
        else:
            evo.tilt = lambda t: tilt

        if isinstance(flattening, np.ndarray):
            evo.flattening = lambda t, n=n: \
                np.polyval(p[n:n+flattening.shape[0]], t-t0)
            n += flattening.shape[0]
        else:
            evo.flattening = lambda t: flattening

        if isinstance(pancaking, np.ndarray):
            evo.pancaking = lambda t, n=n: \
                np.polyval(p[n:n+pancaking.shape[0]], t-t0)
            n += pancaking.shape[0]
        else:
            evo.pancaking = lambda t: pancaking

        if isinstance(skew, np.ndarray):
            evo.skew = lambda t, n=n: np.polyval(p[n:n+skew.shape[0]], t-t0)
            n += skew.shape[0]
        else:
            evo.skew = lambda t: skew

        if isinstance(twist, np.ndarray):
            evo.twist = lambda t, n=n: np.polyval(p[n:n+twist.shape[0]], t-t0)
            n += twist.shape[0]
        else:
            evo.twist = lambda t: twist

        if isinstance(flux, np.ndarray):
            evo.flux = lambda t, n=n: np.polyval(p[n:n+flux.shape[0]], t-t0)
            n += flux.shape[0]
        else:
            evo.flux = lambda t: flux

        if isinstance(sigma, np.ndarray):
            evo.sigma = lambda t, n=n: np.polyval(p[n:n+sigma.shape[0]], t-t0)
            n += sigma.shape[0]
        else:
            evo.sigma = lambda t: sigma

        evo.polarity = polarity
        evo.chirality = chirality
        evo.spline_s_phi_kind = spline_s_phi_kind
        evo.spline_s_phi_n = spline_s_phi_n

        t1_model_coarse = np.arange(t0, t0+period+step_coarse, step_coarse)
        b1_model_coarse = evo.insitu(t_model_coarse, x1, y1, z1)

        nonzero_indices = np.nonzero(np.sqrt(
            b1_model_coarse[:,0]**2+
            b1_model_coarse[:,1]**2+
            b1_model_coarse[:,2]**2
        ))[0]

        if nonzero_indices.size < 2:
            return np.inf

        t1_model_coarse = \
            t1_model_coarse[nonzero_indices[0]:nonzero_indices[-1]+1]
        b1_model_coarse = \
            b1_model_coarse[nonzero_indices[0]:nonzero_indices[-1]+1,:]

        if (np.abs(t1_model_coarse[0]-t1[0]) > max_pre_time1 or
            np.abs(t1_model_coarse[-1]-t1[-1]) > max_post_time1):
            return np.inf

        t2_model_coarse = np.arange(t0, t0+period+step_coarse, step_coarse)
        b2_model_coarse = evo.insitu(t2_model_coarse, x2, y2, z2)

        nonzero_indices = np.nonzero(np.sqrt(
            b1_model_coarse[:,0]**2+
            b1_model_coarse[:,1]**2+
            b1_model_coarse[:,2]**2
        ))[0]

        if nonzero_indices.size < 2:
            return np.inf

        t2_model_coarse = \
            t2_model_coarse[nonzero_indices[0]:nonzero_indices[-1]+1]
        b2_model_coarse = \
            b2_model_coarse[nonzero_indices[0]:nonzero_indices[-1]+1,:]

        if (np.abs(t2_model_coarse[0]-t2[0]) > max_pre_time2 or
            np.abs(t2_model_coarse[-1]-t2[-1]) > max_post_time2):
            return np.inf

        f = interp1d(
            t1_model_coarse, 
            b1_model_coarse, 
            kind='linear', 
            axis=0, 
            fill_value='extrapolate'
        )
        t1_model_fine = np.arange(
            t1_model_coarse[0], 
            t1_model_coarse[-1], 
            step_fine
        )
        b1_model_fine = f(t1_model_fine)

        f = interp1d(
            t2_model_coarse, 
            b2_model_coarse, 
            kind='linear', 
            axis=0, 
            fill_value='extrapolate'
        )
        t2_model_fine = np.arange(
            t2_model_coarse[0], 
            t2_model_coarse[-1], 
            step_fine
        )
        b2_model_fine = f(t2_model_fine)










        t_start = 0.0

        nonzero_indices = np.nonzero(np.sqrt(
            b_model_coarse[:,0]**2+
            b_model_coarse[:,1]**2+
            b_model_coarse[:,2]**2
        ))[0]

        if nonzero_indices.size >= 2:
            t_model_coarse = \
                t_model_coarse[nonzero_indices[0]:nonzero_indices[-1]+1]
            t_model_coarse -= t_model_coarse[0]
            b_model_coarse = \
                b_model_coarse[nonzero_indices[0]:nonzero_indices[-1]+1,:]
            
            t_start -= nonzero_indices[0]*step_coarse

            # coeff = np.mean(np.sqrt(
            #     b_real_fine[:,0]**2+
            #     b_real_fine[:,1]**2+
            #     b_real_fine[:,2]**2
            # ))/np.mean(np.sqrt(
            #     b_model_coarse[:,0]**2+
            #     b_model_coarse[:,1]**2+
            #     b_model_coarse[:,2]**2
            # ))
            # b_model_coarse *= coeff

            f = interp1d(
                t_model_coarse, 
                b_model_coarse, 
                kind='linear', 
                axis=0, 
                fill_value='extrapolate'
            )
            t_model_fine = np.arange(0.0, t_model_coarse[-1], step_fine)
            b_model_fine = f(t_model_fine)

            cor = (
                correlate(b_model_fine[:,0], b_real_fine_original[:,0])+
                correlate(b_model_fine[:,1], b_real_fine_original[:,1])+
                correlate(b_model_fine[:,2], b_real_fine_original[:,2])
            )/3.0
            shift = np.argmax(cor[t_real_fine_original.size-1:])
            t_model_fine += t_real_fine_original[0]-shift*step_fine

            if t_model_fine.size >= t_real_fine_original.size:
                t_start += t_real_fine_original[0]-shift*step_fine

                pre_time = t_real_fine_original[0]-t_model_fine[0]
                post_time = t_model_fine[-1]-t_real_fine_original[-1]

                # print(pre_time, post_time)

                if pre_time < 0.0 or post_time < 0.0:
                    return np.inf

                if (max_pre_time is not None and pre_time > max_pre_time):
                    return np.inf

                if (max_post_time is not None and post_time > max_post_time):
                    return np.inf

                b_model_fine_ = \
                    b_model_fine[shift:shift+t_real_fine_original.size,:]

                if timestamp_mask is not None:
                    mask = timestamp_mask(t_model_fine)
                    t_model_fine = t_model_fine[mask]
                    b_model_fine = b_model_fine[mask,:]

                db = np.mean([euclidean(
                    b_model_fine_[i,:],
                    b_real_fine[i,:]
                ) for i in range(t_real_fine.size)])
                if db < db_prev:
                    db_prev = db
                    
                    if verbose:
                        print(db, t_start, p)

                        d_real_fine = np.array(
                            [datetime.fromtimestamp(t) for t in t_real_fine]
                        )
                        d_model_fine = np.array(
                            [datetime.fromtimestamp(t) for t in t_model_fine]
                        )
                        plt.close('all')
                        fig = plt.figure()
                        plt.plot(
                            d_real_fine, 
                            np.sqrt(
                                b_real_fine[:,0]**2+
                                b_real_fine[:,1]**2+
                                b_real_fine[:,2]**2
                            ), 
                            'k'
                        )
                        plt.plot(
                            d_real_fine,
                            b_real_fine[:,0], 
                            'r'
                        )
                        plt.plot(
                            d_real_fine,
                            b_real_fine[:,1], 
                            'g'
                        )
                        plt.plot(
                            d_real_fine,
                            b_real_fine[:,2], 
                            'b'
                        )
                        plt.plot(
                            d_model_fine, 
                            np.sqrt(
                                b_model_fine[:,0]**2+
                                b_model_fine[:,1]**2+
                                b_model_fine[:,2]**2
                            ), 
                            '--k'
                        )
                        plt.plot(
                            d_model_fine,
                            b_model_fine[:,0],
                            '--r'
                        )
                        plt.plot(
                            d_model_fine,
                            b_model_fine[:,1],
                            '--g'
                        )
                        plt.plot(
                            d_model_fine,
                            b_model_fine[:,2],
                            '--b'
                        )
                        plt.ion()
                        plt.draw()
                        plt.pause(0.001)
                        plt.show()
                return db
        return np.inf

    bounds = []

    if isinstance(latitude, np.ndarray):
        for i in range(latitude.shape[0]):
            bounds.append((latitude[i,0], latitude[i,-1])) 

    if isinstance(longitude, np.ndarray):
        for i in range(longitude.shape[0]):
            bounds.append((longitude[i,0], longitude[i,-1])) 

    if isinstance(toroidal_height, np.ndarray):
        for i in range(toroidal_height.shape[0]):
            bounds.append((toroidal_height[i,0], toroidal_height[i,-1])) 

    if isinstance(poloidal_height, np.ndarray):
        for i in range(poloidal_height.shape[0]):
            bounds.append((poloidal_height[i,0], poloidal_height[i,-1])) 

    if isinstance(half_width, np.ndarray):
        for i in range(half_width.shape[0]):
            bounds.append((half_width[i,0], half_width[i,-1])) 

    if isinstance(tilt, np.ndarray):
        for i in range(tilt.shape[0]):
            bounds.append((tilt[i,0], tilt[i,-1])) 

    if isinstance(flattening, np.ndarray):
        for i in range(flattening.shape[0]):
            bounds.append((flattening[i,0], flattening[i,-1])) 

    if isinstance(pancaking, np.ndarray):
        for i in range(pancaking.shape[0]):
            bounds.append((pancaking[i,0], pancaking[i,-1])) 

    if isinstance(skew, np.ndarray):
        for i in range(skew.shape[0]):
            bounds.append((skew[i,0], skew[i,-1])) 

    if isinstance(twist, np.ndarray):
        for i in range(twist.shape[0]):
            bounds.append((twist[i,0], twist[i,-1])) 

    if isinstance(flux, np.ndarray):
        for i in range(flux.shape[0]):
            bounds.append((flux[i,0], flux[i,-1])) 

    if isinstance(sigma, np.ndarray):
        for i in range(sigma.shape[0]):
            bounds.append((sigma[i,0], sigma[i,-1])) 

    res = differential_evolution(F, bounds=bounds)

    print(res.x)

    return res

def fit2remote(
    cor2a=False,
    cor2a_img=None,
    cor2a_aov=u.deg.to(u.rad, 4.0),
    cor2a_xc=0.0,
    cor2a_yc=0.0,
    sta_r=None,
    sta_lat=None,
    sta_lon=None,

    cor2b=False,
    cor2b_img=None,
    cor2b_aov=u.deg.to(u.rad, 4.0),
    cor2b_xc=0.0,
    cor2b_yc=0.0,
    stb_r=None,
    stb_lat=None,
    stb_lon=None,
    
    c3=False,
    c3_img=None,
    c3_fov=u.R_sun.to(u.m, 30.0),
    c3_xc=0.0,
    c3_yc=0.0,
    soho_r=u.au.to(u.m, 1.0),
    soho_lat=u.deg.to(u.rad, 0.0),
    soho_lon=u.deg.to(u.rad, 0.0),

    latitude=u.deg.to(u.rad, 0.0),
    longitude=u.deg.to(u.rad, 0.0),
    toroidal_height=u.R_sun.to(u.m, 12.5),
    poloidal_height=u.R_sun.to(u.m, 3.5),
    half_width=u.deg.to(u.rad, 40.0),
    tilt=u.deg.to(u.rad, 0.0),
    flattening=0.5,
    pancaking=u.deg.to(u.rad, 20.0),
    skew=u.deg.to(u.rad, 0.0),
    
    spline_s_phi_kind='linear',
    spline_s_phi_n=100):

    fr = FRi3D(
        latitude=latitude, 
        longitude=longitude, 
        toroidal_height=toroidal_height, 
        poloidal_height=poloidal_height, 
        half_width=half_width, 
        tilt=tilt, 
        flattening=flattening, 
        pancaking=pancaking, 
        skew=skew,
        spline_s_phi_kind=spline_s_phi_kind,
        spline_s_phi_n=spline_s_phi_n
    )
    fr.init()

    x0, y0, z0 = fr.shell()

    fig = plt.figure()

    sc = cor2b+c3+cor2a

    gs = gridspec.GridSpec(2, sc)
    gs.update(wspace=0.0, hspace=0.0)

    i = 0
    
    if cor2b:
        cor2b_fov = stb_r*np.tan(cor2b_aov)
        ax = plt.subplot(gs[i])
        ax.imshow(
            plt.imread(cor2b_img),
            zorder=0,
            extent=[
                -cor2b_fov-cor2b_xc, cor2b_fov-cor2b_xc, 
                -cor2b_fov-cor2b_yc, cor2b_fov-cor2b_yc
            ]
        )
        ax.set_xlim([-cor2b_fov-cor2b_xc, cor2b_fov-cor2b_xc])
        ax.set_ylim([-cor2b_fov-cor2b_yc, cor2b_fov-cor2b_yc])
        ax.set_axis_bgcolor('black')
        plt.axis('off')

        ax = plt.subplot(gs[i+sc])
        ax.imshow(
            plt.imread(cor2b_img),
            zorder=0,
            extent=[
                -cor2b_fov-cor2b_xc, cor2b_fov-cor2b_xc, 
                -cor2b_fov-cor2b_yc, cor2b_fov-cor2b_yc
            ]
        )
        # ax.plot([0.0], [0.0], '.y', markersize=5.0)
        T = cs.mx_rot_y(stb_lat)*cs.mx_rot_z(-stb_lon)
        x, y, z = cs.mx_apply(T, x0, y0, z0)
        # x = T[0,0]*x0+T[0,1]*y0+T[0,2]*z0
        # y = T[1,0]*x0+T[1,1]*y0+T[1,2]*z0
        # z = T[2,0]*x0+T[2,1]*y0+T[2,2]*z0
        y = stb_r/(stb_r-x)*y
        z = stb_r/(stb_r-x)*z
        ax.scatter(y, z, 3, 
            color=BLIND_PALETTE['yellow'], 
            marker='.'
        )
        ax.set_xlim([-cor2b_fov-cor2b_xc, cor2b_fov-cor2b_xc])
        ax.set_ylim([-cor2b_fov-cor2b_yc, cor2b_fov-cor2b_yc])
        ax.set_axis_bgcolor('black')
        plt.axis('off')

        i += 1

    if c3:
        ax = plt.subplot(gs[i])
        ax.imshow(
            plt.imread(c3_img),
            zorder=0,
            extent=[
                -c3_fov-c3_xc, c3_fov-c3_xc, 
                -c3_fov-c3_yc, c3_fov-c3_yc
            ]
        )
        ax.set_xlim([-c3_fov-c3_xc, c3_fov-c3_xc])
        ax.set_ylim([-c3_fov-c3_yc, c3_fov-c3_yc])
        ax.set_axis_bgcolor('black')
        plt.axis('off')

        ax = plt.subplot(gs[i+sc])
        ax.imshow(
            plt.imread(c3_img),
            zorder=0,
            extent=[
                -c3_fov-c3_xc, c3_fov-c3_xc, 
                -c3_fov-c3_yc, c3_fov-c3_yc
            ]
        )
        # ax.plot([0.0], [0.0], '.y', markersize=5.0)
        T = cs.mx_rot_y(soho_lat)*cs.mx_rot_z(-soho_lon)
        x, y, z = cs.mx_apply(T, x0, y0, z0)
        # x = T[0,0]*x0+T[0,1]*y0+T[0,2]*z0
        # y = T[1,0]*x0+T[1,1]*y0+T[1,2]*z0
        # z = T[2,0]*x0+T[2,1]*y0+T[2,2]*z0
        y = soho_r/(soho_r-x)*y
        z = soho_r/(soho_r-x)*z
        ax.scatter(y, z, 3, 
            color=BLIND_PALETTE['yellow'], 
            marker='.'
        )
        ax.set_xlim([-c3_fov-c3_xc, c3_fov-c3_xc])
        ax.set_ylim([-c3_fov-c3_yc, c3_fov-c3_yc])
        ax.set_axis_bgcolor('black')
        plt.axis('off')

        i += 1

    if cor2a:
        cor2a_fov = sta_r*np.tan(cor2a_aov)
        ax = plt.subplot(gs[i])
        ax.imshow(
            plt.imread(cor2a_img),
            zorder=0,
            extent=[
                -cor2a_fov-cor2a_xc, cor2a_fov-cor2a_xc, 
                -cor2a_fov-cor2a_yc, cor2a_fov-cor2a_yc
            ]
        )
        ax.set_xlim([-cor2a_fov-cor2a_xc, cor2a_fov-cor2a_xc])
        ax.set_ylim([-cor2a_fov-cor2a_yc, cor2a_fov-cor2a_yc])
        ax.set_axis_bgcolor('black')
        plt.axis('off')

        ax = plt.subplot(gs[i+sc])
        ax.imshow(
            plt.imread(cor2a_img),
            zorder=0,
            extent=[
                -cor2a_fov-cor2a_xc, cor2a_fov-cor2a_xc, 
                -cor2a_fov-cor2a_xc, cor2a_fov-cor2a_yc
            ]
        )
        # ax.plot([0.0], [0.0], '.y', markersize=5.0)
        T = cs.mx_rot_y(sta_lat)*cs.mx_rot_z(-sta_lon)
        x, y, z = cs.mx_apply(T, x0, y0, z0)
        # x = T[0,0]*x0+T[0,1]*y0+T[0,2]*z0
        # y = T[1,0]*x0+T[1,1]*y0+T[1,2]*z0
        # z = T[2,0]*x0+T[2,1]*y0+T[2,2]*z0
        y = sta_r/(sta_r-x)*y
        z = sta_r/(sta_r-x)*z
        ax.scatter(y, z, 3, 
            color=BLIND_PALETTE['yellow'], 
            marker='.'
        )
        ax.set_xlim([-cor2a_fov-cor2a_xc, cor2a_fov-cor2a_xc])
        ax.set_ylim([-cor2a_fov-cor2a_yc, cor2a_fov-cor2a_yc])
        ax.set_axis_bgcolor('black')
        plt.axis('off')

        i += 1
    
    plt.show()
