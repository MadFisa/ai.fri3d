import numpy as np
from ai.fri3d import BaseFRi3DFit
from ai.fri3d import DynamicFRi3D

class FRi3DFitInSitu(BaseFRi3DFit):
    def __init__(self, **kwargs):
        super(FRi3DFitInSitu, self).__init__(DynamicFRi3D, **kwargs)

    def fit(self, **kwargs):
        self._dfr = DynamicFRi3D()
        self._latitude = kwargs.get(
            'latitude',
            ConstProfile(self._dfr.latitude(0))
        )
        self._longitude = kwargs.get(
            'longitude',
            ConstProfile(self._dfr.longitude(0))
        )
        def residual(**kwargs):
            pass
        res = differential_evolution(residual, bounds=bounds)

    @property
    def latitude(self):
        return self._latitude

    @property
    def longitude(self):
        return self._longitude

class BaseProfile:
    def __init__(self, profile):
        self._profile = profile

    @property
    def profile(self):
        return self._profile

class ConstProfile:
    def __init__(self, value):
        super(PolyProfile, self).__init__('const')
        self._value = value

    def eval(self, t):
        return self._value

    @property
    def params(self, t):
        return self._value

    @property
    def bounds(self, t):
        return None

class PolyProfile(BaseProfile):
    def __init__(self, params, bounds):
        super(PolyProfile, self).__init__('poly')
        self._params = params
        self._bounds = bounds

    def eval(self, t):
        np.polyval(self._params, t)

    @property
    def params(self):
        return self._params

    @property
    def bounds(self):
        return self._bounds

class ExpProfile(BaseProfile):
    pass

def fit2insitu(t, b, v,
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
        [1.0, 3.0]
    ]),
    polarity=1.0,
    chirality=1.0, 
    spline_s_phi_kind='linear',
    spline_s_phi_n=100,
    max_pre_time=None,
    max_post_time=None,
    verbose=False,
    timestamp_mask=None,
    fit_speed=True):

    t = np.array([time.mktime(x.timetuple()) for x in t])
    _, mask = np.unique(t, return_index=True)
    t = t[mask]
    b = b[mask,:]
    v = v[mask]
    t_real_fine_original = np.arange(t[0], t[-1], step_fine)
    f = interp1d(t, b, kind='linear', axis=0)
    b_real_fine_original = f(t_real_fine_original)
    f = interp1d(t, v, kind='linear', axis=0)
    v_real_fine_original = f(t_real_fine_original)

    if timestamp_mask is not None:
        mask = timestamp_mask(t_real_fine_original)
        t_real_fine = t_real_fine_original[mask]
        b_real_fine = b_real_fine_original[mask,:]
        v_real_fine = v_real_fine_original[mask]
    else:
        t_real_fine = t_real_fine_original
        b_real_fine = b_real_fine_original
        v_real_fine = v_real_fine_original

    # db_prev = np.inf
    # dv_prev = np.inf
    dd_prev = np.inf

    def F(p):
        global dd_prev
        evo = Evolution()
        n = 0
        
        if isinstance(latitude, np.ndarray):
            evo.latitude = lambda t, n=n: \
                np.polyval(p[n:n+latitude.shape[0]], t)
            n += latitude.shape[0]
        elif isinstance(latitude, LambdaType):
            evo.latitude = latitude
        else:
            evo.latitude = lambda t: latitude

        if isinstance(longitude, np.ndarray):
            evo.longitude = lambda t, n=n: \
                np.polyval(p[n:n+longitude.shape[0]], t)
            n += longitude.shape[0]
        elif isinstance(longitude, LambdaType):
            evo.longitude = longitude
        else:
            evo.longitude = lambda t: longitude

        if isinstance(toroidal_height, np.ndarray):
            evo.toroidal_height = lambda t, n=n: \
                np.polyval(p[n:n+toroidal_height.shape[0]], t)
            n += toroidal_height.shape[0]
        elif isinstance(toroidal_height, LambdaType):
            evo.toroidal_height = toroidal_height
        else:
            evo.toroidal_height = lambda t: toroidal_height

        if isinstance(poloidal_height, np.ndarray):
            evo.poloidal_height = lambda t, n=n: \
                np.polyval(p[n:n+poloidal_height.shape[0]], t)
            n += poloidal_height.shape[0]
        elif isinstance(poloidal_height, LambdaType):
            evo.poloidal_height = poloidal_height
        else:
            evo.poloidal_height = lambda t: poloidal_height

        if isinstance(half_width, np.ndarray):
            evo.half_width = lambda t, n=n: \
                np.polyval(p[n:n+half_width.shape[0]], t)
            n += half_width.shape[0]
        elif isinstance(half_width, LambdaType):
            evo.half_width = half_width
        else:
            evo.half_width = lambda t: half_width

        if isinstance(tilt, np.ndarray):
            evo.tilt = lambda t, n=n: np.polyval(p[n:n+tilt.shape[0]], t)
            n += tilt.shape[0]
        elif isinstance(tilt, LambdaType):
            evo.tilt = tilt
        else:
            evo.tilt = lambda t: tilt

        if isinstance(flattening, np.ndarray):
            evo.flattening = lambda t, n=n: \
                np.polyval(p[n:n+flattening.shape[0]], t)
            n += flattening.shape[0]
        elif isinstance(flattening, LambdaType):
            evo.flattening = flattening
        else:
            evo.flattening = lambda t: flattening

        if isinstance(pancaking, np.ndarray):
            evo.pancaking = lambda t, n=n: \
                np.polyval(p[n:n+pancaking.shape[0]], t)
            n += pancaking.shape[0]
        elif isinstance(pancaking, LambdaType):
            evo.pancaking = pancaking
        else:
            evo.pancaking = lambda t: pancaking

        if isinstance(skew, np.ndarray):
            evo.skew = lambda t, n=n: np.polyval(p[n:n+skew.shape[0]], t)
            n += skew.shape[0]
        elif isinstance(skew, LambdaType):
            evo.skew = skew
        else:
            evo.skew = lambda t: skew

        if isinstance(twist, np.ndarray):
            evo.twist = lambda t, n=n: np.polyval(p[n:n+twist.shape[0]], t)
            n += twist.shape[0]
        elif isinstance(twist, LambdaType):
            evo.twist = twist
        else:
            evo.twist = lambda t: twist

        if isinstance(flux, np.ndarray):
            evo.flux = lambda t, n=n: np.polyval(p[n:n+flux.shape[0]], t)
            n += flux.shape[0]
        elif isinstance(flux, LambdaType):
            evo.flux = flux
        else:
            evo.flux = lambda t: flux

        if isinstance(sigma, np.ndarray):
            evo.sigma = lambda t, n=n: np.polyval(p[n:n+sigma.shape[0]], t)
            n += sigma.shape[0]
        elif isinstance(sigma, LambdaType):
            evo.sigma = sigma
        else:
            evo.sigma = lambda t: sigma

        evo.polarity = polarity
        evo.chirality = chirality
        evo.spline_s_phi_kind = spline_s_phi_kind
        evo.spline_s_phi_n = spline_s_phi_n

        t_model_coarse = np.arange(0.0, period+step_coarse, step_coarse)

        b_model_coarse, v_model_coarse = evo.insitu(t_model_coarse, x, y, z)

        t_start = 0.0

        nonzero_indices = np.where(np.isfinite(np.sqrt(
            b_model_coarse[:,0]**2+
            b_model_coarse[:,1]**2+
            b_model_coarse[:,2]**2
        )))[0]
        
        if nonzero_indices.size >= 2:
            t_model_coarse = \
                t_model_coarse[nonzero_indices[0]:nonzero_indices[-1]+1]
            t_model_coarse -= t_model_coarse[0]
            b_model_coarse = \
                b_model_coarse[nonzero_indices[0]:nonzero_indices[-1]+1,:]
            v_model_coarse = \
                v_model_coarse[nonzero_indices[0]:nonzero_indices[-1]+1]
            
            t_start -= nonzero_indices[0]*step_coarse

            t_model_fine = np.arange(0.0, t_model_coarse[-1], step_fine)
            
            f = interp1d(
                t_model_coarse, 
                b_model_coarse, 
                kind='linear', 
                axis=0, 
                fill_value='extrapolate'
            )
            b_model_fine = f(t_model_fine)

            f = interp1d(
                t_model_coarse, 
                v_model_coarse, 
                kind='linear', 
                axis=0, 
                fill_value='extrapolate'
            )
            v_model_fine = f(t_model_fine)

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
                v_model_fine_ = \
                    v_model_fine[shift:shift+t_real_fine_original.size]

                if timestamp_mask is not None:
                    mask = timestamp_mask(t_model_fine)
                    t_model_fine = t_model_fine[mask]
                    b_model_fine = b_model_fine[mask,:]
                    v_model_fine = v_model_fine[mask]

                # print(
                #     b_model_fine.shape, 
                #     b_real_fine.shape,
                #     v_model_fine.shape,
                #     v_real_fine.shape
                # )

                # print(
                #     np.any(np.isnan(b_real_fine)),
                #     np.any(np.isnan(v_real_fine))
                # )

                mask = np.logical_not(np.any(np.isnan(b_real_fine), axis=1))
                db = np.mean([euclidean(
                    b_model_fine[i,:],
                    b_real_fine[i,:]
                ) for i in np.arange(t_real_fine.size)[mask]])
                if fit_speed == True:
                    mask = np.logical_not(np.isnan(v_real_fine))
                    dv = np.mean([np.abs(
                        v_model_fine[i]-v_real_fine[i]
                    ) for i in np.arange(t_real_fine.size)[mask]])
                    dd = (
                        db/np.nanmax(np.abs(b_real_fine_original))+
                        dv/np.nanmax(np.abs(v_real_fine_original))
                    )
                else:
                    dd = db/np.nanmax(np.abs(b_real_fine_original))

                # print(db, dv, dd)

                # plt.plot(v_real_fine)
                # plt.plot(v_model_fine)
                # plt.show()

                if dd < dd_prev:
                    dd_prev = dd
                    
                    if verbose:
                        print(dd, t_start, p)

                        d_real_fine = np.array(
                            [datetime.fromtimestamp(t) for t in t_real_fine]
                        )
                        d_model_fine = np.array(
                            [datetime.fromtimestamp(t) for t in t_model_fine]
                        )
                        plt.close('all')
                        fig = plt.figure()
                        plt.subplots_adjust(hspace=0.001)
                        ax1 = fig.add_subplot(211)
                        ax1.plot(
                            d_real_fine, 
                            np.sqrt(
                                b_real_fine[:,0]**2+
                                b_real_fine[:,1]**2+
                                b_real_fine[:,2]**2
                            ), 
                            'k'
                        )
                        ax1.plot(
                            d_real_fine,
                            b_real_fine[:,0], 
                            'r'
                        )
                        ax1.plot(
                            d_real_fine,
                            b_real_fine[:,1], 
                            'g'
                        )
                        ax1.plot(
                            d_real_fine,
                            b_real_fine[:,2], 
                            'b'
                        )
                        ax1.plot(
                            d_model_fine, 
                            np.sqrt(
                                b_model_fine[:,0]**2+
                                b_model_fine[:,1]**2+
                                b_model_fine[:,2]**2
                            ), 
                            '--k'
                        )
                        ax1.plot(
                            d_model_fine,
                            b_model_fine[:,0],
                            '--r'
                        )
                        ax1.plot(
                            d_model_fine,
                            b_model_fine[:,1],
                            '--g'
                        )
                        ax1.plot(
                            d_model_fine,
                            b_model_fine[:,2],
                            '--b'
                        )
                        ax2 = fig.add_subplot(212, sharex=ax1)
                        ax2.plot(
                            d_real_fine,
                            v_real_fine,
                            'k'
                        )
                        ax2.plot(
                            d_model_fine,
                            v_model_fine,
                            '--k'
                        )

                        plt.setp(ax1.get_xticklabels(), visible=False)
                        plt.ion()
                        plt.draw()
                        plt.pause(0.001)
                        plt.show()
                return dd
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