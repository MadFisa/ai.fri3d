
from ai.fri3d import Evolution
from astropy import units as u
from scipy.spatial.distance import euclidean
from scipy.signal import correlate
from scipy.interpolate import interp1d

def fit2insitu(t, b, bx, by, bz, 
    x=u.au.to(u.m, 1.0), 
    y=u.au.to(u.m, 0.0), 
    z=u.au.to(u.m, 0.0),
    period=3.0*24.0*3600.0,
    step=3600.0,
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
        [0.1, 0.3]
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
    polarity=-1.0,
    chirality=1.0):

    # resample real data with fixed step
    # get modeled data
    # correlation -> shift
    # residual

    # in progress

    t_ = np.arange(0.0, t[-1]-t[0]+step, step)
    f = interp1d(t)
    tt = np.arange(0.0, period+step, step)

    def F(x):
        
        evo = Evolution()
        n = 0
        
        if isinstance(latitude, np.ndarray):
            evo.latitude = lambda t: polyval(x[n:n+latitude.shape[0]], t)
            n += latitude.shape[0]
        else:
            evo.latitude = lambda t: latitude

        if isinstance(longitude, np.ndarray):
            evo.longitude = lambda t: polyval(x[n:n+longitude.shape[0]], t)
            n += longitude.shape[0]
        else:
            evo.longitude = lambda t: longitude

        if isinstance(toroidal_height, np.ndarray):
            evo.toroidal_height = lambda t: \
                polyval(x[n:n+toroidal_height.shape[0]], t)
            n += toroidal_height.shape[0]
        else:
            evo.toroidal_height = lambda t: toroidal_height

        if isinstance(poloidal_height, np.ndarray):
            evo.poloidal_height = lambda t: \
                polyval(x[n:n+poloidal_height.shape[0]], t)
            n += poloidal_height.shape[0]
        else:
            evo.poloidal_height = lambda t: poloidal_height

        if isinstance(half_width, np.ndarray):
            evo.half_width = lambda t: polyval(x[n:n+half_width.shape[0]], t)
            n += half_width.shape[0]
        else:
            evo.half_width = lambda t: half_width

        if isinstance(tilt, np.ndarray):
            evo.tilt = lambda t: polyval(x[n:n+tilt.shape[0]], t)
            n += tilt.shape[0]
        else:
            evo.tilt = lambda t: tilt

        if isinstance(flattening, np.ndarray):
            evo.flattening = lambda t: polyval(x[n:n+flattening.shape[0]], t)
            n += flattening.shape[0]
        else:
            evo.flattening = lambda t: flattening

        if isinstance(pancaking, np.ndarray):
            evo.pancaking = lambda t: polyval(x[n:n+pancaking.shape[0]], t)
            n += pancaking.shape[0]
        else:
            evo.pancaking = lambda t: pancaking

        if isinstance(skew, np.ndarray):
            evo.skew = lambda t: polyval(x[n:n+skew.shape[0]], t)
            n += skew.shape[0]
        else:
            evo.skew = lambda t: skew

        if isinstance(twist, np.ndarray):
            evo.twist = lambda t: polyval(x[n:n+twist.shape[0]], t)
            n += twist.shape[0]
        else:
            evo.twist = lambda t: twist

        if isinstance(flux, np.ndarray):
            evo.flux = lambda t: polyval(x[n:n+flux.shape[0]], t)
            n += flux.shape[0]
        else:
            evo.flux = lambda t: flux

        if isinstance(sigma, np.ndarray):
            evo.sigma = lambda t: polyval(x[n:n+sigma.shape[0]], t)
            n += sigma.shape[0]
        else:
            evo.sigma = lambda t: sigma

        evo.polarity = polarity
        evo.chirality = chirality

        b = evo.insitu(tt, x, y, z)

        if np.any(b != 0.0):

            np.argmax(correlate(
                np.stack([b[:,1], b[:,2], b[:,3]], axis=1),
                np.stack([bx, by, bz], axis=1)
            ))

            db = np.mean([euclidean(
                [b[i,1], b[i,2], b[i,3]],
                [b[i,1], b[i,2], b[i,3]]
            ) for i in range(tt.size)])
        else:
            return np.inf
