
from ai.fri3d.optimize import fit2insitu
import ai.cdas as cdas
import numpy as np
from datetime import datetime
from astropy import units as u
import time
from scipy.io import readsav
from matplotlib import pyplot as plt
from ai.shared.data import getVEX, getSTA, getMES
from ai.fri3d import Evolution

u.nT = u.def_unit('nT', 1e-9*u.T)

def fit2mes():
    t, b, p = getMES(
        datetime(2013, 1, 7, 18, 17),
        datetime(2013, 1, 8, 14, 45)
    )

    r = np.sqrt(p[:,0]**2+p[:,1]**2+p[:,2]**2)*u.m.to(u.au)
    print(r.min(), r.max())
    print(u.m.to(u.au, p[:,2].min()), u.m.to(u.au, p[:,2].max()))
    phi = u.rad.to(u.deg, np.arctan2(p[:,1], p[:,0]))
    print(phi.min(), phi.max())

    # fig = plt.figure()
    # plt.plot(t, b)
    # plt.show()

    fit2insitu(t, b,
        # latitude=u.deg.to(u.rad, -5.0),
        latitude=np.array([
            u.deg.to(u.rad, [-15.0, 0.0])
        ]),
        longitude=u.deg.to(u.rad, 125.0),
        # longitude=np.array([
        #     u.deg.to(u.rad, [120.0, 150.0])
        # ]), 
        toroidal_height=np.array([
            u.Unit('km/s').to(u.Unit('m/s'), [485.0, 485.0]), 
            u.au.to(u.m, [0.2, 0.2])
        ]),
        poloidal_height=np.array([
            u.Unit('km/s').to(u.Unit('m/s'), [0.0, 100.0]), 
            u.au.to(u.m, [0.01, 0.2])
        ]), 
        half_width=u.deg.to(u.rad, 43.0), 
        tilt=u.deg.to(u.rad, 44.0),
        # tilt=np.array([
        #     u.deg.to(u.rad, [40.0, 60.0])
        # ]), 
        flattening=np.array([
            [0.4, 0.8]
        ]), 
        pancaking=u.deg.to(u.rad, 18.0), 
        skew=u.deg.to(u.rad, 0.0),
        twist=np.array([
            [1.0, 3.0]
        ]), 
        flux=np.array([
            [1e14, 1e15]
        ]),
        sigma=2.0,
        # sigma=np.array([
        #     [1.0, 3.0]
        # ]),
        polarity=1.0,
        chirality=1.0,
        max_pre_time=1.0*3600.0,
        max_post_time=5.0*3600.0,
        x=np.mean(p[:,0]),
        y=np.mean(p[:,1]),
        z=np.mean(p[:,2]),
        verbose=True,
        timestamp_mask=lambda t: np.logical_or.reduce([
            np.logical_and(
                t >= time.mktime(datetime(2013, 1, 7, 18, 17).timetuple()),
                t <= time.mktime(datetime(2013, 1, 7, 22, 52).timetuple())
            ),
            np.logical_and(
                t >= time.mktime(datetime(2013, 1, 8, 1, 28).timetuple()),
                t <= time.mktime(datetime(2013, 1, 8, 6, 52).timetuple())
            ),
            np.logical_and(
                t >= time.mktime(datetime(2013, 1, 8, 9, 37).timetuple()),
                t <= time.mktime(datetime(2013, 1, 8, 14, 45).timetuple())
            )
        ])
    )

# latitude: -10.0
# longitude: 125.0
# expansion speed: 46.0
# poloidal_height: 0.1
# tilt: 74.0
# flattening: 0.7
# twist: 1.21
# flux: 6.31e14
# sigma: 2.81

# [ -1.74476369e-01   2.17916665e+00   4.50000000e+05   1.04718509e+11
#    4.59477708e+04   1.44297164e+10   1.29043899e+00   6.98828388e-01
#    1.20649011e+00   6.30915356e+14   2.80640914e+00]

def fit2vex():
    t, b, p = getVEX(
        datetime(2013, 1, 8, 18),
        datetime(2013, 1, 9, 16)
    )

    r = np.sqrt(p[:,0]**2+p[:,1]**2+p[:,2]**2)
    # print(r.min(), r.max())

    ts = time.mktime(datetime(2013,1,6,10,39).timetuple())
    hs = u.R_sun.to(u.m, 12.5)
    rs = u.R_sun.to(u.m, 3.5)
    tm1 = time.mktime(datetime(2013,1,7,18,17).timetuple())
    tm2 = time.mktime(datetime(2013,1,8,16).timetuple())
    tv1 = time.mktime(datetime(2013,1,8,18).timetuple())
    tv2 = time.mktime(datetime(2013,1,9,16).timetuple())
    hm = u.au.to(u.m, 0.46)
    hv = r.mean()

    ta1 = time.mktime(datetime(2013,1,9,14).timetuple())
    ta2 = time.mktime(datetime(2013,1,10,16).timetuple())
    ha = u.au.to(u.m, 0.96)

    vm1 = (hm-(hs+rs))/(tm1-ts)
    vv1 = (hv-(hs+rs))/(tv1-ts)
    vm1 = 497000.0
    vv1 = 497000.0
    vm2 = (hm-(hs-rs))/(tm2-ts)
    vv2 = (hv-(hs-rs))/(tv2-ts)

    p1 = np.polyfit([tm1, tv1], [vm1, vv1], 1)
    p2 = np.polyfit([tm2, tv2], [vm2, vv2], 1)
    
    p3 = (p1+p2)/2.0
    p4 = (p1-p2)/2.0

    x01 = hm-p1[1]*tm1-p1[0]/2.0*tm1**2
    x02 = hm-p2[1]*tm2-p2[0]/2.0*tm2**2
    x03 = (x01+x02)/2.0
    x04 = (x01-x02)/2.0

    p_prp = np.array([p3[0]/2.0, p3[1], x03])
    p_exp = np.array([p4[0]/2.0, p4[1], x04])

    print(
        np.polyval(p3, tm1)/1e3,
        np.polyval(p4, tm1)/1e3,
        u.m.to(u.au, np.polyval(p_prp, tm1)),
        u.m.to(u.au, np.polyval(p_exp, tm1)),
    )

    print(p_prp, p_exp)

    # tt = np.linspace()
    # plt.plot()

    print('Speed at STA = ', np.polyval(p3, ta1), np.polyval(p3, ta2))
    print('Propagation acceleration = ', p3[0], 'm/s^2')
    print('Expansion acceleration = ', p4[0], 'm/s^2')

    fit2insitu(t, b,
        latitude=np.array([
            u.deg.to(u.rad, [-15.0, 0.0])
        ]),
        # longitude=u.deg.to(u.rad, 125.0),
        longitude=np.array([
            u.deg.to(u.rad, [120.0, 130.0])
        ]), 
        toroidal_height=np.array([
            [1.19777671e-01*0.95, 1.19777671e-01*1.05],
            u.Unit('km/s').to(u.Unit('m/s'), [392.625485341*0.95, 392.625485341*1.05]), 
            u.au.to(u.m, [0.379786123894*0.95, 0.379786123894*1.05])
        ]),
        poloidal_height=np.array([
            [-1.19777671e-01*1.05, -1.19777671e-01*0.95],
            u.Unit('km/s').to(u.Unit('m/s'), [104.374514659*0.95, 104.374514659*1.05]), 
            u.au.to(u.m, [0.0802138758249*0.95, 0.0802138758249*1.05])
        ]), 
        half_width=u.deg.to(u.rad, 43.0), 
        # tilt=u.deg.to(u.rad, 44.0),
        tilt=np.array([
            u.deg.to(u.rad, [40.0, 80.0])
        ]), 
        flattening=np.array([
            [0.6, 0.8]
        ]), 
        pancaking=u.deg.to(u.rad, 18.0), 
        skew=u.deg.to(u.rad, 0.0),
        twist=np.array([
            [1.0, 3.0]
        ]), 
        flux=np.array([
            [1e14, 1e15]
        ]),
        sigma=2.0,
        # sigma=np.array([
        #     [1.5, 3.0]
        # ]),
        polarity=1.0,
        chirality=1.0,
        max_pre_time=2.0*3600.0,
        max_post_time=2.0*3600.0,
        x=np.mean(p[:,0]),
        y=np.mean(p[:,1]),
        z=np.mean(p[:,2]),
        verbose=True,
        timestamp_mask=lambda t: np.logical_or(
            t <= time.mktime(datetime(2013, 1, 9, 4, 10).timetuple()),
            t >= time.mktime(datetime(2013, 1, 9, 7, 17).timetuple())
        )
    )

def fit2sta():
    t, b, p = getSTA(
        datetime(2013, 1, 9, 14),
        datetime(2013, 1, 10, 16)
    )

    r = np.sqrt(p[:,0]**2+p[:,1]**2+p[:,2]**2)
    # print(r.min(), r.max())

    ts = time.mktime(datetime(2013,1,6,10,39).timetuple())
    hs = u.R_sun.to(u.m, 12.5)
    rs = u.R_sun.to(u.m, 3.5)
    tm1 = time.mktime(datetime(2013,1,7,18,17).timetuple())
    tm2 = time.mktime(datetime(2013,1,8,16).timetuple())
    tv1 = time.mktime(datetime(2013,1,8,18).timetuple())
    tv2 = time.mktime(datetime(2013,1,9,16).timetuple())
    hm = u.au.to(u.m, 0.46)
    hv = u.au.to(u.m, 0.72)
    # r.mean()

    ta1 = time.mktime(datetime(2013,1,9,14).timetuple())
    ta2 = time.mktime(datetime(2013,1,10,16).timetuple())
    ha = u.au.to(u.m, 0.96)

    vm1 = (hm-(hs+rs))/(tm1-ts)
    vv1 = (hv-(hs+rs))/(tv1-ts)
    vm1 = 497000.0
    vv1 = 497000.0
    vm2 = (hm-(hs-rs))/(tm2-ts)
    vv2 = (hv-(hs-rs))/(tv2-ts)

    p1 = np.polyfit([tm1, tv1], [vm1, vv1], 1)
    p2 = np.polyfit([tm2, tv2], [vm2, vv2], 1)
    
    p3 = (p1+p2)/2.0
    p4 = (p1-p2)/2.0

    x01 = hm-p1[1]*tm1-p1[0]/2.0*tm1**2
    x02 = hm-p2[1]*tm2-p2[0]/2.0*tm2**2
    x03 = (x01+x02)/2.0
    x04 = (x01-x02)/2.0

    p_prp = np.array([p3[0]/2.0, p3[1], x03])
    p_exp = np.array([p4[0]/2.0, p4[1], x04])

    print(
        np.polyval(p3, tv1)/1e3,
        np.polyval(p4, tv1)/1e3,
        u.m.to(u.au, np.polyval(p_prp, tv1)),
        u.m.to(u.au, np.polyval(p_exp, tv1)),
    )

    print(p_prp, p_exp)

    # tt = np.linspace()
    # plt.plot()

    print('Speed at STA = ', np.polyval(p3, ta1), np.polyval(p3, ta2))
    print('Propagation acceleration = ', p3[0], 'm/s^2')
    print('Expansion acceleration = ', p4[0], 'm/s^2')

    fit2insitu(t, b,
        latitude=np.array([
            u.deg.to(u.rad, [-15.0, 0.0])
        ]),
        # longitude=u.deg.to(u.rad, [120.0, 130.0]),
        longitude=np.array([
            u.deg.to(u.rad, [120.0, 130.0])
        ]), 
        toroidal_height=np.array([
            [1.19777671e-01*0.95, 1.19777671e-01*1.05],
            u.Unit('km/s').to(u.Unit('m/s'), [418.33189022*0.95, 418.33189022*1.05]), 
            u.au.to(u.m, [0.609832340241*0.95, 0.609832340241*1.05])
        ]),
        poloidal_height=np.array([
            [-1.19777671e-01*1.05, -1.19777671e-01*0.95],
            u.Unit('km/s').to(u.Unit('m/s'), [78.6681097801*0.95, 78.6681097801*1.05]), 
            u.au.to(u.m, [0.133820495508*0.95, 0.133820495508*1.05])
        ]), 
        half_width=u.deg.to(u.rad, 43.0), 
        tilt=np.array([
            u.deg.to(u.rad, [40.0, 80.0])
        ]), 
        flattening=np.array([
            [0.5, 0.8]
        ]), 
        pancaking=u.deg.to(u.rad, 18.0), 
        skew=u.deg.to(u.rad, 0.0),
        twist=np.array([
            [1.0, 3.0]
        ]), 
        flux=np.array([
            [1e14, 1e15]
        ]),
        sigma=2.0,
        # sigma=np.array([
        #     [1.0, 3.0]
        # ]),
        polarity=1.0,
        chirality=1.0,
        max_pre_time=1.0*3600.0,
        max_post_time=2.0*3600.0,
        x=np.mean(p[:,0]),
        y=np.mean(p[:,1]),
        z=np.mean(p[:,2]),
        verbose=True,
        timestamp_mask=lambda t: np.logical_or(
            t <= time.mktime(datetime(2013, 1, 10, 2).timetuple()),
            t >= time.mktime(datetime(2013, 1, 10, 7).timetuple())
        )
    )

# longitude
# 123
# tilt
# 44

# longitude
# 2.37035386e+00 136
# tilt
# 9.46053416e-01 54
# twist
# 1.58728098e+00   
# flux
# 5.47900789e+14   

# longitude
# 2.17625421e+00 125
# tilt
# 1.28848680e+00 74  
# twist
# 1.22226705e+00   
# flux
# 5.18637309e+14   

# 7.2095767742e-09 1357538400.0 [ -2.61642625e-01   4.53791928e+05   5.98391483e+10   5.38817215e+04
#    4.22055661e+09   8.75565501e-01   7.59345650e-01   2.06359481e+00
#    1.64030356e+14]

# 1357675200.0 [ -2.10447023e-01   4.50000000e+05   1.04718509e+11   6.58158031e+04
#    1.42248543e+10   1.31772088e+00   5.79993848e-01   1.30328507e+00
#    3.47615568e+14]


def forecast():

    t, b, p = getSTA(
        datetime(2013, 1, 9, 14),
        datetime(2013, 1, 10, 16)
    )
    
    evo = Evolution()
    evo.latitude = lambda t: np.polyval(
        np.array([-2.31726844e-01]), 
        t
    )
    evo.longitude = lambda t: np.polyval(
        np.array([u.deg.to(u.rad, 125.0)]),
        t
    )
    evo.toroidal_height = lambda t: np.polyval(
        np.array([1.47024208e-01, 4.32516259e+05, 5.31779805e+10]),
        t
    )
    evo.poloidal_height = lambda t: np.polyval(
        np.array([-3.72232493e-01, 1.13855334e+05, 1.08051515e+10]),
        t
    )
    evo.half_width = lambda t: np.polyval(
        np.array([u.deg.to(u.rad, 43.0)]),
        t
    )
    evo.tilt = lambda t: np.polyval(
        np.array([1.14736439e+00]),
        t
    )
    evo.flattening = lambda t: np.polyval(
        np.array([7.31460827e-01]),
        t
    )
    evo.pancaking = lambda t: np.polyval(
        np.array([u.deg.to(u.rad, 18.0)]),
        t
    )
    evo.skew = lambda t: np.polyval(
        np.array([u.deg.to(u.rad, 0.0)]),
        t
    )
    evo.twist = lambda t: np.polyval(
        np.array([2.66940993e+00]),
        t
    )
    evo.flux = lambda t: np.polyval(
        np.array([2.20317182e+14]),
        t
    )
    evo.sigma = lambda t: np.polyval(
        np.array([2.0]),
        t
    )
    tm = np.arange(0.0, 4.0*24.0*3600.0, 200)
    bm = evo.insitu(
        tm, 
        np.mean(p[:,0]),
        np.mean(p[:,1]),
        np.mean(p[:,2])
    )
    d = np.array([datetime.fromtimestamp(x) for x in tm+1357545600.0])
    fig = plt.figure()
    plt.plot(t, np.sqrt(b[:,0]**2+b[:,1]**2+b[:,2]**2), 'k')
    plt.plot(t, b[:,0], 'r')
    plt.plot(t, b[:,1], 'g')
    plt.plot(t, b[:,2], 'b')
    plt.plot(d, np.sqrt(bm[:,0]**2+bm[:,1]**2+bm[:,2]**2), '--k')
    plt.plot(d, bm[:,0], '--r')
    plt.plot(d, bm[:,1], '--g')
    plt.plot(d, bm[:,2], '--b')
    plt.show()

# fit2mes()
# fit2vex()
fit2sta()
# forecast()
