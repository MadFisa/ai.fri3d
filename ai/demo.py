
import time
from datetime import datetime, timedelta

import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.path as mpath
import matplotlib.colors as colors
from matplotlib.colorbar import ColorbarBase

import ai.cdas as cdas

from importlib import reload

from ai.fri3d import FRi3D

AU_KM = 149597870.7
RS_KM = 6.957e5
AU_RS = AU_KM/RS_KM
RS_AU = RS_KM/AU_KM

def test_shell(
    latitude=0.0, 
    longitude=0.0, 
    toroidal_height=1.0, 
    poloidal_height=0.2, 
    half_width=np.pi/180.0*45.0, 
    tilt=np.pi/180.0*0.0, 
    flattening=0.5, 
    pancaking=np.pi/180.0*30.0, 
    skew=np.pi/180.0*0.0):
    
    fr = FRi3D(
        latitude=latitude, 
        longitude=longitude, 
        toroidal_height=toroidal_height, 
        poloidal_height=poloidal_height, 
        half_width=half_width, 
        tilt=tilt, 
        flattening=flattening, 
        pancaking=pancaking, 
        skew=skew
    )
    fr.init()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', adjustable='box', aspect=1.0)
    x, y, z = fr.shell()
    ax.plot_wireframe(x, y, z, alpha=0.1)

    ax.set_xlabel('X [AU]')
    ax.set_ylabel('Y [AU]')
    ax.set_zlabel('Z [AU]]')

    fig = plt.figure()
    x, y, z = fr.shell()
    plt.plot(y, z, '.r')

    plt.show()

def test_mf(
    latitude=0.0, 
    longitude=0.0, 
    toroidal_height=1.0, 
    poloidal_height=0.2, 
    half_width=np.pi/180.0*45.0, 
    tilt=np.pi/180.0*0.0, 
    flattening=0.5, 
    pancaking=np.pi/180.0*30.0, 
    skew=np.pi/180.0*0.0, 
    twist=2.0, 
    flux=5e14,
    sigma=1.05,
    polarity=1.0,
    chirality=1.0):
    
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
        twist=twist, 
        flux=flux,
        sigma=1.05,
        polarity=polarity,
        chirality=chirality
    )
    fr.init()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', adjustable='box', aspect=1.0)
    x, y, z = fr.shell()
    ax.plot_wireframe(x, y, z, alpha=0.1)

    _, _, _, b = fr.line(1.0, 0.0, s=0.5)
    bmin = b
    _, _, _, b = fr.line(0.0, 0.0, s=0.9)
    bmax = b

    for i in range(50):
        r = np.random.uniform(0.0, 1.0)
        phi = np.random.uniform(0.0, np.pi*2.0)
        x, y, z, b = fr.line(r, phi, s=np.linspace(0.1, 0.9, 200))
        points = np.array([x, y, z]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        c = (b-bmin)/(bmax-bmin)
        lc = Line3DCollection(
            segments, 
            colors=plt.cm.magma(c)
        )
        ax.add_collection3d(lc)

    ax.set_xlabel('X [AU]')
    ax.set_ylabel('Y [AU]')
    ax.set_zlabel('Z [AU]]')

    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.get_cmap('magma'), 
        norm=plt.Normalize(vmin=bmin, vmax=bmax)
    )
    # fake up the array of the scalar mappable. Urgh...
    sm._A = []
    cb = plt.colorbar(sm)
    cb.set_label('B [nT]')

    plt.show()

def test_mf_cs(
    latitude=0.0, 
    longitude=0.0, 
    toroidal_height=1.0, 
    poloidal_height=0.2, 
    half_width=np.pi/180.0*45.0, 
    tilt=np.pi/180.0*0.0, 
    flattening=0.5, 
    pancaking=np.pi/180.0*30.0, 
    skew=np.pi/180.0*0.0, 
    twist=2.0, 
    flux=5e14,
    sigma=1.05):
    
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
        twist=twist, 
        flux=flux,
        sigma=sigma
    )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', adjustable='box', aspect=1.0)
    x, y, z = fr.shell()
    ax.plot_wireframe(x, y, z, alpha=0.1)

    _, _, _, b = fr.line(1.0, 0.0, s=0.5)
    bmin = b
    _, _, _, b = fr.line(0.0, 0.0, s=0.49)
    bmax = b

    for i in range(500):
        r = np.random.uniform(0.0, 1.0)
        phi = np.random.uniform(0.0, np.pi*2.0)
        x, y, z, b = fr.line(r, phi, s=np.linspace(0.49, 0.51, 10))
        points = np.array([x, y, z]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        c = (b-bmin)/(bmax-bmin)
        lc = Line3DCollection(
            segments, 
            colors=plt.cm.magma(c)
        )
        ax.add_collection3d(lc)

    ax.set_xlabel('X [AU]')
    ax.set_ylabel('Y [AU]')
    ax.set_zlabel('Z [AU]]')

    ax.view_init(elev=0.0, azim=-90.0)

    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.get_cmap('magma'), 
        norm=plt.Normalize(vmin=bmin, vmax=bmax)
    )
    # fake up the array of the scalar mappable. Urgh...
    sm._A = []
    cb = plt.colorbar(sm)
    cb.set_label('B [nT]')

    plt.show()

def test_insitu_static(
    latitude=0.0, 
    longitude=0.0, 
    toroidal_height=1.0, 
    poloidal_height=0.2, 
    half_width=np.pi/180.0*45.0, 
    tilt=np.pi/180.0*0.0, 
    flattening=0.5, 
    pancaking=np.pi/180.0*30.0, 
    skew=np.pi/180.0*0.0, 
    twist=2.0, 
    flux=5e14,
    sigma=1.05,
    polarity=1.0,
    chirality=1.0,
    x=np.linspace(1.2, 0.8, 100),
    y=np.zeros(100),
    z=np.zeros(100),
    r=None,
    theta=None,
    phi=None):

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
        twist=twist, 
        flux=flux,
        sigma=sigma,
        polarity=polarity,
        chirality=chirality
    )
    fr.init()

    if r is not None and theta is not None and phi is not None:
        x, y, z = cs.sp2cart(r, theta, phi)

    b = fr.cut1d(x, y, z)
    
    fig = plt.figure()
    plt.plot(b[:,0], 'k', linewidth=2, label='B')
    plt.plot(b[:,1], 'r', linewidth=2, label='Bx')
    plt.plot(b[:,2], 'g', linewidth=2, label='By')
    plt.plot(b[:,3], 'b', linewidth=2, label='Bz')
    plt.xlabel('time [arb. units]')
    plt.ylabel('B [nT]')
    plt.legend()

    plt.show()

# [ 0.06166189 -0.16725744  0.3817882   0.12310777  4.83976848  2.23878726]



def test_insitu_evo(
    latitude=0.0, 
    longitude=0.0, 
    toroidal_height=0.8,
    poloidal_height=0.2,
    half_width=np.pi/180.0*45.0, 
    tilt=np.pi/180.0*0.0, 
    flattening=0.5, 
    pancaking=np.pi/180.0*30.0, 
    skew=np.pi/180.0*0.0, 
    twist=2.0, 
    flux=5e14,
    sigma=1.05,
    polarity=1.0,
    chirality=1.0,
    x=1.0,
    y=0.0,
    z=0.0,
    r=None,
    theta=None,
    phi=None):

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
        twist=twist, 
        flux=flux,
        sigma=sigma,
        polarity=polarity,
        chirality=chirality
    )
    fr.init()

    if r is not None and theta is not None and phi is not None:
        r = np.array(r, copy=False, ndmin=1)
        theta = np.array(theta, copy=False, ndmin=1)
        phi = np.array(phi, copy=False, ndmin=1)
        x, y, z = cs.sp2cart(r, theta, phi)
        x = x[0]
        y = y[0]
        z = z[0]

    b = fr.evocut1d(x, y, z, 
        toroidal_height=toroidal_height
    )
    
    fig = plt.figure()
    plt.plot(b[:,0], 'k', linewidth=2, label='B')
    plt.plot(b[:,1], 'r', linewidth=2, label='Bx')
    plt.plot(b[:,2], 'g', linewidth=2, label='By')
    plt.plot(b[:,3], 'b', linewidth=2, label='Bz')
    plt.xlabel('time [arb. units]')
    plt.ylabel('B [nT]')
    plt.legend()

    plt.show()

def test_fit2insitu():
    cdas.set_cache(True, 'data')
    # cdas.set_cache(False)
    data = cdas.get_data(
        'sp_phys', 
        'STA_L1_MAG_RTN', 
        datetime(2010, 12, 15, 11), 
        datetime(2010, 12, 16, 3), 
        ['BFIELD'],
        cdf=True
    )
    # for key, _ in data.items() :
    #     print(key)
    t = data['Epoch']
    b = data['BFIELD'][:,3]
    bx = data['BFIELD'][:,0]
    by = data['BFIELD'][:,1]
    bz = data['BFIELD'][:,2]
    # b = b[0::1800]
    # bx = bx[0::1800]
    # by = by[0::1800]
    # bz = bz[0::1800]
    # plt.plot(b, 'k')
    # plt.plot(bx, 'r')
    # plt.plot(by, 'g')
    # plt.plot(bz, 'b')
    # print(b.size, bx.size, by.size, bz.size)
    # plt.show()
    fr = FRi3D()
    fr.fit2insitu(t, b, bx, by, bz)

def orthogonal_proj(zfront, zback):
    a = (zfront+zback)/(zfront-zback)
    b = -2*(zfront*zback)/(zfront-zback)
    return np.array([[1,0,0,0],
                     [0,1,0,0],
                     [0,0,a,b],
                     [0,0,-0.0001,zback]])
proj3d.persp_transformation = orthogonal_proj

# test_shell(
#     latitude=0.16631833, 
#     longitude=-0.27972665, 
#     toroidal_height=0.93,
#     poloidal_height=0.07,
#     half_width=0.69614774, 
#     tilt=-0.13859975, 
#     flattening=0.5, 
#     pancaking=np.pi/180.0*20.0, 
#     skew=np.pi/180.0*0.0, 
# )
# test_insitu_evo(
#     latitude=0.13865431, 
#     longitude=-0.30173361, 
#     toroidal_height=0.8,
#     poloidal_height=0.2,
#     half_width=0.67119132, 
#     tilt=-0.06193924, 
#     flattening=0.5, 
#     pancaking=np.pi/180.0*30.0, 
#     skew=np.pi/180.0*0.0, 
#     twist=1.76410176, 
#     flux=1e14,
#     sigma=1.66959326,
#     polarity=-1.0,
#     chirality=1.0
# )

# 13.21207578 -12.21254998   0.2067406   42.82340851 -13.37854642 0.65363542  24.98744459   2.92498132   1.94266424
# 417
# 10.23566752 -14.49946746   0.14513609  43.14496717  -5.47717313 0.66339224  22.85197466   2.7857445    1.59715411
# 423
# 12.03872107 -14.14146187   0.19314003  41.11990838  -8.48893417 0.52535873  24.0747819    2.80713025   1.73232396
# 400
# 11.46164081 -17.2274398    0.20056907  43.12149505  -8.4092668 0.43678998  24.72850232   2.7988322    1.57838507
# 391
# 11.2748968  -13.99914565   0.19710877  41.53145984  -9.59675635 0.54515974  24.68697312   2.9935804    1.97637335
# 370
# 11.70819633 -15.45342549   0.19479493  40.12490033  -8.26235337 0.4472716   24.73453329   2.98663464   1.75655723
# 372

# [ 11.87781107 -13.77630955   0.19177268  42.13288424 -10.10129675
#    0.56317498  24.91613772   2.98525034   2.4255184 ]
# 370.326755495

# [ 11.52431769 -14.37371846   0.19272505  41.6838522   -9.72321289
#    0.51797323  24.96792237   2.99346144   2.05108272]
# 373.068438143

# 11.10980336 -14.86957811   0.19207429  40.89738226  -9.76996998
#    0.47762824  24.61793549   2.9850835    2.3893733 ]
# 368.189962273

# [ 11.70898979 -14.19051201   0.19047947  41.97595857  -9.4041287
#    0.54897585  24.84332011   2.98538496   2.08590715]
# 367.742772724

# [ 11.70898979 -14.20849647   0.19250725  41.97595857 -10.1126414
#    0.53258256  24.84332011   2.98538496   2.37071168]
# 367.013935122

# [ 12.34797893 -13.98803418   0.19131527  41.86139823  -9.5992345
#    0.55117503  24.6325322    2.97592592   2.11557955]
# 368.07955694

# [ 11.95608249 -13.30593681   0.18922012  40.17245685  -9.46492146
#    0.55524235  24.63690029   2.99717816   2.09791599]
# 366.081938839

# [ 13.25243161 -19.86684401   0.1771352   42.13350893  -7.1148385
#    0.48887183  23.84874631   3.89368811   2.15948889]
# 377.857068442

# [ 15.16886093 -23.87287276   0.20170974  45.84779231  -5.74061498
#    0.48882515  28.01419494   3.82644752   2.5163953 ]
# 350.246134526

# [ 14.03175922 -23.36429757   0.24017725  46.65345715  -6.31398029
#    0.40376073  29.66819131   3.91366104   2.73302248]
# 343.187821475

# fixed pancaking 20

# [  6.04490191 -20.62488491   0.22926915  40.13784242  -1.34878597
#    0.50522434   3.93828161   1.75515485]
# 470.809662897

# [  8.80907578 -20.31396903   0.21011882  39.94823011  -6.78062505
#    0.54911747   4.98100459   2.08810886]
# 442.913531109

# [  4.56130681 -18.89423998   0.20765019  40.38090972   3.39496761
#    0.57245453   4.93964511   1.01345514]
# 383.520005081

# [  1.61717997 -19.12401213   0.13887307   9.28039994   0.50782528
#    3.80327657]

# [ -0.05027295 -22.19693456   0.22628411  11.83874172   0.40029852
#    6.77471195]
# 919.8964026301926

# [  3.40270972 -23.28213318   0.22179218   4.76325834   0.47913118
#    5.99609854]
# 684.7192019598913

# [ 12.44858384 -18.73443069   0.16308063  -5.63368047   0.55211448
#    4.9343563 ]
# 716.9429785908316

# [  8.25212754 -19.71398737   0.19769002  -6.23332826   0.44521114
#    4.9367149 ]
# 675.909969356875

# [  4.8026541  -22.65214433   0.19769002   2.43959311   0.47206949
#    6.065338  ]
# 653.8003294011528

# [  6.05373672 -21.05807102   0.23147909   2.38135057   0.54898635
#    6.2482144 ]
# 630.9223156164162


# [  5.67714517 -14.38368045   0.13550604  -1.15569243   0.57904958
#    6.07544706   2.29977037   1.12553558]
# 3.12532058623

# increased resolution to 0.02 AU

# [ 10.009215   -16.07887594   0.15955503  -7.42553826   0.53295159
#    6.43533042   1.86090296   1.26818314]
# 3.60360658674

# [  7.45525566 -17.24349792   0.17858595  -5.06079521   0.48072999
#    4.88595661   1.33678636   1.23475906]
# 3.4039173696

# [  9.13935855 -18.77690073   0.10548114  -5.08018097   0.43273579
#    6.42297451   1.9372092    1.16858023]
# 2.94157206834

# [  7.91624094 -18.57420865   0.10099705   0.03374686   0.49177376
#    6.05033308   2.24051718   1.17247319]
# 2.65376418521

# [  8.13628301e+00  -1.75189394e+01   1.06670528e-01   9.08934282e-03
#    4.99480528e-01   6.64944091e+00   2.52432559e+00   1.28426189e+00]
# 2.64756760732

# [  8.35804958 -20.33325833   0.12743121   1.08367922   0.49003112
#    6.76876898   2.26008429   1.23151931]
# 2.58214282798

# [  8.14401601 -19.5715945    0.14182336   0.37987035   0.58457917
#    6.53600575   2.36398786   1.22504594]
# 2.54982939034

# [  8.54355628 -20.39145993   0.11084887   2.16074957   0.50537504
#    6.77391367   2.19169087   1.20288016]
# 2.49470631039

def test_article(
    latitude=np.pi/180.0*8.54355628, 
    longitude=-np.pi/180.0*20.39145993, 
    toroidal_height=0.7,
    poloidal_height=0.11084887,
    half_width=np.pi/180.0*40, 
    tilt=np.pi/180.0*2.16074957, 
    flattening=0.50537504, 
    pancaking=np.pi/180.0*20.0, 
    skew=np.pi/180.0*0.0, 
    twist=6.77391367, 
    flux=1e14,
    sigma=2.19169087,
    polarity=-1.0,
    chirality=1.0,
    ratio=1.20288016,
    x=1.0,
    y=0.0,
    z=0.0):

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
        twist=twist, 
        flux=flux,
        sigma=sigma,
        polarity=polarity,
        chirality=chirality
    )
    fr.init()
    

    t0 = datetime(2010, 12, 15, 11)
    t1 = datetime(2010, 12, 16, 3)
    dt = timedelta(minutes=30)
    
    cdas.set_cache(True, 'data')
    data = cdas.get_data(
        'sp_phys', 
        'STA_L1_MAG_RTN', 
        t0, 
        t1,
        ['BFIELD'],
        cdf=True
    )
    t = data['Epoch']
    b = data['BFIELD'][:,3]
    bx = data['BFIELD'][:,0]
    by = data['BFIELD'][:,1]
    bz = data['BFIELD'][:,2]

    n = 300
    t = np.array([time.mktime(x.timetuple()) for x in t])
    t0 = t[0]+(t[-1]-t[0])*np.linspace(0.0, 1.0, n)
    b0 = np.interp(t0, t, b)
    bx0 = np.interp(t0, t, bx)
    by0 = np.interp(t0, t, by)
    bz0 = np.interp(t0, t, bz)

    b0_mean = np.mean(b0)

    b_ = fr.evocut1d(x, y, z, 
        toroidal_height=toroidal_height
    )

    t = t0[0]+(t0[-1]-t0[0])*ratio*np.linspace(0.0, 1.0, b_.shape[0])
    b = b_[:,0]
    bx = b_[:,1]
    by = b_[:,2]
    bz = b_[:,3]
    t1 = t0
    b1 = np.interp(t1, t, b)
    bx1 = np.interp(t1, t, bx)
    by1 = np.interp(t1, t, by)
    bz1 = np.interp(t1, t, bz)

    b1_mean = np.mean(b1)

    coeff = b0_mean/b1_mean
    b1 *= coeff
    bx1 *= coeff
    by1 *= coeff
    bz1 *= coeff

    t0 = np.array([datetime.fromtimestamp(x) for x in t0])
    t1 = np.array([datetime.fromtimestamp(x) for x in t1])

    fig = plt.figure()
    plt.plot(t1, b1, '--k', linewidth=2, label='B')
    plt.plot(t0, b0, 'k', linewidth=2, label='B')
    plt.plot(t1, bx1, '--r', linewidth=2, label='Bx')
    plt.plot(t0, bx0, 'r', linewidth=2, label='Bx')
    plt.plot(t1, by1, '--g', linewidth=2, label='By')
    plt.plot(t0, by0, 'g', linewidth=2, label='By')
    plt.plot(t1, bz1, '--b', linewidth=2, label='Bz')
    plt.plot(t0, bz0, 'b', linewidth=2, label='Bz')
    plt.xlabel('time [arb. units]')
    plt.ylabel('B [nT]')
    # plt.legend()

    plt.show()

def test_remote(
    latitude=-np.pi/180.0*7.0, 
    longitude=-np.pi/180.0*22.0, 
    toroidal_height=13.5/AU_RS,
    poloidal_height=3.8/AU_RS,
    half_width=np.pi/180.0*40.0, 
    tilt=np.pi/180.0*10.0, 
    flattening=0.4, 
    pancaking=np.pi/180.0*20.0, 
    skew=np.pi/180.0*0.0, 
    twist=2.79284826, 
    flux=1e14,
    sigma=2.29962923,
    polarity=-1.0,
    chirality=1.0,
    x=1.0,
    y=0.0,
    z=0.0):
    
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
        twist=twist, 
        flux=flux,
        sigma=sigma,
        polarity=polarity,
        chirality=chirality
    )
    fr.init()

    fr.fit2remote()