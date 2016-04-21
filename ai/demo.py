
from datetime import datetime

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
    # cdas.set_cache(True, 'data')
    cdas.set_cache(False)
    data = cdas.get_data(
        'sp_phys', 
        'STA_L1_MAG_RTN', 
        datetime(2010, 12, 15, 13), 
        datetime(2010, 12, 16, 4), 
        ['BFIELD']
    )
    b = data['BTOTAL']
    bx = data['BR']
    by = data['BT']
    bz = data['BN']
    b = b[0::1800]
    bx = bx[0::1800]
    by = by[0::1800]
    bz = bz[0::1800]
    # plt.plot(b, 'k')
    # plt.plot(bx, 'r')
    # plt.plot(by, 'g')
    # plt.plot(bz, 'b')
    # print(b.size, bx.size, by.size, bz.size)
    # plt.show()
    fr = FRi3D()
    fr.fit2insitu(b, bx, by, bz)

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

def test_article(
    latitude=np.pi/180.0*4.56130681, 
    longitude=-np.pi/180.0*18.89423998, 
    toroidal_height=0.7,
    poloidal_height=0.20765019,
    half_width=np.pi/180.0*40.38090972, 
    tilt=np.pi/180.0*3.39496761, 
    flattening=0.57245453, 
    pancaking=np.pi/180.0*20.0, 
    skew=np.pi/180.0*0.0, 
    twist=4.93964511, 
    flux=1e14,
    sigma=1.01345514,
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

    b_ = fr.evocut1d(x, y, z, 
        toroidal_height=toroidal_height
    )

    
    cdas.set_cache(False)
    data = cdas.get_data(
        'sp_phys', 
        'STA_L1_MAG_RTN', 
        datetime(2010, 12, 15, 13), 
        datetime(2010, 12, 16, 4), # 3
        ['BFIELD']
    )
    b = data['BTOTAL']
    bx = data['BR']
    by = data['BT']
    bz = data['BN']
    b = b[0::1800]
    bx = bx[0::1800]
    by = by[0::1800]
    bz = bz[0::1800]

    mb = np.mean(b)
    
    sc = mb/np.mean(b_[:,0])
    bb = b_[:,0]*sc
    bbx = b_[:,1]*sc
    bby = b_[:,2]*sc
    bbz = b_[:,3]*sc

    t_ = np.linspace(0.0, 1.0, b_.shape[0])
    t = np.linspace(0.0, 1.0, b.size)

    fig = plt.figure()
    plt.plot(t_, bb, '--k', linewidth=2, label='B')
    plt.plot(t, b, 'k', linewidth=2, label='B')
    plt.plot(t_, bbx, '--r', linewidth=2, label='Bx')
    plt.plot(t, bx, 'r', linewidth=2, label='Bx')
    plt.plot(t_, bby, '--g', linewidth=2, label='By')
    plt.plot(t, by, 'g', linewidth=2, label='By')
    plt.plot(t_, bbz, '--b', linewidth=2, label='Bz')
    plt.plot(t, bz, 'b', linewidth=2, label='Bz')
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