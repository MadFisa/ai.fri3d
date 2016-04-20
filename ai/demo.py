
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
    data = cdas.get_data(
        'sp_phys', 
        'STA_L1_MAG_RTN', 
        datetime(2010, 12, 15, 11), 
        datetime(2010, 12, 16, 3), 
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
# 0.16877183 -0.26586265  0.67679073 -0.14286905  2.58854371  2.35582625    
# 0.14807257 -0.25365719  0.7000947  -0.05024849  2.63836177  1.25977319
# 0.13163583 -0.24615301  0.58233117  0.02463952  2.73774167  2.67160919
# 0.13570772 -0.32436623  0.81936515 -0.15440772  2.36374527  2.91100385
# 0.17149158 -0.3181038   0.83511343  0.07445171  2.33873935  2.51572573
# 0.15824666 -0.26875461  0.60119371 -0.09408917  2.4024972   2.96945038
# 0.16877183 -0.26586265  0.65536465 -0.01748483  2.52567021  1.15773627
# 0.16631833 -0.26934347  0.70505349 -0.05024849  2.79284826  1.53810365
# 0.17115155 -0.26248651  0.64937786  0.05194959  2.39038277  2.562603 
# 0.16265419 -0.22318446  0.61579415 -0.0757028   2.534318    1.42928445
# 0.14844871 -0.26586265  0.67679073 -0.13569367  2.24454038  2.35582625
# 0.17295771 -0.30531659  0.78202743  0.00293968  2.22110602  2.69078082
# 0.15367821 -0.2110141   0.564125   -0.15451448  2.42153465  2.34769135
# 0.15582216 -0.28076383  0.70342751  0.11285947  2.62298742  1.83564513
# 0.16476683 -0.24658973  0.6181146  -0.10438553  2.61253526  1.20518249
# 0.16631833 -0.27972665  0.69614774 -0.13859975  2.79284826  2.29962923

# test_shell(
#     latitude=0.16631833, 
#     longitude=-0.27972665, 
#     toroidal_height=0.3,
#     poloidal_height=0.07,
#     half_width=0.69614774, 
#     tilt=-0.13859975, 
#     flattening=0.5, 
#     pancaking=np.pi/180.0*30.0, 
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

def test_article(
    latitude=0.16631833, 
    longitude=-0.27972665, 
    toroidal_height=0.8,
    poloidal_height=0.07,
    half_width=0.69614774, 
    tilt=-0.13859975, 
    flattening=0.5, 
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

    b_ = fr.evocut1d(x, y, z, 
        toroidal_height=toroidal_height
    )

    
    cdas.set_cache(True, 'data')
    data = cdas.get_data(
        'sp_phys', 
        'STA_L1_MAG_RTN', 
        datetime(2010, 12, 15, 13), 
        datetime(2010, 12, 16, 3), 
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
