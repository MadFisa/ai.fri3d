
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
        datetime(2010, 12, 15, 10, 20), 
        datetime(2010, 12, 16, 4), 
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

# [  8.54355628 -20.39145993   0.11084887   2.16074957   0.50537504
#    6.77391367   2.19169087   1.20288016]
# 2.49470631039

# [  8.61910947 -21.10748715   0.12170966   1.62849082   0.51726676
#    6.87282201   2.72165576   1.20756696]
# 2.45206364245

# [  5.90279467 -23.54842414   0.14707356   4.71965664   0.53860386
#    9.57691177   1.29043543]
# 2.66978262064

# [  6.61754667 -19.87803594   0.10132359   5.38408605   0.57112203
#    8.73950104   1.25355065]
# 2.64508988577

# [  6.34967948 -23.52010342   0.10219471   5.54132107   0.4578051
#    9.69389798   1.30678127]
# 2.60204704475

# fix end

# [  8.51140005 -19.76761041   0.12946792   1.13415351   0.47783838
#    5.57315287   0.83631745]
# 2.71855769001


def test_article(
    latitude=np.pi/180.0*8.51140005, 
    longitude=-np.pi/180.0*19.76761041, 
    toroidal_height=0.7,
    poloidal_height=0.12946792,
    half_width=np.pi/180.0*40, 
    tilt=np.pi/180.0*1.13415351, 
    flattening=0.47783838, 
    pancaking=np.pi/180.0*20.0, 
    skew=np.pi/180.0*0.0, 
    twist=5.57315287, 
    flux=1e14,
    sigma=2.05,
    polarity=-1.0,
    chirality=1.0,
    ratio=0.83631745,
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
    fr.toroidal_height = 1.0
    fr.init()
    print(twist/fr._initial_axis_s(fr.half_width))
    fr.toroidal_height = toroidal_height
    fr.init()
    
    

    t_begin = datetime(2010, 12, 15, 10, 20)
    t_end = datetime(2010, 12, 16, 4)
    dt = timedelta(minutes=30)
    
    cdas.set_cache(True, 'data')
    data = cdas.get_data(
        'sp_phys', 
        'STA_L1_MAG_RTN', 
        t_begin, 
        t_end,
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

    t = t0[-1]-(t0[-1]-t0[0])*ratio*np.linspace(1.0, 0.0, b_.shape[0])
    # t = t0[0]+(t0[-1]-t0[0])*ratio*np.linspace(0.0, 1.0, b_.shape[0])
    b = b_[:,0]
    bx = b_[:,1]
    by = b_[:,2]
    bz = b_[:,3]
    if False:
        t1 = t0
        b1 = np.interp(t1, t, b)
        bx1 = np.interp(t1, t, bx)
        by1 = np.interp(t1, t, by)
        bz1 = np.interp(t1, t, bz)
    else:
        t1 = t
        b1 = b
        bx1 = bx
        by1 = by
        bz1 = bz

    b1_mean = np.mean(b1)

    coeff = b0_mean/b1_mean
    b1 *= coeff
    bx1 *= coeff
    by1 *= coeff
    bz1 *= coeff

    t0 = np.array([datetime.fromtimestamp(x) for x in t0])
    t1 = np.array([datetime.fromtimestamp(x) for x in t1])

    cdas.set_cache(False)
    data = cdas.get_data(
        'sp_phys', 
        'STA_L1_MAG_RTN', 
        t_begin-timedelta(hours=12), 
        t_end+timedelta(hours=12),
        ['BFIELD'],
        cdf=True
    )
    t = data['Epoch'][::1800]
    b = data['BFIELD'][::1800,3]
    bx = data['BFIELD'][::1800,0]
    by = data['BFIELD'][::1800,1]
    bz = data['BFIELD'][::1800,2]

    fig = plt.figure()
    mask = t1 <= t_end
    not_mask = t1 >= t[mask][-1]
    plt.plot(t1[mask], b1[mask], '--k', linewidth=2, label='B')
    plt.plot(t1[not_mask], b1[not_mask], '--k', linewidth=2, label='B', alpha=0.2)
    # plt.plot(t0, b0, 'k', linewidth=2, label='B')
    plt.plot(t, b, 'k', label='B')
    plt.plot(t1[mask], bx1[mask], '--r', linewidth=2, label='Bx')
    plt.plot(t1[not_mask], bx1[not_mask], '--r', linewidth=2, label='Bx', alpha=0.2)
    # plt.plot(t0, bx0, 'r', linewidth=2, label='Bx')
    plt.plot(t, bx, 'r', label='Bx')
    plt.plot(t1[mask], by1[mask], '--g', linewidth=2, label='By')
    plt.plot(t1[not_mask], by1[not_mask], '--g', linewidth=2, label='By', alpha=0.2)
    # plt.plot(t0, by0, 'g', linewidth=2, label='By')
    plt.plot(t, by, 'g', label='By')
    plt.plot(t1[mask], bz1[mask], '--b', linewidth=2, label='Bz')
    plt.plot(t1[not_mask], bz1[not_mask], '--b', linewidth=2, label='Bz', alpha=0.2)
    # plt.plot(t0, bz0, 'b', linewidth=2, label='Bz')
    plt.plot(t, bz, 'b', label='Bz')
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