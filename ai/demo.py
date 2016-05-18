
import time
from datetime import datetime, timedelta

import numpy as np

from matplotlib import rcParams
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.path as mpath
import matplotlib.colors as colors
from matplotlib.colorbar import ColorbarBase
import matplotlib.gridspec as gridspec

import ai.cdas as cdas

from scipy.io import readsav

from importlib import reload

from ai.fri3d import FRi3D

from ai import cs

rcParams.update({'font.size': 14})

AU_KM = 149597870.7
RS_KM = 6.957e5
AU_RS = AU_KM/RS_KM
RS_AU = RS_KM/AU_KM

BLIND_PALETTE = {
    'orange': 
        (0.901960784, 0.623529412, 0.0),
    'sky-blue': 
        (0.337254902, 0.705882353, 0.91372549),
    'bluish-green': 
        (0.0, 0.619607843, 0.450980392),
    'yellow': 
        (0.941176471, 0.894117647, 0.258823529),
    'blue': 
        (0.0, 0.447058824, 0.698039216),
    'vermillion': 
        (0.835294118, 0.368627451, 0.0),
    'reddish-purple': 
        (0.8, 0.474509804, 0.654901961)
}

def paper_shell(
    latitude=0.0, 
    longitude=0.0, 
    toroidal_height=1.0, 
    poloidal_height=0.15, 
    half_width=np.pi/180.0*40.0, 
    tilt=np.pi/180.0*0.0, 
    flattening=0.5, 
    pancaking=None, 
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

    fig = plt.figure(figsize=(30,15))

    top_view = (90.0,-90.0)
    side_view = (0.0,-90.0)

    # for i, skew in enumerate([0.0, np.pi/180.0*20.0]):
    # for i, pancaking in enumerate([None, np.pi/180.0*30.0]):
    for i, flattening in enumerate([0.7, 0.35]):
        fr.flattening = flattening
        fr.init()

        ax = fig.add_subplot(2, 3, i*3+1, 
            projection='3d', 
            adjustable='box', 
            aspect=1.0
        )

        x, y, z = fr.shell()
        ax.plot_wireframe(x, y, z, color=BLIND_PALETTE['blue'], alpha=0.4)

        # plt.subplots_adjust(hspace = 0.001)

        ax.set_xlabel('X', labelpad=20)
        ax.set_ylabel('Y', labelpad=20)
        # ax.set_zlabel('Z', labelpad=20)

        ax.set_xlim3d(0.0, 1.2)
        ax.set_ylim3d(-0.6, 0.6)
        ax.set_zlim3d(-0.6, 0.6)
        
        ax.tick_params(axis='x', pad=10)
        ax.tick_params(axis='y', pad=10)
        ax.tick_params(axis='z', pad=10)

        ax.w_zaxis.line.set_lw(0.)
        ax.set_zticks([])

        # ax.w_yaxis.line.set_lw(0.)
        # ax.set_yticks([])

        ax.view_init(*top_view)

        if i == 0:
            ax.set_title('(a) front flattening (top view)')

    fr.flattening = 0.5
    for i, pancaking in enumerate([None, np.pi/180.0*30.0]):
        fr.pancaking = pancaking
        fr.init()

        ax = fig.add_subplot(2, 3, i*3+2, 
            projection='3d', 
            adjustable='box', 
            aspect=1.0
        )

        x, y, z = fr.shell()
        ax.plot_wireframe(x, y, z, color=BLIND_PALETTE['blue'], alpha=0.4)

        # plt.subplots_adjust(hspace = 0.001)

        ax.set_xlabel('X', labelpad=20)
        # ax.set_ylabel('Y', labelpad=20)
        ax.set_zlabel('Z', labelpad=20)
        
        ax.set_xlim3d(0.0, 1.2)
        ax.set_ylim3d(-0.6, 0.6)
        ax.set_zlim3d(-0.6, 0.6)

        ax.tick_params(axis='x', pad=10)
        ax.tick_params(axis='y', pad=10)
        ax.tick_params(axis='z', pad=10)

        # ax.w_zaxis.line.set_lw(0.)
        # ax.set_zticks([])

        ax.w_yaxis.line.set_lw(0.)
        ax.set_yticks([])

        ax.view_init(*side_view)

        if i == 0:
            ax.set_title('(b) pancaking (side view)')

    fr.pancaking = None
    for i, skew in enumerate([0.0, np.pi/180.0*30.0]):
        fr.skew = skew
        fr.init()

        ax = fig.add_subplot(2, 3, i*3+3, 
            projection='3d', 
            adjustable='box', 
            aspect=1.0
        )

        x, y, z = fr.shell()
        ax.plot_wireframe(x, y, z, color=BLIND_PALETTE['blue'], alpha=0.4)

        # plt.subplots_adjust(hspace = 0.001)

        ax.set_xlabel('X', labelpad=20)
        ax.set_ylabel('Y', labelpad=20)
        # ax.set_zlabel('Z', labelpad=20)
        
        ax.set_xlim3d(0.0, 1.2)
        ax.set_ylim3d(-0.6, 0.6)
        ax.set_zlim3d(-0.6, 0.6)

        ax.tick_params(axis='x', pad=10)
        ax.tick_params(axis='y', pad=10)
        ax.tick_params(axis='z', pad=10)

        ax.w_zaxis.line.set_lw(0.)
        ax.set_zticks([])

        # ax.w_yaxis.line.set_lw(0.)
        # ax.set_yticks([])

        ax.view_init(*top_view)

        if i == 0:
            ax.set_title('(c) skewing (top view)')

    plt.tight_layout()
    plt.show()

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

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d', adjustable='box', aspect=1.0)
    x, y, z = fr.shell()
    ax.plot_wireframe(x, y, z, color=BLIND_PALETTE['blue'], alpha=0.4)

    # ax.set_xlabel('X [AU]')
    # ax.set_ylabel('Y [AU]')
    # ax.set_zlabel('Z [AU]]')

    ax.set_xlabel('X', labelpad=10)
    ax.set_ylabel('Y', labelpad=10)
    # ax.set_zlabel('Z')

    ax.set_xlim([0.0, 1.2])
    ax.set_ylim([-0.6, 0.6])
    ax.set_zlim([-0.6, 0.6])

    # ax.xaxis._axinfo['label']['space_factor'] = 2.0
    # ax.yaxis._axinfo['label']['space_factor'] = 2.0
    # ax.zaxis._axinfo['label']['space_factor'] = 2.0
    ax.tick_params(axis='x', pad=10)
    ax.tick_params(axis='y', pad=10)

    ax.w_zaxis.line.set_lw(0.)
    ax.set_zticks([])
    # ax.zaxis.set_visible(False)


    ax.view_init(elev=0.0, azim=-90.0)

    # fig = plt.figure()
    # x, y, z = fr.shell()
    # plt.plot(y, z, '.r')

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
    latitude=-np.pi/180.0*5.0, 
    longitude=np.pi/180.0*123.0, 
    # longitude=np.pi/180.0*90.0, 
    toroidal_height=12.5/AU_RS,
    poloidal_height=3.5/AU_RS,
    half_width=np.pi/180.0*43.0, 
    tilt=np.pi/180.0*44.0, 
    flattening=0.37, 
    pancaking=np.pi/180.0*18.0, 
    skew=np.pi/180.0*5.0, 
    tapering=1.0,
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
        tapering=tapering,
        twist=twist, 
        flux=flux,
        sigma=sigma,
        polarity=polarity,
        chirality=chirality
    )
    fr.init()


    sta_lon = 129.266*np.pi/180.0
    sta_lat = -2.511*np.pi/180.0
    sta_r = 0.960188
    sta_fov = sta_r*AU_RS*np.tan(4.0*np.pi/180.0)
    stb_lon = -132.444*np.pi/180.0
    stb_lat = 7.074*np.pi/180.0
    stb_r = 1.079771
    stb_fov = stb_r*AU_RS*np.tan(4.0*np.pi/180.0)
    soho_lon = 0.0*np.pi/180.0
    soho_lat = 0.0*np.pi/180.0
    soho_r = 1.0
    soho_fov = 32.0

    x0, y0, z0 = fr.shell()
    fig = plt.figure()
    ax = fig.add_subplot(111, 
        projection='3d', 
        adjustable='box', 
        aspect=1.0
    )
    ax.plot_wireframe(x0, y0, z0, color=BLIND_PALETTE['blue'], alpha=0.4)
    plt.show()


    fig = plt.figure()

    gs = gridspec.GridSpec(2, 3)
    gs.update(wspace=0.0, hspace=0.0)

    ax = plt.subplot(gs[0])
    ax.imshow(
        plt.imread('/media/data/Documents/Articles/2016_Isavnin_FRi3D/20130106_103900_dbc2B_opt.png'),
        zorder=0,
        extent=[-stb_fov+0.05, stb_fov+0.05, -stb_fov-0.1, stb_fov-0.1]
    )
    ax.set_xlim([-stb_fov+0.05, stb_fov+0.05])
    ax.set_ylim([-stb_fov-0.1, stb_fov-0.1])
    ax.set_axis_bgcolor('black')
    plt.axis('off')

    ax = plt.subplot(gs[1])
    ax.imshow(
        plt.imread('/media/data/Documents/Articles/2016_Isavnin_FRi3D/20130106_1042_c3_1024_opt.png'),
        zorder=0,
        extent=[-soho_fov+0.3, soho_fov+0.3, -soho_fov+1.33, soho_fov+1.33]
    )
    ax.set_xlim([-soho_fov+0.3, soho_fov+0.3])
    ax.set_ylim([-soho_fov+1.33, soho_fov+1.33])
    ax.set_axis_bgcolor('black')
    plt.axis('off')

    ax = plt.subplot(gs[2])
    ax.imshow(
        plt.imread('/media/data/Documents/Articles/2016_Isavnin_FRi3D/20130106_103900_dbc2A_opt.png'),
        zorder=0,
        extent=[-sta_fov, sta_fov, -sta_fov+0.04, sta_fov+0.04]
    )
    ax.set_xlim([-sta_fov, sta_fov])
    ax.set_ylim([-sta_fov+0.04, sta_fov+0.04])
    ax.set_axis_bgcolor('black')
    plt.axis('off')

    
    ax = plt.subplot(gs[3])
    ax.imshow(
        plt.imread('/media/data/Documents/Articles/2016_Isavnin_FRi3D/20130106_103900_dbc2B_opt.png'),
        zorder=0,
        extent=[-stb_fov-0.03, stb_fov-0.03, -stb_fov-0.0, stb_fov-0.0]
    )
    # ax.plot([0.0], [0.0], '.y', markersize=5.0)
    T = cs.mx_rot_z(-stb_lon)*cs.mx_rot_y(stb_lat)
    x = T[0,0]*x0+T[0,1]*y0+T[0,2]*z0
    y = T[1,0]*x0+T[1,1]*y0+T[1,2]*z0
    z = T[2,0]*x0+T[2,1]*y0+T[2,2]*z0
    y = stb_r/(stb_r-x)*y
    z = stb_r/(stb_r-x)*z
    ax.scatter(y*AU_RS, z*AU_RS, 3, color=BLIND_PALETTE['yellow'], marker='.')
    ax.set_xlim([-stb_fov-0.03, stb_fov-0.03])
    ax.set_ylim([-stb_fov-0.0, stb_fov-0.0])
    ax.set_axis_bgcolor('black')
    plt.axis('off')

    ax = plt.subplot(gs[4])
    ax.imshow(
        plt.imread('/media/data/Documents/Articles/2016_Isavnin_FRi3D/20130106_1042_c3_1024_opt.png'),
        zorder=0,
        extent=[-soho_fov+0.37, soho_fov+0.37, -soho_fov+1.19, soho_fov+1.19]
    )
    # ax.plot([0.0], [0.0], '.y', markersize=5.0)
    T = cs.mx_rot_z(-soho_lon)*cs.mx_rot_y(soho_lat)
    x = T[0,0]*x0+T[0,1]*y0+T[0,2]*z0
    y = T[1,0]*x0+T[1,1]*y0+T[1,2]*z0
    z = T[2,0]*x0+T[2,1]*y0+T[2,2]*z0
    y = soho_r/(soho_r-x)*y
    z = soho_r/(soho_r-x)*z
    ax.scatter(y*AU_RS, z*AU_RS, 3, color=BLIND_PALETTE['yellow'], marker='.')
    ax.set_xlim([-soho_fov+0.37, soho_fov+0.37])
    ax.set_ylim([-soho_fov+1.19, soho_fov+1.19])
    ax.set_axis_bgcolor('black')
    plt.axis('off')

    
    ax = plt.subplot(gs[5])
    ax.imshow(
        plt.imread('/media/data/Documents/Articles/2016_Isavnin_FRi3D/20130106_103900_dbc2A_opt.png'),
        zorder=0,
        extent=[-sta_fov+0.1, sta_fov+0.1, -sta_fov+0.04, sta_fov+0.04]
    )
    # ax.plot([0.0], [0.0], '.y', markersize=5.0)
    T = cs.mx_rot_z(-sta_lon)*cs.mx_rot_y(sta_lat)
    x = T[0,0]*x0+T[0,1]*y0+T[0,2]*z0
    y = T[1,0]*x0+T[1,1]*y0+T[1,2]*z0
    z = T[2,0]*x0+T[2,1]*y0+T[2,2]*z0
    y = sta_r/(sta_r-x)*y
    z = sta_r/(sta_r-x)*z
    ax.scatter(y*AU_RS, z*AU_RS, 3, color=BLIND_PALETTE['yellow'], marker='.')
    ax.set_xlim([-sta_fov+0.1, sta_fov+0.1])
    ax.set_ylim([-sta_fov+0.04, sta_fov+0.04])
    ax.set_axis_bgcolor('black')
    ax.patch.set_facecolor('black')
    plt.axis('off')

    plt.show()

def test_insitu_mes(
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

def test_insitu_vex():
    pass

def test_insitu_sta():
    pass

def prepare_data():
    dt = time.mktime(datetime(1979, 1, 1).timetuple())
    # MESSENGER
    data=readsav(
        'MES_2007to2014_HEEQ.sav', 
        python_dict=True, 
        verbose=True
    )
    t1 = time.mktime(datetime(2013, 1, 7).timetuple())
    t2 = time.mktime(datetime(2013, 1, 10).timetuple())
    mask = np.logical_and(
        data['mes']['time']+dt >= t1, 
        data['mes']['time']+dt <= t2
    )
    t = np.array(
        [datetime.fromtimestamp(x+dt) for x in data['mes']['time'][mask]]
    )
    b = data['mes']['btot'][mask]
    bx = data['mes']['bx'][mask]
    by = data['mes']['by'][mask]
    bz = data['mes']['bz'][mask]
    np.savez('./mes', t=t, b=b, bx=bx, by=by, bz=bz)
    # Venus Express
    data=readsav(
        'VEX_2007to2014_HEEQ_removed.sav', 
        python_dict=True, 
        verbose=True
    )
    t1 = time.mktime(datetime(2013, 1, 7).timetuple())
    t2 = time.mktime(datetime(2013, 1, 11).timetuple())
    mask = np.logical_and(
        data['vex']['time']+dt >= t1, 
        data['vex']['time']+dt <= t2
    )
    t = np.array(
        [datetime.fromtimestamp(x+dt) for x in data['vex']['time'][mask]]
    )
    b = data['vex']['btot'][mask]
    bx = data['vex']['bx'][mask]
    by = data['vex']['by'][mask]
    bz = data['vex']['bz'][mask]
    np.savez('./vex', t=t, b=b, bx=bx, by=by, bz=bz)
