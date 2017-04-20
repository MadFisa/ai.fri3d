
from ai.fri3d import FRi3D
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from astropy import units as u
from datetime import datetime
import calendar
from ai.shared.color import BLIND_PALETTE
import matplotlib.animation as animation

def fr(dd, d0, 
    theta0, theta1, atheta,
    phi0, vphi,
    Rt0, aRt, v0Rt, v1Rt,
    Rp0, Rp1, aRp,
    phi_hw,
    gamma0, vgamma,
    n, 
    theta_p,
    phi_s):

    tt = calendar.timegm(dd.timetuple())
    t0 = calendar.timegm(d0.timetuple())
    
    latitude = lambda t: (
        (theta0-theta1)*np.exp(-atheta*(t-t0))+theta1
    )
    longitude = lambda t: phi0+vphi*(t-t0)
    toroidal_height = lambda t: (
        (v0Rt-v1Rt)/aRt*(1.0-np.exp(-aRt*(t-t0)))+v1Rt*(t-t0)+Rt0
    )
    poloidal_height = lambda t: (Rp0-Rp1)*np.exp(-aRp*(t-t0))+Rp1
    half_width = lambda t: phi_hw
    tilt = lambda t: vgamma*(t-t0)+gamma0
    flattening = lambda t: n
    pancaking = lambda t: theta_p
    skew = lambda t: phi_s

    return FRi3D(
        latitude=latitude(tt),
        longitude=longitude(tt),
        toroidal_height=toroidal_height(tt),
        poloidal_height=poloidal_height(tt),
        half_width=half_width(tt),
        tilt=tilt(tt),
        flattening=flattening(tt),
        pancaking=pancaking(tt),
        skew=skew(tt)
    )

# MESSENGER:  0
# VEX:  0 300
# STA:  0 300
# -1.17432678485 0.000306500305199 1.98298246858e-05 0.000142307095103 1555.85227047 1055.21504775 0.0449207339106 0.000403800687986 -3.22440178699e-05
# [ -2.04958689e-02   3.06500305e-04   3.46095731e-07   1.42307095e-04
#    1.55585227e+06   6.95508000e+09   1.05521505e+06   6.72004614e+09
#    4.03800688e-04  -5.62764276e-07   5.00000000e-01   1.00000000e+14]

# STEREO-A:  0.047619047619 1.08492960138 0.24968096797
# AVERAGE:  0.566274324498
# [  8.60681862e-02   1.09480852e+00   4.28129047e-04   1.44777011e+06
#    9.02207628e+05   3.63949184e+09   8.44765805e-01   1.02216193e-01
#    5.57265924e+13]
# 4.93134381766 62.7279073549 48.4015152892

def f1(dd):
    return fr(dd, datetime(2011, 6, 4, 8, 54),
        u.deg.to(u.rad, 34.0), -2.04958689e-02, 3.06500305e-04,
        u.deg.to(u.rad, 130.0), 3.46095731e-07,
        u.R_sun.to(u.m, 12.0), 1.42307095e-04,
        1.55585227e+06,
        1.05521505e+06,
        u.R_sun.to(u.m, 3.0), 6.72004614e+09, 4.03800688e-04,
        u.deg.to(u.rad, 44.0),
        u.deg.to(u.rad, -35.0), -5.62764276e-07,
        0.3,
        u.deg.to(u.rad, 18.0),
        u.deg.to(u.rad, 0.0)
    )

# MESSENGER:  0
# VEX:  3000 0
# STA:  0 0
# 2.19147940736 0.000286256450714 5.33285997461e-05 9.31364197379e-05 1730.01821538 1297.42507118 0.0984987149627 6.41357227586e-05 3.248726452e-05
# [  3.82485311e-02   2.86256451e-04   9.30759651e-07   9.31364197e-05
#    1.73001822e+06   6.95508000e+09   1.29742507e+06   1.47351980e+10
#    6.41357228e-05   5.67009731e-07   5.00000000e-01   1.00000000e+14]

def f2(dd):
    return fr(dd, datetime(2011, 6, 4, 22, 54),
        u.deg.to(u.rad, 22.0), 3.82485311e-02, 2.86256451e-04,
        u.deg.to(u.rad, 125.0), 9.30759651e-07,
        u.R_sun.to(u.m, 12.0), 9.31364197e-05,
        1.73001822e+06,
        1.29742507e+06,
        u.R_sun.to(u.m, 4.0), 1.47351980e+10, 6.41357228e-05,
        u.deg.to(u.rad, 35.0),
        u.deg.to(u.rad, 35.0), 5.67009731e-07,
        0.4,
        u.deg.to(u.rad, 30.0),
        u.deg.to(u.rad, 0.0)
    )

fr1 = f1(datetime(2011, 6, 6, 12, 25))
print(
    u.rad.to(u.deg, fr1.latitude),
    u.rad.to(u.deg, fr1.longitude),
    u.m.to(u.au, fr1.poloidal_height),
    u.rad.to(u.deg, fr1.half_width),
    u.rad.to(u.deg, fr1.tilt),
    u.rad.to(u.deg, fr1.pancaking),
)
# theta       1.05786595            -1.17432678485
# phi         100.49858031          133.67763928622847
# Vt          978.437981
# Rp          0.0341321535935471    0.0449207339106
# thetaHW     39.4998412            44.0
# gamma       -49.52962457          -40.97997555415165
# n           0.306020000
# thetaP      29.887956865670798    18.0
# tau         1.60766417
# F           2.43052878e+14


fr2 = f2(datetime(2011, 6, 6, 16, 30))
print(
    u.rad.to(u.deg, fr2.latitude),
    u.rad.to(u.deg, fr2.longitude),
    u.m.to(u.au, fr2.poloidal_height),
    u.rad.to(u.deg, fr2.half_width),
    u.rad.to(u.deg, fr2.tilt),
    u.rad.to(u.deg, fr2.pancaking),
)
# theta       16.74504536           2.19147940736
# phi         122.24508571          132.98649109797597
# at          -0.385845146
# ! Vt        1298.18672
# Rp          0.07337108508724248   0.0984933301094
# thetaHW     43.00339063           35.0
# gamma       27.04916694           39.865292734515194
# n           0.470745851
# thetaP      27.434469883139027    29.999999999999996
# tau         1.04345531
# F           1.09514269e+14

def animate():

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d', adjustable='box', aspect=1.0)
    ax.set_xlabel('X [AU]')
    ax.set_zlabel('Z [AU]')
    ax.set_ylabel('Y [AU]')
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.0)
    ax.view_init(90.0, 0.0)

    # d = datetime(2011, 6, 5, 11, 30)

    dd = np.array([
        datetime.utcfromtimestamp(x) for x in np.linspace(
            calendar.timegm(datetime(2011, 6, 4, 6).timetuple()),
            calendar.timegm(datetime(2011, 6, 6, 12).timetuple()),
            500
        )
    ])

    cmes = []
    for d in dd:
        
        cmes_next = []
        if d > datetime(2011, 6, 4, 8, 54):
            fr1 = f1(d)
            x, y, z = fr1.shell()
            x *= u.m.to(u.au)
            y *= u.m.to(u.au)
            z *= u.m.to(u.au)
            cme1 = ax.plot_wireframe(x, y, z, color=BLIND_PALETTE['blue'], alpha=0.4)
            cmes_next.append(cme1)

        if d > datetime(2011, 6, 4, 22, 54):
            fr2 = f2(d)
            x, y, z = fr2.shell()
            x *= u.m.to(u.au)
            y *= u.m.to(u.au)
            z *= u.m.to(u.au)
            cme2 = ax.plot_wireframe(x, y, z, color=BLIND_PALETTE['vermillion'], alpha=0.4)
            cmes_next.append(cme2)

        if len(cmes_next) > 0:
            cmes.append(cmes_next)


    # plt1 = ax.plot_wireframe([], [], [])
    # plt2 = ax.plot_wireframe([], [], [])

    # def init():
    #     fr1 = f1(d[0])
    #     fr2 = f2(d[0])

    #     x, y, z = fr1.shell()
    #     x *= u.m.to(u.au)
    #     y *= u.m.to(u.au)
    #     z *= u.m.to(u.au)
    #     plt1 = ax.plot_wireframe(x, y, z, color=BLIND_PALETTE['blue'], alpha=0.4)

    #     x, y, z = fr2.shell()
    #     x *= u.m.to(u.au)
    #     y *= u.m.to(u.au)
    #     z *= u.m.to(u.au)
    #     plt2 = ax.plot_wireframe(x, y, z, color=BLIND_PALETTE['vermillion'], alpha=0.4)
        
    #     return (plt1, plt2)

    # def animate(i):
    #     fr1 = f1(d[i])
    #     x, y, z = fr1.shell()
    #     x *= u.m.to(u.au)
    #     y *= u.m.to(u.au)
    #     z *= u.m.to(u.au)
    #     plt1 = ax.plot_wireframe(x, y, z, color=BLIND_PALETTE['blue'], alpha=0.4)

    #     fr2 = f2(d[i])
    #     x, y, z = fr2.shell()
    #     x *= u.m.to(u.au)
    #     y *= u.m.to(u.au)
    #     z *= u.m.to(u.au)
    #     plt2 = ax.plot_wireframe(x, y, z, color=BLIND_PALETTE['vermillion'], alpha=0.4)

    #     return (plt1, plt2)

    anim = animation.ArtistAnimation(
        fig, 
        cmes, 
        interval=50,
        repeat=False,
        blit=False
    )

    anim.save('out.mp4')
    plt.close(fig)








animate()















# plt.show()

# dd = datetime(2011, 6, 5, 11, 30)
# tt = calendar.timegm(dd.timetuple())

# di = datetime(2011, 6, 5, 11, 30)
# ti = calendar.timegm(di.timetuple())

# d0 = datetime(2011, 6, 4, 8, 54)
# t0 = calendar.timegm(d0.timetuple())

# d1_sta = datetime(2011, 6, 6, 14, 10)
# t1_sta = calendar.timegm(d1_sta.timetuple())

# # tt = t1_sta

# # t = np.linspace(t0, t1_sta)

# theta0 = u.deg.to(u.rad, 34.0)
# theta1 = u.deg.to(u.rad, -1.17432678485)
# atheta = 0.000306500305199

# theta = lambda t: (
#     (theta0-theta1)*np.exp(-atheta*(t-t0))+theta1
# )

# phi0 = u.deg.to(u.rad, 130.0)
# vphi = u.deg.to(u.rad, 1.98298246858e-05)

# phi = lambda t: phi0+vphi*(t-t0)

# Rt0 = u.R_sun.to(u.m, 12.0)
# aRt = 0.000142307095103
# v0Rt = u.Unit('km/s').to(u.Unit('m/s'), 1555.85227047)
# v1Rt = u.Unit('km/s').to(u.Unit('m/s'), 1055.21504775)

# Rt = lambda t: (
#     (v0Rt-v1Rt)/aRt*(1.0-np.exp(-aRt*(t-t0)))+v1Rt*(t-t0)+Rt0
# )

# Rp0 = u.R_sun.to(u.m, 3.0)
# Rp1 = u.au.to(u.m, 0.0449207339106)
# aRp = 0.000403800687986

# Rp = lambda t: (Rp0-Rp1)*np.exp(-aRp*(t-t0))+Rp1

# phi_hw = lambda t: u.deg.to(u.rad, 44.0)

# gamma0 = u.deg.to(u.rad, -35.0)
# vgamma = u.deg.to(u.rad, -3.22440178699e-05)

# gamma = lambda t: vgamma*(t-t0)+gamma0

# n = lambda t: 0.3

# theta_p = lambda t: u.deg.to(u.rad, 18.0)

# phi_s = lambda t: u.deg.to(u.rad, 0.0)

# fr1 = FRi3D(
#     latitude=theta(tt),
#     longitude=phi(tt),
#     toroidal_height=Rt(tt),
#     poloidal_height=Rp(tt),
#     half_width=phi_hw(tt),
#     tilt=gamma(tt),
#     flattening=n(tt),
#     pancaking=theta_p(tt),
#     skew=phi_s(tt)
# )

# ##############################################



# d0 = datetime(2011, 6, 4, 22, 54)
# t0 = calendar.timegm(d0.timetuple())

# d1_sta = datetime(2011, 6, 7, 1)
# t1_sta = calendar.timegm(d1_sta.timetuple())

# theta0 = u.deg.to(u.rad, 22.0)
# theta1 = u.deg.to(u.rad, 2.69688579413)
# atheta = 0.000249269453776

# theta = lambda t: (
#     (theta0-theta1)*np.exp(-atheta*(t-t0))+theta1
# )

# phi0 = u.deg.to(u.rad, 125.0)
# vphi = u.deg.to(u.rad, 5.5948121101e-05)

# phi = lambda t: phi0+vphi*(t-t0)

# Rt0 = u.R_sun.to(u.m, 12.0)
# aRt = 0.000134671915099
# v0Rt = u.Unit('km/s').to(u.Unit('m/s'), 1869.96937536)
# v1Rt = u.Unit('km/s').to(u.Unit('m/s'), 1331.73412132)

# Rt = lambda t: (
#     (v0Rt-v1Rt)/aRt*(1.0-np.exp(-aRt*(t-t0)))+v1Rt*(t-t0)+Rt0
# )

# Rp0 = u.R_sun.to(u.m, 4.0)
# Rp1 = u.au.to(u.m, 0.0961103821753)
# aRp = 9.10911358522e-05

# Rp = lambda t: (Rp0-Rp1)*np.exp(-aRp*(t-t0))+Rp1

# phi_hw = lambda t: u.deg.to(u.rad, 35.0)

# gamma0 = u.deg.to(u.rad, 35.0)
# vgamma = u.deg.to(u.rad, 3.40477040217e-05)

# gamma = lambda t: vgamma*(t-t0)+gamma0

# n = lambda t: 0.4

# theta_p = lambda t: u.deg.to(u.rad, 30.0)

# phi_s = lambda t: u.deg.to(u.rad, 0.0)

# fr2 = FRi3D(
#     latitude=theta(tt),
#     longitude=phi(tt),
#     toroidal_height=Rt(tt),
#     poloidal_height=Rp(tt),
#     half_width=phi_hw(tt),
#     tilt=gamma(tt),
#     flattening=n(tt),
#     pancaking=theta_p(tt),
#     skew=phi_s(tt)
# )

# #########################################

# fig = plt.figure(figsize=(10,10))
# ax = fig.add_subplot(111, projection='3d', adjustable='box', aspect=1.0)
# x, y, z = fr1.shell()
# x *= u.m.to(u.au)
# y *= u.m.to(u.au)
# z *= u.m.to(u.au)
# ax.plot_wireframe(x, y, z, color=BLIND_PALETTE['blue'], alpha=0.4)

# x, y, z = fr2.shell()
# x *= u.m.to(u.au)
# y *= u.m.to(u.au)
# z *= u.m.to(u.au)
# ax.plot_wireframe(x, y, z, color=BLIND_PALETTE['vermillion'], alpha=0.4)

# ax.set_xlabel('X [AU]')
# ax.set_ylabel('Y [AU]')
# ax.set_zlabel('Z [AU]')

# plt.show()