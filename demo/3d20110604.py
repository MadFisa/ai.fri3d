
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

def f1(dd):
    return fr(dd, datetime(2011, 6, 4, 8, 54),
        u.deg.to(u.rad, 34.0), u.deg.to(u.rad, -1.17432678485), 0.000306500305199,
        u.deg.to(u.rad, 130.0), u.deg.to(u.rad, 1.98298246858e-05),
        u.R_sun.to(u.m, 12.0), 0.000142307095103,
        u.Unit('km/s').to(u.Unit('m/s'), 1555.85227047),
        u.Unit('km/s').to(u.Unit('m/s'), 1055.21504775),
        u.R_sun.to(u.m, 3.0), u.au.to(u.m, 0.0449207339106), 0.000403800687986,
        u.deg.to(u.rad, 44.0),
        u.deg.to(u.rad, -35.0), u.deg.to(u.rad, -3.22440178699e-05),
        0.3,
        u.deg.to(u.rad, 18.0),
        u.deg.to(u.rad, 0.0)
    )

def f2(dd):
    return fr(dd, datetime(2011, 6, 4, 22, 54),
        u.deg.to(u.rad, 22.0), u.deg.to(u.rad, 2.19147940736), 0.000286256450714,
        u.deg.to(u.rad, 125.0), u.deg.to(u.rad, 5.33285997461e-05),
        u.R_sun.to(u.m, 12.0), 9.31364197379e-05,
        u.Unit('km/s').to(u.Unit('m/s'), 1730.01821538),
        u.Unit('km/s').to(u.Unit('m/s'), 1297.42507118),
        u.R_sun.to(u.m, 4.0), u.au.to(u.m, 0.0984987149627), 6.41357227586e-05,
        u.deg.to(u.rad, 35.0),
        u.deg.to(u.rad, 35.0), u.deg.to(u.rad, 3.248726452e-05),
        0.4,
        u.deg.to(u.rad, 30.0),
        u.deg.to(u.rad, 0.0)
    )

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