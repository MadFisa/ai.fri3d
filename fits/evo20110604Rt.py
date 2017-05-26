
import numpy as np
from datetime import datetime
import calendar
from astropy import units as u
from matplotlib import pyplot as plt
from ai.fri3d import FRi3D
from mpl_toolkits.mplot3d import proj3d
from ai.shared.color import BLIND_PALETTE

di = datetime(2011, 6, 5, 11, 30)
ti = calendar.timegm(di.timetuple())

d0_cor1 = datetime(2011, 6, 4, 8, 54)
t0_cor1 = calendar.timegm(d0_cor1.timetuple())
latitude_cor1 = u.deg.to(u.rad, 30.0)
longitude_cor1 = u.deg.to(u.rad, 110.0)
toroidal_height_cor1 = u.R_sun.to(u.m, 12.5)
poloidal_height_cor1 = u.R_sun.to(u.m, 3.5)
half_width_cor1 = u.deg.to(u.rad, 40.0)
tilt_cor1 = u.deg.to(u.rad, 37.0)
flattening_cor1 = 0.4
pancaking_cor1 = u.deg.to(u.rad, 25.0)
d0_vex1 = datetime(2011, 6, 5, 8, 45)
d1_vex1 = datetime(2011, 6, 5, 11, 50)
t0_vex1 = calendar.timegm(d0_vex1.timetuple())
t1_vex1 = calendar.timegm(d1_vex1.timetuple())

p1 = np.array([
    0.00000000e+00, 0.00000000e+00, 1.11498287e+06, 0.00000000e+00,
    1.82344644e+00, 9.32055287e-02, 2.60248888e+14, 8.12884830e-02,
    2.57463174e+00, 1.49408743e+10, 6.98131701e-01, 8.33229899e-01,
    2.06464233e-01, 4.36332313e-01, 2.30014363e+14, 0.00000000e+00,
    0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
    0.00000000e+00, 0.00000000e+00, 0.00000000e+00
])

p_theta1 = np.polyfit([t0_cor1, t1_vex1], [latitude_cor1, p1[7]], 1)
theta1 = lambda t: (
    np.polyval(p_theta1, t)
    if t <= t1_vex1 else
    p1[7]
)
p_phi1 = np.polyfit([t0_cor1, t1_vex1], [longitude_cor1, p1[8]], 1)
phi1 = lambda t: (
    np.polyval(p_phi1, t)
    if t <= t1_vex1 else
    p1[8]
)
Rt1 = lambda t: p1[2]*(t-t0_cor1)+toroidal_height_cor1
p_Rp1 = np.polyfit([t0_cor1, t1_vex1], [poloidal_height_cor1, p1[9]], 1)
Rp1 = lambda t: (
    np.polyval(p_Rp1, t)
    if t <= t1_vex1 else
    p1[9]
)
p_gamma1 = np.polyfit([t0_cor1, t1_vex1], [tilt_cor1, p1[11]], 1)
gamma1 = lambda t: (
    np.polyval(p_gamma1, t)
    if t <= t1_vex1 else
    p1[10]
)

d0_cor2 = datetime(2011, 6, 4, 22, 54)
t0_cor2 = calendar.timegm(d0_cor2.timetuple())
latitude_cor2 = u.deg.to(u.rad, 22.0)
longitude_cor2 = u.deg.to(u.rad, 125.0)
toroidal_height_cor2 = u.R_sun.to(u.m, 12.0)
poloidal_height_cor2 = u.R_sun.to(u.m, 4.0)
half_width_cor2 = u.deg.to(u.rad, 35.0)
tilt_cor2 = u.deg.to(u.rad, 35.0)
flattening_cor2 = 0.4
pancaking_cor2 = u.deg.to(u.rad, 30.0)
d0_vex2 = datetime(2011, 6, 5, 15, 30)
d1_vex2 = datetime(2011, 6, 5, 22, 30)
t0_vex2 = calendar.timegm(d0_vex2.timetuple())
t1_vex2 = calendar.timegm(d1_vex2.timetuple())
d0_sta2 = datetime(2011, 6, 6, 12, 25)
d1_sta2 = datetime(2011, 6, 6, 14, 10)
t0_sta2 = calendar.timegm(d0_sta2.timetuple())
t1_sta2 = calendar.timegm(d1_sta2.timetuple())


p2 = np.array([
    0.00000000e+00, 0.00000000e+00, 1.63103908e+06, 1.68411547e+06,
    1.95001830e+00, 9.71007874e-01, 5.86563348e+14, 2.13230550e-01,
    2.11464106e+00, 1.33347207e+10, 6.10865238e-01 -1.86526317e-01,
    7.83222407e-01, 5.23598776e-01, 4.15008461e+14, 0.00000000e+00,
    0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
    0.00000000e+00, 0.00000000e+00, 1.00000000e+14
])

p_theta21 = np.polyfit([t0_cor2, t1_vex2], [latitude_cor2, p2[7]], 1)
p_theta22 = np.polyfit([t1_vex2, t1_sta2], [p2[7], p2[15]], 1)
theta2 = lambda t: (
    np.polyval(p_theta21, t) 
    if t <= t1_vex2 else
    np.polyval(p_theta22, t)
)

p_phi21 = np.polyfit([t0_cor2, t1_vex2], [longitude_cor2, p2[8]], 1)
p_phi22 = np.polyfit([t1_vex2, t1_sta2], [p2[8], p2[16]], 1)
phi2 = lambda t: (
    np.polyval(p_phi21, t)
    if t <= t1_vex2 else
    np.polyval(p_phi22, t)
)

Rt2 = lambda t: (
    p2[2]*(t-t0_cor2)+toroidal_height_cor2
    if t <= ti else
    p2[2]*(ti-t0_cor2)+toroidal_height_cor2+p2[3]*(t-ti)
)

p_Rp21 = np.polyfit([t0_cor2, t1_vex2], [poloidal_height_cor2, p2[9]], 1)
p_Rp22 = np.polyfit([t1_vex2, t1_sta2], [p2[9], p2[17]], 1)
Rp2 = lambda t: (
    np.polyval(p_Rp21, t)
    if t <= t1_vex2 else
    np.polyval(p_Rp22, t)
)

p_gamma21 = np.polyfit([t0_cor2, t1_vex2], [tilt_cor2, p2[11]], 1)
p_gamma22 = np.polyfit([t1_vex2, t1_sta2], [p2[11], p2[19]], 1)
gamma2 = lambda t: (
    np.polyval(p_gamma21, t)
    if t <= t1_vex2 else
    np.polyval(p_gamma22, t)
)

d0_cor3 = datetime(2011, 6, 4, 23, 56)
t0_cor3 = calendar.timegm(d0_cor3.timetuple())
latitude_cor3 = u.deg.to(u.rad, -2.0)
longitude_cor3 = u.deg.to(u.rad, 92.0)
toroidal_height_cor3 = u.R_sun.to(u.m, 16.5)
poloidal_height_cor3 = u.R_sun.to(u.m, 4.5)
half_width_cor3 = u.deg.to(u.rad, 30.0)
tilt_cor3 = u.deg.to(u.rad, 65.0)
flattening_cor3 = 0.3
pancaking_cor3 = u.deg.to(u.rad, 18.0)
d0_sta3 = datetime(2011, 6, 6, 16, 30)
d1_sta3 = datetime(2011, 6, 7, 1)
t0_sta3 = calendar.timegm(d0_sta3.timetuple())
t1_sta3 = calendar.timegm(d1_sta3.timetuple())

p3 = np.array([
    0.00000000e+00, 0.00000000e+00, 1.11320638e+06, 0.00000000e+00,
    1.87028467e+00, 9.96313673e-01, 0.00000000e+00, 0.00000000e+00,
    0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
    0.00000000e+00, 0.00000000e+00, 1.00000000e+14, 2.14172346e-01,
    2.03693608e+00, 9.78555131e+09, 6.98131701e-01, 5.39515444e-01,
    6.55698132e-01, 4.36332313e-01, 8.26308441e+13
])

p_theta3 = np.polyfit([t0_cor3, t1_sta3], [latitude_cor3, p3[15]], 1)
theta3 = lambda t: np.polyval(p_theta3, t)
p_phi3 = np.polyfit([t0_cor3, t1_sta3], [longitude_cor3, p3[16]], 1)
phi3 = lambda t: np.polyval(p_phi3, t)
Rt3 = lambda t: p3[2]*(t-t0_cor3)+toroidal_height_cor3
p_Rp3 = np.polyfit([t0_cor3, t1_sta3], [poloidal_height_cor3, p3[17]], 1)
Rp3 = lambda t: np.polyval(p_Rp3, t)
p_gamma3 = np.polyfit([t0_cor3, t1_sta3], [tilt_cor3, p3[19]], 1)
gamma3 = lambda t: np.polyval(p_gamma3, t)

t1 = np.linspace(t0_cor1, t1_vex1, 100)
t2 = np.linspace(t0_cor2, t1_sta2, 100)
t3 = np.linspace(t0_cor3, t1_sta3, 100)

# plt.figure()
# plt.plot(t1, [Rt1(x) for x in t1])
# plt.plot(t2, [Rt2(x) for x in t2])
# plt.plot(t3, [Rt3(x) for x in t3])
# plt.show()

# for t in np.linspace(t0_cor1+15*3600, t0_cor1+50*3600, 200):

t = t0_vex2

fr1 = FRi3D(theta1(t), phi1(t), Rt1(t), Rp1(t), half_width_cor1, gamma1(t), flattening_cor1, pancaking_cor1)
fr2 = FRi3D(theta2(t), phi2(t), Rt2(t), Rp2(t), half_width_cor2, gamma2(t), flattening_cor2, pancaking_cor2)
fr3 = FRi3D(theta3(t), phi3(t), Rt3(t), Rp3(t), half_width_cor3, gamma3(t), flattening_cor3, pancaking_cor3)

# fr1 = FRi3D(p1[7], p1[8], Rt1(t1_sta3), p1[9], p1[10], p1[11], p1[12], p1[13])
# fr2 = FRi3D(p2[15], p2[16], Rt2(t1_sta3), p2[17], p2[18], p2[19], p2[20], p2[21])
# fr3 = FRi3D(p3[15], p3[16], Rt3(t1_sta3), p3[17], p3[18], p3[19], p3[20], p3[21])

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d', adjustable='box', aspect=1.0)
ax.set_xlim(-1.5, 0.5)
ax.set_ylim(0.0, 2.0)
ax.set_zlim(-1.0, 1.0)
ax.view_init(45.0, 45.0)

x, y, z = fr1.shell()
x *= u.m.to(u.au)
y *= u.m.to(u.au)
z *= u.m.to(u.au)
ax.plot_wireframe(x, y, z, color=BLIND_PALETTE['vermillion'], alpha=0.4)

x, y, z = fr2.shell()
x *= u.m.to(u.au)
y *= u.m.to(u.au)
z *= u.m.to(u.au)
ax.plot_wireframe(x, y, z, color=BLIND_PALETTE['bluish-green'], alpha=0.4)

x, y, z = fr3.shell()
x *= u.m.to(u.au)
y *= u.m.to(u.au)
z *= u.m.to(u.au)
ax.plot_wireframe(x, y, z, color=BLIND_PALETTE['blue'], alpha=0.4)

ax.plot([-0.24780182], [0.20551227], [0.01735895], '.g', markersize=10)
ax.plot([-0.4918486], [0.5307865], [0.03558903], '.b', markersize=10)
ax.plot([-0.08090355], [0.94649878], [0.12191591], '.r', markersize=10)

# fig.savefig('CMExCMExCME_'+str(t)+'.png')

plt.show()


# CME 1

# MESSENGER:  0.0540540540541
# VEX:  0.027027027027 0.099474436912
# AVERAGE:  0.0601851726644
# SHARED toroidal_height speed =  1094.62947891
# SHARED sigma =  1.82910491845
# SHARED twist =  0.137251561394
# MESSENGER flux =  2.16703775611e+14
# VEX latitude =  -2.08207379051
# VEX longitude =  139.969561307
# VEX poloidal_height =  0.047094292214
# VEX half_width =  40.6753583782
# VEX tilt =  44.8938664755
# VEX flattening =  0.558357402407
# VEX pancaking =  21.8073158989
# VEX flux =  1.94095355935e+14
# [  0.00000000e+00   0.00000000e+00   1.09462948e+06   0.00000000e+00
#    1.82910492e+00   1.37251561e-01   2.16703776e+14  -3.63390429e-02
#    2.44292970e+00   7.04520584e+09   7.09918928e-01   7.83545784e-01
#    5.58357402e-01   3.80609463e-01   1.94095356e+14   0.00000000e+00
#    0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
#    0.00000000e+00   0.00000000e+00   0.00000000e+00]

# CME 2

# MESSENGER:  0.0
# VEX:  0.0 0.284802368595
# STEREO-A:  0.047619047619 0.463426890829 0.0175609666235
# AVERAGE:  0.135568212278
# SHARED toroidal_height speed =  1634.28420266
# SHARED toroidal_height speed =  1378.4377414
# SHARED sigma =  1.86551864579
# SHARED twist =  0.713969549189
# MESSENGER flux =  5.56341188364e+14
# VEX latitude =  11.571126533
# VEX longitude =  129.071341191
# VEX poloidal_height =  0.0696361207668
# VEX half_width =  20.4498224535
# VEX tilt =  -17.6414757564
# VEX flattening =  0.593706418683
# VEX pancaking =  39.7556405293
# VEX flux =  4.23256538291e+14
# STEREO-A latitude =  -19.966317885
# STEREO-A longitude =  98.4221567877
# STEREO-A poloidal_height =  0.0151102725599
# STEREO-A half_width =  38.6123371973
# STEREO-A tilt =  -45.6170221329
# STEREO-A flattening =  0.766754748488
# STEREO-A pancaking =  30.3765340348
# STEREO-A flux =  7.30233034896e+13
# [  0.00000000e+00   0.00000000e+00   1.63428420e+06   1.37843774e+06
#    1.86551865e+00   7.13969549e-01   5.56341188e+14   2.01954256e-01
#    2.25271987e+00   1.04174154e+10   3.56916733e-01  -3.07901837e-01
#    5.93706419e-01   6.93866823e-01   4.23256538e+14  -3.48477987e-01
#    1.71779069e+00   2.26046460e+09   6.73912416e-01  -7.96167231e-01
#    7.66754748e-01   5.30170534e-01   7.30233035e+13]

# CME 3

# STEREO-A:  0.0 0.0476390185064 0.0388334503763
# AVERAGE:  0.0288241562942
# SHARED toroidal_height speed =  1049.95046066
# SHARED sigma =  1.70085913751
# SHARED twist =  0.744155854282
# STEREO-A latitude =  9.11951553025
# STEREO-A longitude =  112.159584221
# STEREO-A poloidal_height =  0.0615651693267
# STEREO-A half_width =  37.432378156
# STEREO-A tilt =  27.5966515011
# STEREO-A flattening =  0.79893673596
# STEREO-A pancaking =  31.5372131593
# STEREO-A flux =  8.63176178802e+13
# [  0.00000000e+00   0.00000000e+00   1.04995046e+06   0.00000000e+00
#    1.70085914e+00   7.44155854e-01   0.00000000e+00   0.00000000e+00
#    0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
#    0.00000000e+00   0.00000000e+00   1.00000000e+14   1.59165572e-01
#    1.95755403e+00   9.21001824e+09   6.53318246e-01   4.81652431e-01
#    7.98936736e-01   5.50428207e-01   8.63176179e+13]