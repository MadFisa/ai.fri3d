
import numpy as np
from datetime import datetime
import calendar
from astropy import units as u
from matplotlib import pyplot as plt

d0_cor1 = datetime(2011, 6, 4, 8, 54)
t0_cor1 = calendar.timegm(d0_cor1.timetuple())
toroidal_height_cor1 = u.R_sun.to(u.m, 12.5)

p1 = [  
    4.10087695e-03, 1.53239612e+06, 1.09074414e+06, 0.00000000e+00,
    1.72208573e+00, 9.75235369e-02, 2.95738716e+14, 8.76460076e-03,
    2.43337280e+00, 6.85080642e+09, 6.75083121e-01, 7.78536041e-01,
    3.21601842e-01, 5.96513135e-01, 2.60208297e+14, 0.00000000e+00,
    0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
    0.00000000e+00, 0.00000000e+00, 0.00000000e+00
]

d0_cor2 = datetime(2011, 6, 4, 22, 54)
t0_cor2 = calendar.timegm(d0_cor2.timetuple())
toroidal_height_cor2 = u.R_sun.to(u.m, 12.0)

p2 = [  
    7.48802787e-02, 2.15873067e+06, 1.60718562e+06, 1.16841864e+06,
    1.78277985e+00, 4.17389768e-01, 9.30223126e+14, 2.69069427e-01,
    2.11733102e+00, 1.39415766e+10, 5.57617881e-01, 1.05192491e+00,
    8.67515533e-01, 5.93952746e-01, 4.27330784e+14, 3.18016032e-01,
    2.00965164e+00, 1.77926756e+09, 5.30139312e-01, 1.26932862e+00,
    7.25633433e-01, 6.43186839e-01, 7.04993500e+13
]

d0_cor3 = datetime(2011, 6, 4, 23, 56)
t0_cor3 = calendar.timegm(d0_cor3.timetuple())
toroidal_height_cor3 = u.R_sun.to(u.m, 7.5)

p3 = [  
    9.36787673e-03, 1.03308533e+06, 1.01328946e+06, 0.00000000e+00,
    1.86287442e+00, 5.45612614e-01, 0.00000000e+00, 0.00000000e+00,
    0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
    0.00000000e+00, 0.00000000e+00, 1.00000000e+14, 1.33368540e-01,
    1.91927534e+00, 9.07163638e+09, 5.83000319e-01, 5.31262222e-01,
    7.83716277e-01, 6.77584455e-01, 1.28830053e+14
]

di = datetime(2011, 6, 5, 11, 30)
# di = datetime(2011, 6, 5, 5)
ti = calendar.timegm(di.timetuple())

# Rt1 = lambda t: (
#     (p1[1]-p1[2])/p1[0]*(1.0-np.exp(-p1[0]*(t-t0_cor1)))+p1[2]*(t-t0_cor1)+
#     toroidal_height_cor1
# )

# Rt2 = lambda t: (
#     (p2[1]-p2[2])/p2[0]*(1.0-np.exp(-p2[0]*(t-t0_cor2)))+p2[2]*(t-t0_cor2)+
#     toroidal_height_cor2
#     if t <= ti else
#     (p2[1]-p2[2])/p2[0]*(1.0-np.exp(-p2[0]*(ti-t0_cor2)))+p2[2]*(ti-t0_cor2)+
#     toroidal_height_cor2+
#     p2[3]*(t-ti)
# )

# Rt3 = lambda t: (
#     (p3[1]-p3[2])/p3[0]*(1.0-np.exp(-p3[0]*(t-t0_cor3)))+p3[2]*(t-t0_cor3)+
#     toroidal_height_cor3
# )

# d0 = datetime(2011, 6, 4, 6)
# d1 = datetime(2011, 6, 7, 12)

# t1 = np.linspace(
#     t0_cor1,
#     calendar.timegm(d1.timetuple()),
#     100
# )

# t2 = np.linspace(
#     t0_cor2,
#     calendar.timegm(d1.timetuple()),
#     100
# )

# t3 = np.linspace(
#     t0_cor3,
#     calendar.timegm(d1.timetuple()),
#     100
# )

# plt.plot(t1, [Rt1(x) for x in t1], 'r')
# plt.plot(t2, [Rt2(x) for x in t2], 'g')
# plt.plot(t3, [Rt3(x) for x in t3], 'b')
# plt.show()

d0_cor = datetime(2011, 6, 4, 22, 54)
t0_cor = calendar.timegm(d0_cor.timetuple())
d0_mes = datetime(2011, 6, 5, 4, 40)
d1_mes = datetime(2011, 6, 5, 9, 29)
t0_mes = calendar.timegm(d0_mes.timetuple())
t1_mes = calendar.timegm(d1_mes.timetuple())

t1 = np.linspace(
    t0_cor,
    t1_mes,
    10
)
v1 = 1.66589259e+06

d0_sta = datetime(2011, 6, 6, 12, 25)
d1_sta = datetime(2011, 6, 6, 14, 10)
t0_sta = calendar.timegm(d0_sta.timetuple())
t1_sta = calendar.timegm(d1_sta.timetuple())

d0_vex = datetime(2011, 6, 5, 15, 30)
d1_vex = datetime(2011, 6, 5, 22, 30)
t0_vex = calendar.timegm(d0_vex.timetuple())
t1_vex = calendar.timegm(d1_vex.timetuple())

t2 = np.linspace(
    ti,
    t1_vex,
    10
)
v2 = 1.39337996e+06

t3 = np.linspace(
    t0_sta,
    t1_sta,
    10
)
v3 = 1.00778480e+06

toroidal_height_cor = u.R_sun.to(u.m, 12.0)




# plt.plot(t1, np.ones(t1.shape)*v1)
# plt.plot(t2, np.ones(t2.shape)*v2)
# plt.plot(t3, np.ones(t3.shape)*v3)
# plt.show()

Rt = lambda t: (
    toroidal_height_cor+v1*(t-t0_cor)
    if t <= ti else
        toroidal_height_cor+1.01717379e+06*(ti-t0_cor)+v3*(t-ti)
        if t >= t1_vex+3600*11.5 else
        toroidal_height_cor+v1*(ti-t0_cor)+650e3*(t-ti)
)

# t1_vex+3600*5 = datetime(2011, 6, 6, 10)

t = np.linspace(
    t0_cor,
    t1_sta,
    100
)

plt.plot(t, [Rt(x) for x in t])




Rt1 = lambda t: 1.09284773e+06*(t-t0_cor1)+toroidal_height_cor1
t1 = np.linspace(
    calendar.timegm(datetime(2011, 6, 4, 8, 54).timetuple()),
    calendar.timegm(datetime(2011, 6, 5, 11, 50).timetuple()),
    100
)
plt.plot(t1, [Rt1(x) for x in t1])



Rt3 = lambda t: 1.01519278e+06*(t-t0_cor3)+toroidal_height_cor3
t3 = np.linspace(
    calendar.timegm(datetime(2011, 6, 4, 23, 56).timetuple()),
    calendar.timegm(datetime(2011, 6, 7, 1).timetuple()),
    100
)
plt.plot(t3, [Rt3(x) for x in t3])



plt.show()




# CME 1

# MESSENGER:  0.0540540540541
# VEX:  0.027027027027 0.0257009934907
# AVERAGE:  0.0355940248573
# SHARED toroidal_height decay =  0.00410087695021
# SHARED toroidal_height speed =  1532.39611789
# SHARED toroidal_height speed =  1090.74414456
# SHARED sigma =  1.72208572816
# SHARED twist =  0.0975235369307
# MESSENGER flux =  2.95738715957e+14
# VEX latitude =  0.50217463248
# VEX longitude =  139.421991365
# VEX poloidal_height =  0.0457948123976
# VEX half_width =  38.6794136686
# VEX tilt =  44.6068293323
# VEX flattening =  0.321601842477
# VEX pancaking =  34.177685088
# VEX flux =  2.60208296555e+14
# [  4.10087695e-03   1.53239612e+06   1.09074414e+06   0.00000000e+00
#    1.72208573e+00   9.75235369e-02   2.95738716e+14   8.76460076e-03
#    2.43337280e+00   6.85080642e+09   6.75083121e-01   7.78536041e-01
#    3.21601842e-01   5.96513135e-01   2.60208297e+14   0.00000000e+00
#    0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
#    0.00000000e+00   0.00000000e+00   0.00000000e+00]

# CME 2

# MESSENGER:  0.0714285714286
# VEX:  0.0238095238095 0.147885690074
# STEREO-A:  0.047619047619 0.256383357026 0.0764026480497
# AVERAGE:  0.103921473001
# SHARED toroidal_height decay =  0.074880278671
# SHARED toroidal_height speed =  2158.73067257
# SHARED toroidal_height speed =  1607.18562046
# SHARED toroidal_height speed =  1168.41863675
# SHARED sigma =  1.78277984837
# SHARED twist =  0.417389768315
# MESSENGER flux =  9.30223125511e+14
# VEX latitude =  15.4165425873
# VEX longitude =  121.314131546
# VEX poloidal_height =  0.0931936834373
# VEX half_width =  31.9491511754
# VEX tilt =  60.2708575138
# VEX flattening =  0.867515533104
# VEX pancaking =  34.030985596
# VEX flux =  4.27330783546e+14
# STEREO-A latitude =  18.2209764426
# STEREO-A longitude =  115.144557025
# STEREO-A poloidal_height =  0.0118936690065
# STEREO-A half_width =  30.3747451069
# STEREO-A tilt =  72.7271727662
# STEREO-A flattening =  0.725633433038
# STEREO-A pancaking =  36.8518912917
# STEREO-A flux =  7.04993499698e+13
# [  7.48802787e-02   2.15873067e+06   1.60718562e+06   1.16841864e+06
#    1.78277985e+00   4.17389768e-01   9.30223126e+14   2.69069427e-01
#    2.11733102e+00   1.39415766e+10   5.57617881e-01   1.05192491e+00
#    8.67515533e-01   5.93952746e-01   4.27330784e+14   3.18016032e-01
#    2.00965164e+00   1.77926756e+09   5.30139312e-01   1.26932862e+00
#    7.25633433e-01   6.43186839e-01   7.04993500e+13]

# CME 3

# STEREO-A:  0.0 0.0121611550752 0.039531843147
# AVERAGE:  0.0172309994074
# SHARED toroidal_height decay =  0.0093678767328
# SHARED toroidal_height speed =  1033.08532747
# SHARED toroidal_height speed =  1013.28945534
# SHARED sigma =  1.8628744185
# SHARED twist =  0.545612614115
# STEREO-A latitude =  7.64145445659
# STEREO-A longitude =  109.966376822
# STEREO-A poloidal_height =  0.0606401436962
# STEREO-A half_width =  33.4034577557
# STEREO-A tilt =  30.4390831437
# STEREO-A flattening =  0.78371627718
# STEREO-A pancaking =  38.8227295464
# STEREO-A flux =  1.28830052586e+14
# [  9.36787673e-03   1.03308533e+06   1.01328946e+06   0.00000000e+00
#    1.86287442e+00   5.45612614e-01   0.00000000e+00   0.00000000e+00
#    0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
#    0.00000000e+00   0.00000000e+00   1.00000000e+14   1.33368540e-01
#    1.91927534e+00   9.07163638e+09   5.83000319e-01   5.31262222e-01
#    7.83716277e-01   6.77584455e-01   1.28830053e+14]
