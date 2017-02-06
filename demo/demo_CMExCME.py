
"""

CME1

20110604_073900
theta       27.0
phi         132.0
Rt          6.0
Rp          2.0
thetaHW     40.0
gamma       -30.0
n           0.3
thetaP      18.0

20110604_082400
theta       34.0
phi         132.0
Rt          9.0
Rp          3.0
thetaHW     42.0
gamma       -35.0
n           0.3
thetaP      18.0

20110604_085400
theta       34.0
phi         130.0
Rt          12.0
Rp          3.0
thetaHW     44.0
gamma       -35.0
n           0.3
thetaP      18.0

MES
MES_RADIUS_IN_KM_HEEQ: 48513521.991095826,
MES_LATITUDE_IN_RADIANS_HEEQ: 0.052753889393141892,
MES_LONGITUDE_IN_RADIANS_HEEQ: 2.4105646887741350

VEX
datetime(2011, 6, 5, 8, 45), datetime(2011, 6, 5, 11, 50)
VEX_RADIUS_IN_KM_HEEQ: 108388449.07699333,
VEX_LATITUDE_IN_RADIANS_HEEQ: 0.049570023397835422,
VEX_LONGITUDE_IN_RADIANS_HEEQ: 2.3138245305617384

STA
0.302630741777 1307285100.0 [  1.84632438e-02   1.75403112e+00   9.78437981e+05   7.47989354e+10
   5.10609750e+09   6.89402283e-01  -8.64455026e-01   3.06020000e-01
   5.21643254e-01   1.60766417e+00   2.43052878e+14]
theta       1.05786595
phi         100.49858031
Vt          978.437981
Rp          0.0341321535935471
thetaHW     39.4998412
gamma       -49.52962457
n           0.306020000
thetaP      29.887956865670798
tau         1.60766417
F           2.43052878e+14

STA_RADIUS_IN_KM_HEEQ: 143273070.16038799,
STA_LATITUDE_IN_RADIANS_HEEQ: 0.12770683697430035,
STA_LONGITUDE_IN_RADIANS_HEEQ: 1.6554713871518154

CME2

20110604_222400
theta       20.0
phi         125.0
Rt          7.0
Rp          2.0
thetaHW     30.0
gamma       35.0
n           0.4
thetaP      25.0

20110604_225400
theta       22.0
phi         125.0
Rt          12.0
Rp          4.0
thetaHW     35.0
gamma       35.0
n           0.4
thetaP      30.0

MES
MES_RADIUS_IN_KM_HEEQ: 48209360.141018368,
MES_LATITUDE_IN_RADIANS_HEEQ: 0.053992672618883394,
MES_LONGITUDE_IN_RADIANS_HEEQ: 2.4524511903856760

VEX
datetime(2011, 6, 5, 15, 30), datetime(2011, 6, 5, 22, 30)
VEX_RADIUS_IN_KM_HEEQ: 108385426.78194684,
VEX_LATITUDE_IN_RADIANS_HEEQ: 0.049121164536206213,
VEX_LONGITUDE_IN_RADIANS_HEEQ: 2.3181389148256630

STA
0.238447116448 1307287200.0 [  2.92256175e-01   2.13357924e+00  -3.85845146e-01   1.29818672e+06
   7.47989354e+10   1.09761581e+10   7.50550756e-01   4.72097023e-01
   4.70745851e-01   4.78821828e-01   1.04345531e+00   1.09514269e+14]
theta       16.74504536
phi         122.24508571
at          -0.385845146
! Vt        1298.18672
Rp          0.07337108508724248
thetaHW     43.00339063
gamma       27.04916694
n           0.470745851
thetaP      27.434469883139027
tau         1.04345531
F           1.09514269e+14

STA_RADIUS_IN_KM_HEEQ: 143276200.34338978,
STA_LATITUDE_IN_RADIANS_HEEQ: 0.12763937523286392,
STA_LONGITUDE_IN_RADIANS_HEEQ: 1.6560867488947024

"""

import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
import time
from astropy import units as u
from ai.shared import cs
from ai.fri3d import FRi3D

# # MES
# fr1 = FRi3D(
#     # latitude=1.84632438e-02, 
#     latitude=u.deg.to(u.rad, 20.0), 
#     # longitude=1.75403112e+00, 
#     longitude=u.deg.to(u.rad, 130.0), 
#     toroidal_height=u.au.to(u.m, 0.3),
#     poloidal_height=5.10609750e+09, 
#     half_width=6.89402283e-01, 
#     tilt=-8.64455026e-01, 
#     flattening=3.06020000e-01, 
#     pancaking=5.21643254e-01, 
#     skew=0.0
# )
# for r in np.linspace(0.3,0.6,100):
#     fr1.toroidal_height = u.au.to(u.m, r)
#     x, y, z = cs.sp2cart(
#         u.km.to(u.m, 48513521.991095826),
#         0.052753889393141892,
#         2.4105646887741350
#     )
#     print(r, fr1.data(x, y, z))

# VEX
# fr1 = FRi3D(
#     latitude=1.84632438e-02, 
#     # latitude=u.deg.to(u.rad, 10.0), 
#     # longitude=1.75403112e+00, 
#     longitude=u.deg.to(u.rad, 130.0), 
#     toroidal_height=u.au.to(u.m, 0.5),
#     poloidal_height=5.10609750e+09, 
#     half_width=6.89402283e-01, 
#     # tilt=-8.64455026e-01, 
#     tilt=-6.97403705e-01,
#     flattening=3.06020000e-01, 
#     pancaking=5.21643254e-01, 
#     skew=0.0
# )
# for r in np.linspace(0.5,0.9,100):
#     fr1.toroidal_height = u.au.to(u.m, r)
#     x, y, z = cs.sp2cart(
#         u.km.to(u.m, 108388449.07699333),
#         0.049570023397835422,
#         2.3138245305617384
#     )
#     print(r, fr1.data(x, y, z))

# STA
# fr1 = FRi3D(
#     latitude=1.84632438e-02, 
#     # longitude=1.75403112e+00, 
#     longitude=u.deg.to(u.rad, 130.0), 
#     toroidal_height=u.au.to(u.m, 0.5),
#     poloidal_height=5.10609750e+09, 
#     half_width=6.89402283e-01, 
#     tilt=-8.64455026e-01, 
#     flattening=3.06020000e-01, 
#     pancaking=5.21643254e-01, 
#     skew=0.0
# )
# for r in np.linspace(0.8,1.4,100):
#     fr1.toroidal_height = u.au.to(u.m, r)
#     x, y, z = cs.sp2cart(
#         u.km.to(u.m, 143273070.16038799),
#         0.12770683697430035,
#         1.6554713871518154
#     )
#     print(r, fr1.data(x, y, z))

# MES
# fr2 = FRi3D(
#     latitude=2.92256175e-01, 
#     longitude=2.13357924e+00, 
#     toroidal_height=u.au.to(u.m, 0.3),
#     poloidal_height=1.09761581e+10, 
#     half_width=7.50550756e-01, 
#     tilt=4.72097023e-01, 
#     flattening=4.70745851e-01, 
#     pancaking=4.78821828e-01, 
#     skew=0.0
# )
# for r in np.linspace(0.2,0.6,100):
#     fr2.toroidal_height = u.au.to(u.m, r)
#     x, y, z = cs.sp2cart(
#         u.km.to(u.m, 48209360.141018368),
#         0.053992672618883394,
#         2.4524511903856760
#     )
#     print(r, fr2.data(x, y, z))

# VEX
# fr2 = FRi3D(
#     latitude=2.92256175e-01, 
#     longitude=2.13357924e+00, 
#     toroidal_height=u.au.to(u.m, 0.5),
#     poloidal_height=1.09761581e+10, 
#     half_width=7.50550756e-01, 
#     tilt=4.72097023e-01, 
#     flattening=4.70745851e-01, 
#     pancaking=4.78821828e-01, 
#     skew=0.0
# )
# for r in np.linspace(0.5,0.9,100):
#     fr2.toroidal_height = u.au.to(u.m, r)
#     x, y, z = cs.sp2cart(
#         u.km.to(u.m, 108385426.78194684),
#         0.049121164536206213,
#         2.3181389148256630
#     )
#     print(r, fr2.data(x, y, z))

# STA
# fr2 = FRi3D(
#     latitude=2.92256175e-01, 
#     longitude=2.13357924e+00, 
#     toroidal_height=u.au.to(u.m, 0.5),
#     poloidal_height=1.09761581e+10, 
#     half_width=7.50550756e-01, 
#     tilt=4.72097023e-01, 
#     flattening=4.70745851e-01, 
#     pancaking=4.78821828e-01, 
#     skew=0.0
# )
# for r in np.linspace(0.8,1.4,100):
#     fr2.toroidal_height = u.au.to(u.m, r)
#     x, y, z = cs.sp2cart(
#         u.km.to(u.m, 143276200.34338978),
#         0.12763937523286392,
#         1.6560867488947024
#     )
#     print(r, fr2.data(x, y, z))




def profiles():

    R1 = np.array([
        u.R_sun.to(u.au, 1.0),
        u.R_sun.to(u.au, 6.0), 
        u.R_sun.to(u.au, 9.0), 
        u.R_sun.to(u.au, 12.0), 
        0.354,
        0.725,
        1.266,
        # u.m.to(
        #     u.au, 
        #     np.polyval(
        #         [9.78437981e+05, u.au.to(u.m, 0.5)], 
        #         time.mktime(datetime(2011, 6, 6, 13, 30).timetuple())-(1307285100.0+10800)
        #     )
        # ),
    ])
    
    D1 = np.array([
        datetime(2011, 6, 4, 6, 45),
        datetime(2011, 6, 4, 7, 39),
        datetime(2011, 6, 4, 8, 24),
        datetime(2011, 6, 4, 8, 54),
        datetime(2011, 6, 4, 20),
        datetime(2011, 6, 5, 10, 20),
        datetime(2011, 6, 6, 13, 30),
    ])

    R2 = np.array([
        u.R_sun.to(u.au, 1.0),
        u.R_sun.to(u.au, 7.0),
        u.R_sun.to(u.au, 12.0),
        0.337,
        0.722,
        # 1.25,
        u.m.to(
            u.au, 
            np.polyval(
                [-3.85845146e-01, 1.29818672e+06, u.au.to(u.m, 0.5)],
                time.mktime(datetime(2011, 6, 6, 21).timetuple())-(1307286600.0+10800)
            )
        ),
    ])

    D2 = np.array([
        datetime(2011, 6, 4, 21, 45),
        datetime(2011, 6, 4, 22, 24),
        datetime(2011, 6, 4, 22, 54),
        datetime(2011, 6, 5, 8),
        datetime(2011, 6, 5, 19),
        datetime(2011, 6, 6, 21),
    ])

    plt.plot(D1, R1, D2, R2, linewidth=3.0)
    plt.axvline(datetime(2011, 6, 5, 11), c='red', lw=3.0, ls='dashed')
    plt.show()

profiles()

