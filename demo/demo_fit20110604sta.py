from ai.fri3d import FRi3D
from astropy import units as u
from ai.shared.color import BLIND_PALETTE
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import proj3d
import calendar
from datetime import datetime
import time
import numpy as np

def demo_shells():
    
# 0.297211009311 1307280600.0 [ -1.41713590e-01   1.78074116e+00   1.03461028e+06   7.47989354e+10
#    6.85062459e+09   7.21880038e-01  -6.97403705e-01   4.01758992e-01
#    6.75686178e-01   2.73534510e+00   4.17596950e+14]

# 0.302630741777 1307285100.0 [  1.84632438e-02   1.75403112e+00   9.78437981e+05   7.47989354e+10
#    5.10609750e+09   6.89402283e-01  -8.64455026e-01   3.06020000e-01
#    5.21643254e-01   1.60766417e+00   2.43052878e+14]

    fr1 = FRi3D(
        # latitude=1.84632438e-02, 
        latitude=u.deg.to(u.rad, 5.0),
        # longitude=1.75403112e+00, 
        longitude=u.deg.to(u.rad, 120.0), 
        # toroidal_height=u.au.to(u.m, 0.74),
        toroidal_height=u.au.to(u.m, 1.2),
        # toroidal_height=np.polyval(
        #     [9.78437981e+05, u.au.to(u.m, 0.5)], 
        #     time.mktime(t.timetuple())-(1307285100.0+10800)
        # ),
        poloidal_height=5.10609750e+09, 
        # half_width=6.89402283e-01, 
        half_width=u.deg.to(u.rad, 44),
        tilt=-8.64455026e-01, 
        flattening=3.06020000e-01, 
        pancaking=5.21643254e-01, 
        skew=0.0
    )
    print(fr1.toroidal_height)

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d', adjustable='box', aspect=1.0)
    x, y, z = fr1.shell()
    x *= u.m.to(u.au)
    y *= u.m.to(u.au)
    z *= u.m.to(u.au)
    ax.plot_wireframe(x, y, z, color=BLIND_PALETTE['blue'], alpha=0.4)

# 0.238447116448 1307287200.0 [  2.92256175e-01   2.13357924e+00  -3.85845146e-01   1.29818672e+06
#    7.47989354e+10   1.09761581e+10   7.50550756e-01   4.72097023e-01
#    4.70745851e-01   4.78821828e-01   1.04345531e+00   1.09514269e+14]

    fr2 = FRi3D(
        latitude=2.92256175e-01, 
        longitude=2.13357924e+00, 
        toroidal_height=u.au.to(u.m, 1.01),
        # toroidal_height=u.au.to(u.m, 0.45),
        # toroidal_height=np.polyval(
        #     [-3.85845146e-01, 1.29818672e+06, u.au.to(u.m, 0.5)],
        #     time.mktime(t.timetuple())-(1307286600.0+10800)
        # ),
        poloidal_height=1.09761581e+10, 
        half_width=7.50550756e-01, 
        tilt=4.72097023e-01, 
        flattening=4.70745851e-01, 
        pancaking=4.78821828e-01, 
        skew=0.0
    )

    print(fr2.toroidal_height)

    x, y, z = fr2.shell()
    x *= u.m.to(u.au)
    y *= u.m.to(u.au)
    z *= u.m.to(u.au)
    ax.plot_wireframe(x, y, z, color=BLIND_PALETTE['vermillion'], alpha=0.4)

    ax.set_xlabel('X [AU]')
    ax.set_ylabel('Y [AU]')
    ax.set_zlabel('Z [AU]')

    plt.show()

demo_shells()

