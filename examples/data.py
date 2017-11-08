
import numpy as np
from astropy import units as u
from astropy import constants as c
from ai.fri3d import FRi3D
from matplotlib import pyplot as plt

u.nT = u.def_unit('nT', 1e-9*u.T)

def demo_data(
    latitude=u.deg.to(u.rad, 5.0), 
    longitude=u.deg.to(u.rad, 10.0), 
    toroidal_height=u.au.to(u.m, 1.0), 
    poloidal_height=u.au.to(u.m, 0.15), 
    half_width=u.deg.to(u.rad, 40.0), 
    tilt=u.deg.to(u.rad, 0.0), 
    flattening=0.5, 
    pancaking=u.deg.to(u.rad, 30.0), 
    skew=u.deg.to(u.rad, 0.0), 
    twist=3.0/c.au.value, 
    flux=5e14,
    polarity=1.0,
    chirality=1.0,
    x=u.au.to(u.m)*np.linspace(1.3, 0.7, 100),
    y=u.au.to(u.m)*np.zeros(100),
    z=u.au.to(u.m)*np.zeros(100)):

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
        polarity=polarity,
        chirality=chirality
    )

    b, _ = fr.data(x, y, z)
    b = u.T.to(u.nT, b)

    fig = plt.figure()
    plt.plot(np.sqrt(b[:,0]**2+b[:,1]**2+b[:,2]**2), 'k')
    plt.plot(b[:,0], 'r')
    plt.plot(b[:,1], 'g')
    plt.plot(b[:,2], 'b')

    plt.axis('tight')
    # plt.ylim([-12,12])

    plt.xlabel('time [arb. units]')
    plt.ylabel('B [nT]')



    plt.show()

demo_data()
