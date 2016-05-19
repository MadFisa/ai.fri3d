
import numpy as np
from astropy import units as u
from ai.FRi3D import FRi3D
from matplotlib import pyplot as plt

u.nT = u.def_unit('nT', 1e-9*u.T)

def demo_data(
    latitude=u.deg.to(u.rad, 0.0), 
    longitude=u.deg.to(u.rad, 0.0), 
    toroidal_height=u.au.to(u.m, 1.0), 
    poloidal_height=u.au.to(u.m, 0.2), 
    half_width=u.deg.to(u.rad, 40.0), 
    tilt=u.deg.to(u.rad, 0.0), 
    flattening=0.5, 
    pancaking=u.deg.to(u.rad, 20.0), 
    skew=u.deg.to(u.rad, 0.0), 
    twist=5.0, 
    flux=5e14,
    polarity=1.0,
    chirality=1.0,
    x=u.au.to(u.m)*np.linspace(1.2, 0.8, 100),
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

    print(fr.data(
        x=u.au.to(u.m, 1.0), 
        y=u.au.to(u.m, 0.0), 
        z=u.au.to(u.m, 0.0)
    )*u.T.to(u.nT))

    b = fr.data(x, y, z)*u.T.to(u.nT)

    fig = plt.figure()
    plt.plot(b[:,0], 'k')
    plt.plot(b[:,1], 'r')
    plt.plot(b[:,2], 'g')
    plt.plot(b[:,3], 'b')
    plt.show()

demo_data()
