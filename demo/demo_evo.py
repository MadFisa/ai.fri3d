
from ai.fri3d import Evolution
from astropy import units as u
from matplotlib import pyplot as plt
import numpy as np

u.nT = u.def_unit('nT', 1e-9*u.T)

def demo_evo():
    evo = Evolution(
        latitude=lambda t: u.deg.to(u.rad, -5.0),
        longitude=lambda t: u.deg.to(u.rad, 0.0),
        tilt=lambda t: u.deg.to(u.rad, 45.0),
        twist=lambda t: 0.5,
        polarity=1.0,
        chirality=1.0
    )
    b = evo.insitu(
        np.linspace(0.0, 24.0*3600.0*2.0, 100), 
        u.au.to(u.m, 1.0), 
        u.au.to(u.m, 0.0), 
        u.au.to(u.m, 0.0)
    )*u.T.to(u.nT)

    fig = plt.figure()
    plt.plot(np.sqrt(b[:,0]**2+b[:,1]**2+b[:,2]**2), 'k')
    plt.plot(b[:,0], 'r')
    plt.plot(b[:,1], 'g')
    plt.plot(b[:,2], 'b')
    plt.show()

demo_evo()
