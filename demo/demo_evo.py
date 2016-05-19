
from ai.fri3d import Evolution
from astropy import units as u
from matplotlib import pyplot as plt
import numpy as np

u.nT = u.def_unit('nT', 1e-9*u.T)

def demo_evo():
    evo = Evolution()
    b = evo.insitu(
        np.linspace(0.0, 24.0*3600.0*2.0, 100), 
        u.au.to(u.m, 1.0), 
        u.au.to(u.m, 0.0), 
        u.au.to(u.m, 0.0)
    )*u.T.to(u.nT)

    fig = plt.figure()
    plt.plot(b[:,0], 'k')
    plt.plot(b[:,1], 'r')
    plt.plot(b[:,2], 'g')
    plt.plot(b[:,3], 'b')
    plt.show()

demo_evo()
