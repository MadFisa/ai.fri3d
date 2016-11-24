
from ai.fri3d import Evolution, FRi3D
from astropy import units as u
from astropy import constants as c
from matplotlib import pyplot as plt
import numpy as np
from ai.shared.color import BLIND_PALETTE

u.nT = u.def_unit('nT', 1e-9*u.T)

def demo_evo():
    evo = Evolution(
        latitude=lambda t: u.deg.to(u.rad, 0.0),
        longitude=lambda t: u.deg.to(u.rad, 0.0),
        poloidal_height=lambda t: u.au.to(u.m, 0.15)+50e3*t,
        half_width=lambda t: u.deg.to(u.rad, 40.0),
        tilt=lambda t: u.deg.to(u.rad, 0.0),
        flattening=lambda t: 0.5,
        pancaking=lambda t: u.deg.to(u.rad, 30.0),
        twist=lambda t: 3.0,
        polarity=1.0,
        chirality=1.0
    )
    b = evo.insitu(
        np.linspace(0.0, 24.0*3600.0*2.0, 100), 
        u.au.to(u.m, 1.0), 
        u.au.to(u.m, 0.0), 
        u.au.to(u.m, 0.0)
    )*u.T.to(u.nT)

    nonzero_indices = np.nonzero(np.sqrt(b[:,0]**2+b[:,1]**2+b[:,2]**2))[0]

    if nonzero_indices.size >= 2:
        b = b[nonzero_indices[0]:nonzero_indices[-1]+1,:]

        fig = plt.figure()
        plt.plot(np.sqrt(b[:,0]**2+b[:,1]**2+b[:,2]**2), 'k')
        plt.plot(b[:,0], 'r')
        plt.plot(b[:,1], 'g')
        plt.plot(b[:,2], 'b')
        
        plt.axis('tight')
        plt.ylim([-12,12])

        plt.show()

def demo_compare():
    fr = FRi3D(
        latitude=u.deg.to(u.rad, -20.0), 
        longitude=u.deg.to(u.rad, 0.0), 
        toroidal_height=u.au.to(u.m, 1.0), 
        poloidal_height=u.au.to(u.m, 0.15), 
        half_width=u.deg.to(u.rad, 40.0), 
        tilt=u.deg.to(u.rad, 0.0), 
        flattening=0.5, 
        pancaking=u.deg.to(u.rad, 30.0), 
        skew=u.deg.to(u.rad, 0.0), 
        twist=3.0/c.au.value, 
        # twist=3.0, 
        flux=5e14,
        sigma=1.0,
        polarity=1.0,
        chirality=1.0,
    )

    b = fr.data(
        x=u.au.to(u.m)*np.linspace(1.15, 0.85, 57),
        y=u.au.to(u.m)*np.zeros(57),
        z=u.au.to(u.m)*np.zeros(57)
    )*u.T.to(u.nT)

    fig = plt.figure()
    plt.subplots_adjust(hspace=0.001)
    ax1 = fig.add_subplot(3,1,1)
    ax1.plot(np.sqrt(b[:,0]**2+b[:,1]**2+b[:,2]**2), 'k', label='B', linewidth=2)
    ax1.plot(b[:,0], color=BLIND_PALETTE['vermillion'], label='Bx', linewidth=2)
    ax1.plot(b[:,1], color=BLIND_PALETTE['bluish-green'], label='By', linewidth=2)
    ax1.plot(b[:,2], color=BLIND_PALETTE['blue'], label='Bz', linewidth=2)

    plt.legend()

    ax1.set_ylabel('B [nT]')

    evo = Evolution(
        latitude=lambda t: u.deg.to(u.rad, -20.0),
        longitude=lambda t: u.deg.to(u.rad, 20.0),
        poloidal_height=lambda t: u.au.to(u.m, 0.15),#+50e3*t,
        half_width=lambda t: u.deg.to(u.rad, 40.0),
        tilt=lambda t: u.deg.to(u.rad, 0.0),
        flattening=lambda t: 0.5,
        pancaking=lambda t: u.deg.to(u.rad, 30.0),
        twist=lambda t: 3.0/c.au.value,
        # twist=lambda t: 3.0,
        flux=lambda t: 5e14,
        sigma=lambda t: 1.0,
        polarity=1.0,
        chirality=1.0
    )
    b = evo.insitu(
        np.linspace(0.0, 24.0*3600.0*3.0, 100), 
        u.au.to(u.m, 1.0), 
        u.au.to(u.m, 0.0), 
        u.au.to(u.m, 0.0)
    )*u.T.to(u.nT)

    # nonzero_indices = np.nonzero(np.sqrt(b[:,0]**2+b[:,1]**2+b[:,2]**2))[0]
    nonzero_indices = np.where(np.isfinite(
        np.sqrt(b[:,0]**2+b[:,1]**2+b[:,2]**2)
    ))[0]

    if nonzero_indices.size >= 2:
        b = b[nonzero_indices[0]:nonzero_indices[-1]+1,:]
        print(b.shape)
        ax2 = fig.add_subplot(3,1,2,sharex=ax1)
        ax2.plot(np.sqrt(b[:,0]**2+b[:,1]**2+b[:,2]**2), 'k', linewidth=2)
        ax2.plot(b[:,0], color=BLIND_PALETTE['vermillion'], label='Bx', linewidth=2)
        ax2.plot(b[:,1], color=BLIND_PALETTE['bluish-green'], label='By', linewidth=2)
        ax2.plot(b[:,2], color=BLIND_PALETTE['blue'], label='Bz', linewidth=2)
        ax2.set_ylabel('B [nT]')

    evo = Evolution(
        latitude=lambda t: u.deg.to(u.rad, -20.0),
        longitude=lambda t: u.deg.to(u.rad, 0.0),
        poloidal_height=lambda t: u.au.to(u.m, 0.15)+50e3*t,
        half_width=lambda t: u.deg.to(u.rad, 40.0),
        tilt=lambda t: u.deg.to(u.rad, 0.0),
        flattening=lambda t: 0.5,
        pancaking=lambda t: u.deg.to(u.rad, 30.0),
        twist=lambda t: 3.0/c.au.value,
        # twist=lambda t: 3.0,
        flux=lambda t: 5e14,
        sigma=lambda t: 1.0,
        polarity=1.0,
        chirality=1.0
    )
    b = evo.insitu(
        np.linspace(0.0, 24.0*3600.0*2.0, 100), 
        u.au.to(u.m, 1.0), 
        u.au.to(u.m, 0.0), 
        u.au.to(u.m, 0.0)
    )*u.T.to(u.nT)

    # nonzero_indices = np.nonzero(np.sqrt(b[:,0]**2+b[:,1]**2+b[:,2]**2))[0]
    nonzero_indices = np.where(np.isfinite(
        np.sqrt(b[:,0]**2+b[:,1]**2+b[:,2]**2)
    ))[0]

    if nonzero_indices.size >= 2:
        b = b[nonzero_indices[0]:nonzero_indices[-1]+1,:]

        ax3 = fig.add_subplot(3,1,3,sharex=ax1)
        ax3.plot(np.sqrt(b[:,0]**2+b[:,1]**2+b[:,2]**2), 'k', linewidth=2)
        ax3.plot(b[:,0], color=BLIND_PALETTE['vermillion'], label='Bx', linewidth=2)
        ax3.plot(b[:,1], color=BLIND_PALETTE['bluish-green'], label='By', linewidth=2)
        ax3.plot(b[:,2], color=BLIND_PALETTE['blue'], label='Bz', linewidth=2)

        ax3.set_xlabel('time [arb. units]')
        ax3.set_ylabel('B [nT]')

    xticklabels = ax1.get_xticklabels() + ax2.get_xticklabels()
    plt.setp(xticklabels, visible=False)

    plt.show()

demo_compare()

# demo_evo()
