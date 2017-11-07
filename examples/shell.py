
import numpy as np
from ai.fri3d import FRi3D
from astropy import units as u
from ai.shared.color import BLIND_PALETTE
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import proj3d

def example_shell(
        latitude=u.deg.to(u.rad, 0.0),
        longitude=u.deg.to(u.rad, 0.0),
        toroidal_height=u.au.to(u.m, 1.0),
        poloidal_height=u.au.to(u.m, 0.2),
        half_width=u.deg.to(u.rad, 40.0),
        tilt=u.deg.to(u.rad, 0.0),
        flattening=0.5,
        pancaking=u.deg.to(u.rad, 20.0),
        skew=u.deg.to(u.rad, 10.0)):
    fr = FRi3D(
        latitude=latitude,
        longitude=longitude,
        toroidal_height=toroidal_height,
        poloidal_height=poloidal_height,
        half_width=half_width,
        tilt=tilt,
        flattening=flattening,
        pancaking=pancaking,
        skew=skew
    )

    # phi = np.linspace(-half_width, half_width, 100)
    # plt.plot(phi, np.array([fr._vanilla_axis_length(x) for x in phi]))
    # plt.show()

    # phi = np.linspace(-half_width, half_width, 100)
    # plt.plot(phi, np.array([fr._vanilla_axis_length(x) for x in phi]))
    # plt.show()

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d', adjustable='box', aspect=1.0)
    x, y, z = fr.shell()
    x *= u.m.to(u.au)
    y *= u.m.to(u.au)
    z *= u.m.to(u.au)
    ax.plot_wireframe(x, y, z, color=BLIND_PALETTE['blue'], alpha=0.4)

    ax.set_xlabel('X [AU]')
    ax.set_ylabel('Y [AU]')
    ax.set_zlabel('Z [AU]')

    plt.show()

example_shell()
