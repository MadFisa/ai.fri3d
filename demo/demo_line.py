
from ai.fri3d import FRi3D
import numpy as np
from astropy import units as u
from ai.shared.color import BLIND_PALETTE
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection

u.nT = u.def_unit('nT', 1e-9*u.T)

def demo_line(
    latitude=u.deg.to(u.rad, 0.0), 
    longitude=u.deg.to(u.rad, 0.0), 
    toroidal_height=u.au.to(u.m, 1.0), 
    poloidal_height=u.au.to(u.m, 0.15), 
    half_width=u.deg.to(u.rad, 40.0), 
    tilt=u.deg.to(u.rad, 0.0), 
    flattening=0.5, 
    pancaking=u.deg.to(u.rad, 30.0), 
    skew=u.deg.to(u.rad, 0.0), 
    twist=3.0, 
    flux=5e14,
    polarity=1.0,
    chirality=1.0):
    
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

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d', adjustable='box', aspect=1.0)
    x, y, z = fr.shell()
    x *= u.m.to(u.au)
    y *= u.m.to(u.au)
    z *= u.m.to(u.au)
    ax.plot_wireframe(x, y, z, color=BLIND_PALETTE['blue'], alpha=0.1)

    _, _, _, b = fr.line(1.0, 0.0, s=0.5)
    bmin = u.T.to(u.nT, b)
    _, _, _, b = fr.line(0.0, 0.0, s=0.49)
    bmax = u.T.to(u.nT, b)


    for i in range(1000):
        r = np.random.uniform(0.0, 1.0)
        phi = np.random.uniform(0.0, np.pi*2.0)
        x, y, z, b = fr.line(r, phi, s=np.linspace(0.49, 0.51, 10))
        x *= u.m.to(u.au)
        y *= u.m.to(u.au)
        z *= u.m.to(u.au)
        b *= u.T.to(u.nT)
        points = np.array([x, y, z]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        c = (b-bmin)/(bmax-bmin)
        lc = Line3DCollection(
            segments, 
            colors=plt.cm.magma(c)
        )
        ax.add_collection3d(lc)

    ax.set_xlabel('X [AU]')
    ax.set_ylabel('Y [AU]')
    ax.set_zlabel('Z [AU]')

    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.get_cmap('magma'), 
        norm=plt.Normalize(vmin=bmin, vmax=bmax)
    )
    # fake up the array of the scalar mappable. Urgh...
    sm._A = []
    cb = plt.colorbar(sm)
    cb.set_label('B [nT]')

    ax.set_xlim3d(0, 1.2)
    ax.set_ylim3d(-0.6, 0.6)
    ax.set_zlim3d(-0.6, 0.6)

    top_view = (90.0,-90.0)
    side_view = (0.0,-90.0)
    front_view = (0.0,0.0)
    ax.view_init(*side_view)

    def orthogonal_proj(zfront, zback):
        a = (zfront+zback)/(zfront-zback)
        b = -2*(zfront*zback)/(zfront-zback)
        return np.array([[1,0,0,0],
                         [0,1,0,0],
                         [0,0,a,b],
                         [0,0,-0.0001,zback]])
    proj3d.persp_transformation = orthogonal_proj

    plt.tight_layout()
    plt.show()

demo_line()
