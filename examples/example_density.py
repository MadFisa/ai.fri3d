"""Sample density assignment using FRi3D geometry.

This follows the same coordinate-mapping steps as StaticFRi3D.data(),
but returns a scalar plasma density instead of a magnetic field vector.
"""

import numpy as np
from astropy import units as u

from ai import cs
from ai.fri3d.model import StaticFRi3D


def density_at_points(sfr, x, y, z, n_axis=30.0, n_background=5.0, sigma=0.35):
    """Estimate density [cm^-3] at given point(s) in space.

    Args:
        sfr (StaticFRi3D): FRi3D snapshot.
        x, y, z (scalar or array_like): Cartesian coordinates [m].
        n_axis (float): Density near the magnetic axis [cm^-3].
        n_background (float): Background density outside rope [cm^-3].
        sigma (float): Width of radial Gaussian profile in normalized radius.

    Returns:
        scalar or ndarray: Density [cm^-3].
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    scalar_input = False
    if x.ndim == 0 and y.ndim == 0 and z.ndim == 0:
        x = x[None]
        y = y[None]
        z = z[None]
        scalar_input = True

    # 1) Reverse skew
    r, theta, phi = cs.cart2sp(x, y, z)
    phi -= sfr.skew * (1 - r / sfr.toroidal_height)
    x, y, z = cs.sp2cart(r, theta, phi)

    # 2) Reverse global orientation
    T = cs.mx_rot_reverse(sfr.latitude, -sfr.longitude, -sfr.tilt)
    x, y, z = cs.mx_apply(T, x, y, z)

    # 3) Same radial-expansion correction as in StaticFRi3D.data()
    r, theta, phi = cs.cart2sp(x, y, z)
    x, y, z = cs.cyl2cart(r, phi, z)

    with np.errstate(invalid="ignore"):
        mask_inside_axis_loop = sfr.vanilla_axis_height(phi) >= r

    # 4) Closest point on axis
    v_min = np.vectorize(sfr.vanilla_axis_min_distance, otypes=[np.float64, np.float64])
    _, phi_ax = v_min(r, phi)
    r_ax = sfr.vanilla_axis_height(phi_ax)

    # 5) Local cross-section coordinates
    x_ax, y_ax, z_ax = cs.cyl2cart(r_ax, phi_ax, np.zeros(r_ax.size))
    dx = x - x_ax
    dy = y - y_ax
    dz = z - z_ax
    r_abs = np.sqrt(dx**2 + dy**2 + dz**2)
    theta_cs = np.arctan2(dz, np.sqrt(dx**2 + dy**2))
    theta_cs[mask_inside_axis_loop] = np.pi - theta_cs[mask_inside_axis_loop]

    rx = r_ax * sfr._poloidal_height / sfr.toroidal_height
    ry = r_ax * sfr._poloidal_height / sfr.toroidal_height
    pancaking = 1 - (1 - sfr.pancaking) / np.sqrt(
        1 + (sfr.flattening * sfr._coeff_angle * np.tan(sfr._coeff_angle * phi_ax)) ** 2
    )
    rx *= pancaking
    r_tot = rx * ry / np.sqrt((ry * np.cos(theta_cs)) ** 2 + (rx * np.sin(theta_cs)) ** 2)
    r_rel = r_abs / r_tot

    # 6) Density model in normalized cross-section radius
    density = np.full(r_rel.shape, n_background, dtype=float)
    inside = r_rel <= 1
    density[inside] = n_background + (n_axis - n_background) * np.exp(-0.5 * (r_rel[inside] / sigma) ** 2)

    if scalar_input:
        return density.squeeze()
    return density


if __name__ == "__main__":
    sfr = StaticFRi3D(
        toroidal_height=u.au.to(u.m, 1.0),
        half_width=np.deg2rad(40),
        half_height=np.deg2rad(25),
        flattening=0.5,
        pancaking=0.6,
        skew=np.deg2rad(20),
        latitude=np.deg2rad(5),
        longitude=np.deg2rad(10),
        tilt=np.deg2rad(-15),
    )

    # Synthetic trajectory through near-Earth space
    n = 100
    x = u.au.to(u.m, np.linspace(0.8, 1.2, n))
    y = np.zeros(n)
    z = np.zeros(n)

    density = density_at_points(sfr, x, y, z)
    print("Density shape:", density.shape)
    print("Density min/max [cm^-3]:", float(np.nanmin(density)), float(np.nanmax(density)))
