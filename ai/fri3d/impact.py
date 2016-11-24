
import numpy as np
from ai.shared import cs

def impact(self, x, y, z):
    x0 = x
    y0 = y
    z0 = z

    # reverse skew
    r, theta, phi = cs.cart2sp(x, y, z)
    phi -= self.skew*(1.0-r/self.toroidal_height)
    x, y, z = cs.sp2cart(r, theta, phi)

    # reverse orientation
    T = cs.mx_rot_reverse(self.latitude, -self.longitude, -self.tilt)
    x, y, z = cs.mx_apply(T, x, y, z)
    
    # reverse pancaking
    r, theta, phi = cs.cart2sp(x, y, z)
    theta = (
        theta/self.pancaking*
        np.arctan2(self.poloidal_height, self.toroidal_height)
    )
    # x, y, z = cs.sp2cart(r, theta, phi)

    # inside axis loop mask
    p_in = self._initial_axis_r(phi) >= r*np.cos(theta)
    # outside axis loop mask
    p_out = np.logical_not(p_in)
    # get r_ax and phi_ax of the closest point on axis
    v_initial_axis_min_l_phi = np.vectorize(
        self._initial_axis_min_l_phi, 
        otypes=[np.float64]
    )
    phi_ax = v_initial_axis_min_l_phi(r*np.cos(theta), phi)
    r_ax = self._initial_axis_r(phi_ax)
    v_initial_axis_s = np.vectorize(self._initial_axis_s, otypes=[np.float64])
    s = v_initial_axis_s(phi_ax)/self._initial_axis_s(self.half_width)
    x_ax, y_ax, z_ax = cs.sp2cart(r_ax, np.zeros(r_ax.size), phi_ax)

    # pancaking
    r, theta, phi = cs.cart2sp(x_ax, y_ax, z_ax)
    theta = (
        theta/np.arctan2(self.poloidal_height, self.toroidal_height)*
        self.pancaking
    )
    x, y, z = cs.sp2cart(r, theta, phi)

    # orientation
    T = cs.mx_rot(-self.latitude, self.longitude, self.tilt)
    x, y, z = cs.mx_apply(T, x, y, z)

    # skew
    r, theta, phi = cs.cart2sp(x, y, z)
    phi += self.skew*(1.0-r/self.toroidal_height)
    x, y, z = cs.sp2cart(r, theta, phi)

    # get r_ax and phi_ax of the closest delta points on axis
    dphi = 1e-5
    phi_ax1 = phi_ax-dphi
    phi_ax2 = phi_ax+dphi
    r_ax1 = self._initial_axis_r(phi_ax1)
    r_ax2 = self._initial_axis_r(phi_ax2)
    s1 = v_initial_axis_s(phi_ax1)/self._initial_axis_s(self.half_width)
    s2 = v_initial_axis_s(phi_ax2)/self._initial_axis_s(self.half_width)
    x_ax1, y_ax1, z_ax1 = cs.sp2cart(r_ax1, np.zeros(r_ax1.size), phi_ax1)
    x_ax2, y_ax2, z_ax2 = cs.sp2cart(r_ax2, np.zeros(r_ax2.size), phi_ax2)

    # pancaking
    r, theta, phi = cs.cart2sp(x_ax1, y_ax1, z_ax1)
    theta = (
        theta/np.arctan2(self.poloidal_height, self.toroidal_height)*
        self.pancaking
    )
    x1, y1, z1 = cs.sp2cart(r, theta, phi)

    r, theta, phi = cs.cart2sp(x_ax2, y_ax2, z_ax2)
    theta = (
        theta/np.arctan2(self.poloidal_height, self.toroidal_height)*
        self.pancaking
    )
    x2, y2, z2 = cs.sp2cart(r, theta, phi)

    # orientation
    T = cs.mx_rot(-self.latitude, self.longitude, self.tilt)
    x1, y1, z1 = cs.mx_apply(T, x1, y1, z1)
    x2, y2, z2 = cs.mx_apply(T, x2, y2, z2)

    # skew
    r, theta, phi = cs.cart2sp(x1, y1, z1)
    phi += self.skew*(1.0-r/self.toroidal_height)
    x1, y1, z1 = cs.sp2cart(r, theta, phi)

    r, theta, phi = cs.cart2sp(x2, y2, z2)
    phi += self.skew*(1.0-r/self.toroidal_height)
    x2, y2, z2 = cs.sp2cart(r, theta, phi)

    d = np.array([x2, y2, z1])-np.array([x1, y1, z1])
    d /= np.linalg.norm(d)

    return (np.linalg.norm(np.array([x-x0, y-y0, z-z0])), x, y, z, d[0], d[1], d[2])
