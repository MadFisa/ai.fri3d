
import numpy as np
from ai.shared import cs

def data(self, x, y, z, ds=1e-5):
    x = np.array(x, copy=False, ndmin=1)
    y = np.array(y, copy=False, ndmin=1)
    z = np.array(z, copy=False, ndmin=1)



    # twist = self.twist*self._initial_axis_s(self.half_width)

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
    x, y, z = cs.sp2cart(r, theta, phi)

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
    # get s
    v_initial_axis_s = np.vectorize(self._initial_axis_s, otypes=[np.float64])
    s = v_initial_axis_s(phi_ax)/self._initial_axis_s(self.half_width)
    # print('finished quad')
    # return(
    #     np.array([np.nan, np.nan, np.nan]),
    #     np.array([np.nan, np.nan])
    # )
    # get r[0,1] and phi[0,2pi] params
    x_ax, y_ax, z_ax = cs.sp2cart(r_ax, np.zeros(r_ax.size), phi_ax)
    dx = x-x_ax
    dy = y-y_ax
    dz = z-z_ax
    r_abs = np.sqrt(dx**2+dy**2+dz**2)
    r = r_abs/(r_ax*self.poloidal_height/self.toroidal_height)
    
    def div0(a, b):
        with np.errstate(divide='ignore', invalid='ignore'):
            c = np.true_divide(a, b)
            c[~np.isfinite(c)] = 0  # -inf inf NaN
        return c

    phi = (
        np.piecewise(dz, [dz < 0, dz >= 0], [-1, 1])*
        np.arccos(div0(np.sqrt(dx**2+dy**2), r_abs))
    )
    phi[p_in] = np.pi-phi[p_in]
    # reverse twist
    phi -= s*self.twist*np.pi*2.0*self.chirality
    # phi -= s*twist*np.pi*2.0*self.chirality
    # reverse rotation to x
    phi -= np.pi/2.0
    # only inside FR
    mask = r <= 1.0
    # r = r[mask]
    # phi = phi[mask]
    # s = s[mask]
    
    # get magnetic field along sc trajectory
    b = []
    c = []
    for i in range(r.size):
        if r[i] <= 1.0:
            x_, y_, z_, b_ = self.line(
                r[i],
                phi[i],
                [s[i]-ds, s[i]+ds]
            )
            if x_.size < 2 or y_.size < 2 or z_.size < 2:
                b.append([np.nan, np.nan, np.nan])
                c.append([np.nan, np.nan])
            else:
                vtc = r_ax[i]/self.toroidal_height
                vpc = (
                    r_ax[i]/self.toroidal_height*
                    (np.sqrt(np.mean(x_)**2+np.mean(y_)**2+np.mean(z_)**2)-r_ax[i])/
                    self.poloidal_height*np.cos(self._initial_axis_tan(phi_ax[i]))
                )
                # print(
                #     r_ax[i]/
                #     self.toroidal_height*
                #     (np.sqrt(np.mean(x_)**2+np.mean(y_)**2+np.mean(z_)**2)-r_ax[i])/
                #     self.poloidal_height*
                #     np.cos(self._initial_axis_tan(phi_ax[i]))*
                #     50+
                #     r_ax[i]/self.toroidal_height*
                #     400
                # )
                dr = np.array([
                    x_[1]-x_[0],
                    y_[1]-y_[0],
                    z_[1]-z_[0]
                ])
                dr /= np.linalg.norm(dr)
                b.append(dr*np.mean(b_)*self.polarity)
                c.append(np.array([vtc, vpc]))
        else:
            b.append([np.nan, np.nan, np.nan])
            c.append([np.nan, np.nan])
    b = np.array(b)
    if b.shape[0] == 1:
        b = b[0,:]
    c = np.array(c)
    if c.shape[0] == 1:
        c = c[0,:]

    return (b, c)
