
import numpy as np
from ai.shared import cs
from astropy import constants as c

def shell(
        self,
        s=np.linspace(0.0, 1.0, 50, endpoint=True),
        phi=np.linspace(0.0, np.pi*2.0, 24)):
    s = np.array(s, copy=False, ndmin=1)
    phi = np.array(phi, copy=False, ndmin=1)

    # start the FR from the solar surface
    # s_max = self._vanilla_axis_length(self.half_width)
    # s[s < c.R_sun.value/s_max] = c.R_sun.value/s_max
    # s[s > 1.0-c.R_sun.value/s_max] = 1.0-c.R_sun.value/s_max
    # s = np.unique(s)

    s = np.transpose(np.tile(s, (phi.size, 1)))
    phi = np.tile(phi, (s.shape[0], 1))

    # extension to full axis length
    r = np.ones(s.shape)
    z = s*self._vanilla_axis_length(self.half_width)

    # tapering
    r = (
        r*self.poloidal_height*
        (
            self._vanilla_axis_height(self._vanilla_axis_phi(z))/
            self.toroidal_height
        )
    )
    x, y, z = cs.cyl2cart(r, phi, z)

    # rotation to x axis
    T = cs.mx_rot_y(np.pi/2.0)
    x, y, z = cs.mx_apply(T, x, y, z)

    # bending
    phi = self._vanilla_axis_phi(x)
    r = self._vanilla_axis_height(phi)
    t = self._vanilla_axis_tan(phi)
    x = r*np.cos(phi)+np.sin(t-phi-np.pi/2.0)*y
    y = r*np.sin(phi)+np.cos(t-phi-np.pi/2.0)*y

    # pancaking
    r, theta, phi = cs.cart2sp(x, y, z)
    theta = (
        theta/np.arctan2(self.poloidal_height, self.toroidal_height)*
        self.pancaking
    )
    x, y, z = cs.sp2cart(r, theta, phi)

    # orientation
    T = cs.mx_rot(-self.latitude, self.longitude, self.tilt)
    x, y, z = cs.mx_apply(T, x, y, z)

    # skew
    r, phi, z = cs.cart2cyl(x, y, z)
    phi += self.skew*(1.0-r/self.toroidal_height)
    x, y, z = cs.cyl2cart(r, phi, z)

    return(x, y, z)
