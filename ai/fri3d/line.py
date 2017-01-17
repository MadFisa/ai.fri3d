
import numpy as np
from astropy import constants as c
from astropy import units as u
from ai.shared import cs

def line(self, r=0.0, phi=0.0, s=np.linspace(0.0, 1.0, 50)):
    s = np.array(s, copy=False, ndmin=1)
    
    s_max = self._initial_axis_s(self.half_width)
    s[s < c.R_sun.value/s_max] = c.R_sun.value/s_max
    s[s > 1.0-c.R_sun.value/s_max] = 1.0-c.R_sun.value/s_max
    s = np.unique(s)

    # twist = self.twist*self._initial_axis_s(self.half_width)
    
    phi = np.ones(s.size)*phi

    # twist
    # phi += s*twist*np.pi*2.0*self.chirality
    phi += s*self.twist*np.pi*2.0*self.chirality
    # elongation
    z = s*self._initial_axis_s(self.half_width)

    # distance to axis from origin
    R = self._initial_axis_r(self._spline_initial_axis_s_phi(z))
    # cross-section radial size in the FR plane
    rx = R*self.poloidal_height/self.toroidal_height
    # cross-section radial size perp to FR plane
    ry = R*self.pancaking
    # coefficient of flux decay
    kappa = rx*ry
    # taper
    r *= rx
    # magnetic field
    # b = ghb(ghb0(self.flux, self.twist, rx, ry), self.twist, r*np.cos(phi), r*np.sin(phi))
    b = self._unit_b/kappa*np.exp(
        -((r/rx)**2)/2.0/self.sigma**2
    )
    x, y, z = cs.cyl2cart(r, phi, z)

    # rotation to x
    T = cs.mx_rot_y(np.pi/2.0)
    x, y, z = cs.mx_apply(T, x, y, z)
    
    # bend
    phi = self._spline_initial_axis_s_phi(x)
    r = self._initial_axis_r(phi)
    t = self._initial_axis_tan(phi)
    x = r*np.cos(phi)+np.sin(t-phi-np.pi/2.0)*y
    y = r*np.sin(phi)+np.cos(t-phi-np.pi/2.0)*y

    # pancake
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

    return (x, y, z, b)

# def ghb0(F, T0, Rx, Ry):
#     return F*T0**2*Rx/np.pi/Ry/np.log(np.abs(1.0+T0**2*Rx**2))

# def ghb(B0, T0, r):
#     return (B0/np.sqrt(1.0+T0**2*r**2))

# def ghb(B0, T0, x, y):
#     return (B0/np.sqrt(1.0+T0**2*(x**2+y**2)))

