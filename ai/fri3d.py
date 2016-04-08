
import numpy as np
from ai import cs

import scipy.special
import scipy.interpolate
import scipy.integrate
import scipy.optimize

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.path as mpath
import matplotlib.colors as colors
from matplotlib.colorbar import ColorbarBase

AU_KM = 1.496e8
RS_KM = 6.957e5
RS_AU = RS_KM/AU_KM

class FRi3D:
    
    def __init__(
            self,
            latitude = 0.0, 
            longitude = 0.0, 
            toroidal_height = 1.0, 
            poloidal_height = 0.1, 
            half_width = np.pi/6.0, 
            tilt = 0.0, 
            flattening = 0.5, 
            pancaking = np.pi/6.0, 
            skew = 0.0, 
            twist = 1.0, 
            flux = 1e15):
        self.latitude = latitude
        self.longitude = longitude
        self.toroidal_height = toroidal_height
        self.poloidal_height = poloidal_height
        self.half_width = half_width
        self.tilt = tilt
        self.flattening = flattening
        self.pancaking = pancaking
        self.skew = skew
        self.twist = twist
        self.flux = flux
        self._init_spline_initial_axis_s_phi()
    _sigma = 1.05

    @property
    def twist(self):
        return self._twist

    @twist.setter
    def twist(self, twist):
        if twist > 0.0:
            self._twist = twist

    @property
    def toroidal_height(self):
        return self._toroidal_height

    @toroidal_height.setter
    def toroidal_height(self, toroidal_height):
        self._toroidal_height = toroidal_height

    @property
    def poloidal_height(self):
        return self._poloidal_height

    @poloidal_height.setter
    def poloidal_height(self, poloidal_height):
        self._poloidal_height = poloidal_height

    @property
    def half_width(self):
        return self._half_width

    @half_width.setter
    def half_width(self, half_width):
        self._half_width = half_width
        self._coeff_angle = np.pi/2.0/self.half_width

    @property
    def coeff_angle(self):
        return self._coeff_angle

    @property
    def flattening(self):
        return self._flattening

    @flattening.setter
    def flattening(self, flattening):
        self._flattening = flattening

    @property
    def pancaking(self):
        return self._pancaking

    @pancaking.setter
    def pancaking(self, pancaking):
        self._pancaking = pancaking

    @property
    def skew(self):
        return self._skew

    @skew.setter
    def skew(self, skew):
        self._skew = skew

    @property
    def latitude(self):
        return self._latitude

    @latitude.setter
    def latitude(self, latitude):
        self._latitude = latitude

    @property
    def longitude(self):
        return self._longitude

    @longitude.setter
    def longitude(self, longitude):
        self._longitude = longitude

    @property
    def tilt(self):
        return self._tilt

    @tilt.setter
    def tilt(self, tilt):
        self._tilt = tilt    

    @property
    def flux(self):
        return self._flux

    @flux.setter
    def flux(self, flux):
        self._flux = flux
        self._unit_b = flux/(2.0*np.pi*self._sigma**2)

    def _init_spline_initial_axis_s_phi(self):
        phi = np.linspace(-self.half_width, self.half_width, 100)
        s = np.array([self._initial_axis_s(p) for p in phi])
        self._spline_initial_axis_s_phi = scipy.interpolate.interp1d(
            s, phi, kind='cubic',
            bounds_error=False,
            fill_value=(-self.half_width, self.half_width)
        )

    def _initial_axis_r(self, phi):
        return np.nan_to_num(
            self.toroidal_height*
            np.cos(self.coeff_angle*phi)**self.flattening
        )

    def _initial_axis_dr(self, phi):
        return (
            -self.coeff_angle*self.toroidal_height*self.flattening*
            np.cos(self.coeff_angle*phi)**(self.flattening-1.0)*
            np.sin(self.coeff_angle*phi)
        )

    def _initial_axis_l(self, phi, r0, phi0):
        return (
            (self._initial_axis_r(phi)*np.cos(phi)-r0*np.cos(phi0))**2+
            (self._initial_axis_r(phi)*np.sin(phi)-r0*np.sin(phi0))**2
        )

    def _initial_axis_min_l_phi(self, r0, phi0):
        res = scipy.optimize.minimize_scalar(
            lambda phi: self._initial_axis_l(phi, r0, phi0),
            bounds=[-self.half_width, self.half_width],
            method='Brent'
        )
        return res.x

    def _initial_axis_tan(self, phi):
        return np.arctan(
            -self.coeff_angle*self.flattening*np.tan(self.coeff_angle*phi)
        )

    def _initial_axis_ds(self, phi):
        return np.sqrt(
            self._initial_axis_r(phi)**2+
            self._initial_axis_dr(phi)**2
        )

    def _initial_axis_s(self, phi):
        s = scipy.integrate.quad(self._initial_axis_ds, -self.half_width, phi)
        return s[0]

    def shell(self, 
            s=np.linspace(0.0, 1.0, 50), 
            phi=np.linspace(0.0, np.pi*2.0, 24)):
        s = np.array(s, copy=False, ndmin=1)
        phi = np.array(phi, copy=False, ndmin=1)

        s_max = self._initial_axis_s(self.half_width)
        s[s < RS_AU/s_max] = RS_AU/s_max
        s[s > 1.0-RS_AU/s_max] = 1.0-RS_AU/s_max
        s = np.unique(s)

        s = np.transpose(np.tile(s, (phi.size, 1)))
        phi = np.tile(phi, (s.shape[0], 1))

        r = np.ones(s.shape)
        z = s*self._initial_axis_s(self.half_width)
        
        r = (
            r*self._initial_axis_r(self._spline_initial_axis_s_phi(z))*
            self.poloidal_height/self.toroidal_height
        )
        x_, y_, z_ = cs.cyl2cart(r, phi, z)

        T = cs.mx_rot_y(-np.pi/2.0)
        x = T[0,0]*x_+T[0,1]*y_+T[0,2]*z_
        y = T[1,0]*x_+T[1,1]*y_+T[1,2]*z_
        z = T[2,0]*x_+T[2,1]*y_+T[2,2]*z_

        # bending
        phi = self._spline_initial_axis_s_phi(x)
        r = self._initial_axis_r(phi)
        t = self._initial_axis_tan(phi)
        x_ = r*np.cos(phi)+np.sin(t-phi-np.pi/2.0)*y
        y_ = r*np.sin(phi)+np.cos(t-phi-np.pi/2.0)*y
        z_ = z

        # pancaking
        r, theta, phi = cs.cart2sp(x_, y_, z_)
        theta = (
            theta/np.arctan2(self.poloidal_height, self.toroidal_height)*
            self.pancaking
        )
        x_, y_, z_ = cs.sp2cart(r, theta, phi)

        # orientation
        T = cs.mx_rot(self.latitude, -self.longitude, -self.tilt)
        x = T[0,0]*x_+T[0,1]*y_+T[0,2]*z_
        y = T[1,0]*x_+T[1,1]*y_+T[1,2]*z_
        z = T[2,0]*x_+T[2,1]*y_+T[2,2]*z_

        # skew
        r, phi, z = cs.cart2cyl(x, y, z)
        phi += self.skew*(1.0-r/self.toroidal_height)
        x, y, z = cs.cyl2cart(r, phi, z)
        
        return (x, y, z)

    def line(self, r=0.0, phi=0.0, s=np.linspace(0.0, 1.0, 50)):
        s = np.array(s, copy=False, ndmin=1)
        
        s_max = self._initial_axis_s(self.half_width)
        s[s < RS_AU/s_max] = RS_AU/s_max
        s[s > 1.0-RS_AU/s_max] = 1.0-RS_AU/s_max
        s = np.unique(s)
        
        r = np.ones(s.size)*r
        phi = np.ones(s.size)*phi

        # twist
        phi += s*self.twist*np.pi*2.0
        # elongation
        z = s*self._initial_axis_s(self.half_width)
        # magnetic field
        R = self._initial_axis_r(self._spline_initial_axis_s_phi(z))
        rx = R*self.poloidal_height/self.toroidal_height
        ry = R*self.pancaking
        kappa = rx*ry*(AU_KM*1e3)**2
        # taper
        r *= rx
        # magnetic field
        x = r*np.cos(phi)
        y = r*np.sin(phi)
        b = self._unit_b/kappa*np.exp(
            -((r/rx)**2)/2.0/self._sigma**2
        )
        # r = (
        #     r*self._initial_axis_r(self._spline_initial_axis_s_phi(z))*
        #     self.poloidal_height/self.toroidal_height
        # )
        x_, y_, z_ = cs.cyl2cart(r, phi, z)

        T = cs.mx_rot_y(-np.pi/2.0)
        x = T[0,0]*x_+T[0,1]*y_+T[0,2]*z_
        y = T[1,0]*x_+T[1,1]*y_+T[1,2]*z_
        z = T[2,0]*x_+T[2,1]*y_+T[2,2]*z_

        # bend
        phi = self._spline_initial_axis_s_phi(x)
        r = self._initial_axis_r(phi)
        t = self._initial_axis_tan(phi)
        x_ = r*np.cos(phi)+np.sin(t-phi-np.pi/2.0)*y
        y_ = r*np.sin(phi)+np.cos(t-phi-np.pi/2.0)*y
        z_ = z

        # pancake
        r, theta, phi = cs.cart2sp(x_, y_, z_)
        theta = (
            theta/np.arctan2(self.poloidal_height, self.toroidal_height)*
            self.pancaking
        )
        x_, y_, z_ = cs.sp2cart(r, theta, phi)

        # orientation
        T = cs.mx_rot(self.latitude, -self.longitude, -self.tilt)
        x = T[0,0]*x_+T[0,1]*y_+T[0,2]*z_
        y = T[1,0]*x_+T[1,1]*y_+T[1,2]*z_
        z = T[2,0]*x_+T[2,1]*y_+T[2,2]*z_

        # skew
        r, phi, z = cs.cart2cyl(x, y, z)
        phi += self.skew*(1.0-r/self.toroidal_height)
        x, y, z = cs.cyl2cart(r, phi, z)

        b *= 1e9
        
        return (x, y, z, b)

    def cut(self, x, y, z):
        x = np.array(x, copy=False, ndmin=1)
        y = np.array(y, copy=False, ndmin=1)
        z = np.array(z, copy=False, ndmin=1)

        r, theta, phi = cs.cart2sp(x, y, z)
        phi -= self.skew*(1.0-r/self.toroidal_height)
        x, y, z = cs.sp2cart(r, theta, phi)

        # reverse orientation
        T = cs.mx_rot(-self.latitude, self.longitude, self.tilt)
        x_ = T[0,0]*x+T[0,1]*y+T[0,2]*z
        y_ = T[1,0]*x+T[1,1]*y+T[1,2]*z
        z_ = T[2,0]*x+T[2,1]*y+T[2,2]*z

        # reverse pancake
        r, theta, phi = cs.cart2sp(x_, y_, z_)
        theta = (
            theta/self.pancaking*
            np.arctan2(self.poloidal_height, self.toroidal_height)
        )
        x, y, z = cs.sp2cart(r, theta, phi)

        p_in = self._initial_axis_r(phi) >= r*np.cos(theta)
        p_out = np.logical_not(p_in)
        # get r_ax and phi_ax of the closest point on axis
        v_initial_axis_min_l_phi = np.vectorize(self._initial_axis_min_l_phi)
        phi_ax = v_initial_axis_min_l_phi(r*np.cos(theta), phi)
        r_ax = self._initial_axis_r(phi_ax)
        # get s
        v_initial_axis_s = np.vectorize(self._initial_axis_s)
        s = v_initial_axis_s(phi_ax)/self._initial_axis_s(self.half_width)
        # get r and phi params
        x_ax, y_ax, z_ax = cs.sp2cart(r_ax, np.zeros(r_ax.size), phi_ax)
        dx = x-x_ax
        dy = y-y_ax
        dz = z-z_ax
        r_abs = np.sqrt(dx**2+dy**2+dz**2)
        r = r_abs/(r_ax*self.poloidal_height/self.toroidal_height)

        phi = np.piecewise(dz, [dz < 0, dz >= 0], [-1, 1])*np.arccos(np.sqrt(dx**2+dy**2)/r_abs)
        phi[p_in] = np.pi-phi[p_in]
        # reverse twist
        phi -= s*self.twist*np.pi*2.0
        phi -= np.pi/2.0
        
        mask = r <= 1.0
        r = r[mask]
        phi = phi[mask]
        s = s[mask]
        # print(r,phi*180.0/np.pi,s)

        b = []

        for i in range(r.size):
            x_, y_, z_, b_ = self.line(
                r[i],
                phi[i],
                [s[i]-0.0001, s[i]+0.0001]
            )
            dr = np.array([
                x_[1]-x_[0],
                y_[1]-y_[0],
                z_[1]-z_[0]
            ])
            dr /= np.linalg.norm(dr)
            b.append(np.insert(dr*np.mean(b_), 0, np.mean(b_)))

        return np.array(b)

    def evocut(self, x, y, z, 
            toroidal_height=np.linspace(0.8, 1.5, 100)):
        b = []
        for i in range(toroidal_height.size):
            self.toroidal_height = toroidal_height[i]
            self._init_spline_initial_axis_s_phi()
            b_ = self.cut(x, y, z)
            print(b_)
            if b_.size > 0:
                b.append(b_.ravel())
        return np.array(b)

def test():
    fr = FRi3D(
        twist=2.0,
        half_width=np.pi/4.0, 
        pancaking=np.pi/6.0, 
        poloidal_height=0.2,
        flattening=0.6,
        tilt=np.pi/180.0*0.0,
        skew=np.pi/180.0*10.0,
        longitude=-np.pi/180.0*0.0
    )

    b = fr.evocut(1.0, 0.0, 0.0)
    print(b)
    fig = plt.figure()
    plt.plot(b[:,0], 'k')
    plt.plot(b[:,1], 'r')
    plt.plot(b[:,2], 'g')
    plt.plot(b[:,3], 'b')

    fr.toroidal_height = 1.0

    n = 103

    r = np.linspace(1.5, 0.4, n)
    theta = np.ones(n)*np.pi/180.0*0.0
    phi = np.ones(n)*np.pi/180.0*0.0
    x, y, z = cs.sp2cart(r, theta, phi)

    b = fr.cut(x, y, z)
    fig = plt.figure()
    plt.plot(b[:,0], 'k')
    plt.plot(b[:,1], 'r')
    plt.plot(b[:,2], 'g')
    plt.plot(b[:,3], 'b')
    # plt.show()
    # return

    # fr.field()

    fig = plt.figure(figsize=(8, 8), dpi=72)
    ax = fig.add_subplot(111, projection='3d', adjustable='box', aspect=1.0)
    x, y, z = fr.shell()
    ax.plot_wireframe(x, y, z, alpha=0.1)
    
    _, _, _, b = fr.line(0.0, 0.0, s=0.5)
    print(b)
    _, _, _, b = fr.line(1.0, 0.0, s=0.5)
    print(b)
    bmin = b
    _, _, _, b = fr.line(0.0, 0.0, s=0.9)
    print(b)
    bmax = b
    _, _, _, b = fr.line(1.0, 0.0, s=0.9)
    print(b)

    for i in range(50):
        r = np.random.uniform(0.0, 1.0)
        phi = np.random.uniform(0.0, np.pi*2.0)
        x, y, z, b = fr.line(r, phi, s=np.linspace(0.1, 0.9, 200))
        points = np.array([x, y, z]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        # print((b-bmin)/(bmax-bmin))
        c = (b-bmin)/(bmax-bmin)
        # print(b, c)
        lc = Line3DCollection(
            segments, 
            colors=plt.cm.magma(c)
        )
        ax.add_collection3d(lc)

    ax.set_xlim(0.0, 1.2)
    ax.set_ylim(-0.6, 0.6)
    ax.set_zlim(-0.6, 0.6)

    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.get_cmap('magma'), 
        norm=plt.Normalize(vmin=bmin, vmax=bmax)
    )
    # fake up the array of the scalar mappable. Urgh...
    sm._A = []
    plt.colorbar(sm)
    
    plt.show()

def orthogonal_proj(zfront, zback):
    a = (zfront+zback)/(zfront-zback)
    b = -2*(zfront*zback)/(zfront-zback)
    return np.array([[1,0,0,0],
                     [0,1,0,0],
                     [0,0,a,b],
                     [0,0,-0.0001,zback]])
proj3d.persp_transformation = orthogonal_proj