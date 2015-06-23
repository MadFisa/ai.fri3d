
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import scipy.special

class AIFR:
    _radius_tor = 1.0
    _radius_pol = 0.1
    _half_width = numpy.pi/6.0
    _coeff_angle = 3.0
    _coeff_flat = 0.5
    _lat = 0.0
    _lon = 0.0
    _tilt = 0.0
    _coeff_twist = 0.0
    _radius_corot = 0.02
    _coeff_panc = 0.1
    # _grid_tor = None
    # _grid_pol = None
    # _helicity = None
    # _polarity = None
    # _core_field = None

    def set_radius_tor(self, new_radius_tor):
        self._radius_tor = new_radius_tor

    def set_radius_pol(self, new_radius_pol):
        self._radius_pol = new_radius_pol

    def set_half_width(self, new_half_width):
        self._half_width = new_half_width
        self._coeff_angle = numpy.pi/2.0/self._half_width

    def set_coeff_flat(self, new_coeff_flat):
        self._coeff_flat = new_coeff_flat

    def set_lat(self, new_lat):
        self._lat = new_lat

    def set_lon(self, new_lon):
        self._lon = new_lon

    def set_tilt(self, new_tilt):
        self._tilt = new_tilt

    def set_coeff_twist(self, new_coeff_twist):
        self._coeff_twist = new_coeff_twist

    def set_radius_corot(self, new_radius_corot):
        self._radius_corot = new_radius_corot

    def set_coeff_panc(self, new_coeff_panc):
        self._coeff_panc = new_coeff_panc

    def _axis_r(self, phi):
        return self._radius_tor* \
               numpy.cos(self._coeff_angle*phi)**self._coeff_flat

    def _axis_dr(self, phi):
        return self._axis_r(phi)*(-self._coeff_angle*
                                   self._coeff_flat*
                                   numpy.tan(self._coeff_angle*phi))

    def _axis_d2r(self, phi):
        return self._axis_r(phi)*self._coeff_angle**2* \
               (self._coeff_flat*(self._coeff_flat-1)*
                numpy.tan(self._coeff_angle*phi)**2-self._coeff_flat)

    def axis_curv(self, phi):
        return (self._axis_r(phi)**2+self._axis_dr(phi)**2)**(1.5)/ \
               numpy.abs(self._axis_r(phi)**2+
                         2.0*self._axis_dr(phi)**2-
                         self._axis_r(phi)*self._axis_d2r(phi))

    def _axis_tangent(self, phi):
        return numpy.arctan2(1.0, self._coeff_angle*self._coeff_flat*
                                  numpy.tan(self._coeff_angle*phi))

    def _shell_x(self, theta, phi):
        return self._axis_r(phi)* \
               (self._radius_pol/self._radius_tor*numpy.cos(theta)* \
                numpy.sin(self._axis_tangent(phi)-phi)+numpy.cos(phi))

    def _shell_y(self, theta, phi):
        return self._axis_r(phi)* \
               (self._radius_pol/self._radius_tor*numpy.cos(theta)* \
                numpy.cos(self._axis_tangent(phi)-phi)+numpy.sin(phi))

    def _shell_z(self, theta, phi):
        return self._axis_r(phi)*self._radius_pol/ \
               self._radius_tor*numpy.sin(theta)

    @staticmethod
    def mx_rot_x(gamma):
        return numpy.matrix([[1.0, 0.0, 0.0], 
                             [0.0, numpy.cos(gamma), numpy.sin(gamma)], 
                             [0.0, -numpy.sin(gamma), numpy.cos(gamma)]])

    @staticmethod
    def mx_rot_y(theta):
        return numpy.matrix([[numpy.cos(theta), 0.0, -numpy.sin(theta)],
                             [0.0, 1.0, 0.0],
                             [numpy.sin(theta), 0.0, numpy.cos(theta)]])

    @staticmethod
    def mx_rot_z(phi):
        return numpy.matrix([[numpy.cos(phi), numpy.sin(phi), 0.0],
                             [-numpy.sin(phi), numpy.cos(phi), 0.0],
                             [0.0, 0.0, 1.0]])

    @staticmethod
    def mx_rot(theta, phi, gamma):
        return numpy.dot(AIFR.mx_rot_z(phi), 
                         numpy.dot(AIFR.mx_rot_y(theta), 
                                   AIFR.mx_rot_x(gamma)))

    @staticmethod
    def cart2sp(x, y, z):
        r = numpy.sqrt(x**2+y**2+z**2)
        theta = numpy.arcsin(z/r)
        phi = numpy.arctan2(y, x)
        return (r, theta, phi)

    @staticmethod
    def sp2cart(r, theta, phi):
        x = r*numpy.cos(theta)*numpy.cos(phi)
        y = r*numpy.cos(theta)*numpy.sin(phi)
        z = r*numpy.sin(theta)
        return (x, y, z)

    @staticmethod
    def cyl2cart(r, phi, z):
        x = r*numpy.cos(phi)
        y = r*numpy.sin(phi)
        return (x, y, z)

    @staticmethod
    def cart2cyl(x, y, z):
        r = numpy.sqrt(x**2+y**2)
        phi = numpy.arctan2(y, x)
        return (r, phi, z)

    def axis(self, phi):
        x0, y0, z0 = self.sp2cart(self._axis_r(phi), 0.0, phi)
        T = self.mx_rot(self._lat, -self._lon, -self._tilt)
        x = T[0,0]*x0+T[0,1]*y0+T[0,2]*z0
        y = T[1,0]*x0+T[1,1]*y0+T[1,2]*z0
        z = T[2,0]*x0+T[2,1]*y0+T[2,2]*z0
        r, theta, phi = self.cart2sp(x, y, z)
        # phi = phi+self._coeff_twist*(r-self._radius_corot).clip(min=0)
        phi = phi+self._coeff_twist*r/r.max()
        x, y, z = self.sp2cart(r, theta, phi)
        return (x, y, z)

    def shell(self, theta, phi):
        theta, phi = numpy.meshgrid(theta, phi)
        x0 = self._shell_x(theta, phi)
        y0 = self._shell_y(theta, phi)
        z0 = self._shell_z(theta, phi)
        
        # pancaking
        r, theta, phi = self.cart2sp(x0, y0, z0)
        # r_max = r.max()
        # r = r+r_max*self._coeff_panc
        # r = r/r.max()*r_max
        theta_max = numpy.arctan2(self._radius_pol, self._radius_tor)
        theta = theta/theta_max*self._coeff_panc#*r/self._radius_tor
        x0, y0, z0 = self.sp2cart(r, theta, phi)

        T = self.mx_rot(self._lat, -self._lon, -self._tilt)
        x = T[0,0]*x0+T[0,1]*y0+T[0,2]*z0
        y = T[1,0]*x0+T[1,1]*y0+T[1,2]*z0
        z = T[2,0]*x0+T[2,1]*y0+T[2,2]*z0
        r, theta, phi = self.cart2sp(x, y, z)
        # phi = phi+self._coeff_twist*(r-self._radius_corot).clip(min=0)
        phi = phi+self._coeff_twist*r/r.max()
        x, y, z = self.sp2cart(r, theta, phi)
        return (x, y, z)

def demo():

    def orthogonal_proj(zfront, zback):
        a = (zfront+zback)/(zfront-zback)
        b = -2*(zfront*zback)/(zfront-zback)
        return numpy.array([[1,0,0,0],
                            [0,1,0,0],
                            [0,0,a,b],
                            [0,0,-0.0001,zback]])
    proj3d.persp_transformation = orthogonal_proj

    fr = AIFR()
    han = numpy.pi/6.0
    fr.set_lat(numpy.pi/180.0*0.0)
    fr.set_lon(numpy.pi/180.0*50.0)
    fr.set_tilt(numpy.pi/180.0*0.0)
    fr.set_radius_tor(1.0)
    fr.set_radius_pol(0.1)
    fr.set_coeff_flat(0.4)
    fr.set_half_width(han)
    fr.set_coeff_twist(-numpy.pi/180.0*50.0)
    fr.set_radius_corot(0.02)
    fr.set_coeff_panc(numpy.pi/180.0*30.0)

    theta = numpy.linspace(0.0, 2.0*numpy.pi, 30)
    phi = numpy.linspace(-han, han, 1000)
    n = 20
    ds = numpy.sum(fr._axis_r(phi[1:])*(phi[1]-phi[0]))/2.0/n
    phi = [0.0]
    for i in range(1, n+1):
        phi = numpy.append(phi, phi[i-1]+ds/fr._axis_r(phi[i-1]))
    phi = numpy.append(phi, han)
    # phi[-1] = han
    phi = numpy.append(-phi[::-1], phi[1:])
    print phi*180.0/numpy.pi
    # phi = numpy.linspace(-han, han, 20)
    phir = numpy.linspace(-han*0.9, han*0.9, 100)
    
    fig = plt.figure()

    x, y, z = fr.axis(phi)
    ax = fig.add_subplot(221, projection='3d')
    ax.plot(x, y, z)

    ax = fig.add_subplot(222)
    ax.plot(phir*180.0/numpy.pi, fr.axis_curv(phir))
    ax.plot(phir*180.0/numpy.pi, fr._axis_r(phir)*fr._radius_pol/fr._radius_tor)

    ax = fig.add_subplot(212, projection='3d')
    x, y, z = fr.shell(theta, phi)
    print x.shape
    print y.shape
    print z.shape
    ax.plot_wireframe(x, y, z)
    max_range = numpy.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max()/2.0
    mean_x = x.mean()
    mean_y = y.mean()
    mean_z = z.mean()
    ax.set_xlim(0.0, mean_x+max_range)
    ax.set_ylim(mean_y-max_range, mean_y+max_range)
    ax.set_zlim(mean_z-max_range, mean_z+max_range)
    x, y, z = fr.axis(phi)
    ax.plot(x, y, z, color='red', linewidth=2.0)

    fig.show()

def b_cyl(r, phi, z):
    a = 2.4048
    return (0.0, scipy.special.j1(a*r), scipy.special.j0(a*r))

def b_cart(x, y, z):
    r, phi, z = AIFR.cart2cyl(x, y, z)
    br, bp, bz = b_cyl(r, phi, z)
    return (numpy.cos(phi)*br-numpy.sin(phi)*bp,
            numpy.sin(phi)*br+numpy.cos(phi)*bp,
            bz)



def test():

    r = 0.5
    phi = numpy.pi/180.0*30.0

    

    

"""
Order of transformations:
1. Flattening
2. Latitude
3. Longitude
4. Tilt
5. Twist
6. Pancaking
"""
 