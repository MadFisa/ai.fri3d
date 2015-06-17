
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class AIFR:
    _radius_tor = 1.0
    _radius_pol = 0.1
    _half_width = numpy.pi/6.0
    _coeff_angle = 3.0
    _coeff_flat = 0.5
    _lat = 0.0
    _lon = 0.0
    _tilt = 0.0
    # _coeff_twist = None
    # _radius_corot = None
    # _coeff_panc = None
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

    def _axis_r(self, phi):
        return self._radius_tor* \
               numpy.cos(self._coeff_angle*phi)**self._coeff_flat

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

    def axis(self, phi):
        x0, y0, z0 = self.sp2cart(self._axis_r(phi), 0.0, phi)
        T = self.mx_rot(self._lat, -self._lon, -self._tilt)
        x = T[0,0]*x0+T[0,1]*y0+T[0,2]*z0
        y = T[1,0]*x0+T[1,1]*y0+T[1,2]*z0
        z = T[2,0]*x0+T[2,1]*y0+T[2,2]*z0
        return (x, y, z)

    def shell(self, theta, phi):
        theta, phi = numpy.meshgrid(theta, phi)
        x0 = self._shell_x(theta, phi)
        y0 = self._shell_y(theta, phi)
        z0 = self._shell_z(theta, phi)
        T = self.mx_rot(self._lat, -self._lon, -self._tilt)
        x = T[0,0]*x0+T[0,1]*y0+T[0,2]*z0
        y = T[1,0]*x0+T[1,1]*y0+T[1,2]*z0
        z = T[2,0]*x0+T[2,1]*y0+T[2,2]*z0
        return (x, y, z)

def demo_axis():
    fr = AIFR()
    fr.set_lat(numpy.pi/6.0)
    fr.set_lon(numpy.pi/6.0)
    fr.set_tilt(numpy.pi/6.0)
    phi = numpy.linspace(-numpy.pi/6.0, numpy.pi/6.0, 30)
    x, y, z = fr.axis(phi)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z)
    fig.show()

def demo_shell():
    fr = AIFR()
    fr.set_lat(numpy.pi/6.0)
    fr.set_lon(numpy.pi/6.0)
    fr.set_tilt(numpy.pi/6.0)
    fr.set_radius_pol(0.2)
    theta = numpy.linspace(0.0, 2.0*numpy.pi, 30)
    phi = numpy.linspace(-numpy.pi/6.0, numpy.pi/6.0, 60)
    x, y, z = fr.shell(theta, phi)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(x, y, z)
    fig.show()

    
"""
Order of transformations:
1. Flattening
2. Latitude
3. Longitude
4. Tilt
5. Twist
6. Pancaking
"""
