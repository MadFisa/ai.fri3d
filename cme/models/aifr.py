
import numpy
import cme.cs as cs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import fractions
import scipy.special
from scipy.interpolate import interp1d
import scipy.integrate
import mpmath

class AIFR:
    _b0 = 1.0
    _h = 1.0
    _sigma = 2.05
    _twist = 1.0
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

    _spline_axis_s_phi = None

    def set_b0(self, new_b0):
        self._b0 = new_b0

    def set_h(self, new_h):
        self._h = new_h

    def set_sigma(self, new_sigma):
        self._sigma = new_sigma

    def set_twist(self, new_twist):
        self._twist = new_twist

    def set_radius_tor(self, new_radius_tor):
        self._radius_tor = new_radius_tor
        # self.set_spline_s_phi()

    def set_radius_pol(self, new_radius_pol):
        self._radius_pol = new_radius_pol

    def set_half_width(self, new_half_width):
        self._half_width = new_half_width
        self._coeff_angle = numpy.pi/2.0/self._half_width
        # self.set_spline_s_phi()

    def set_coeff_flat(self, new_coeff_flat):
        self._coeff_flat = new_coeff_flat
        # self.set_spline_s_phi()

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

    def init_spline_axis0_s_phi(self):
        phi = numpy.linspace(-self._half_width, self._half_width, 500)
        s = numpy.array(map(self._axis0_s, phi))
        self._spline_axis0_s_phi = interp1d(s, phi, kind='cubic')

    def _axis0_r(self, phi):
        return self._radius_tor* \
               numpy.cos(self._coeff_angle*phi)**self._coeff_flat

    def _axis0_ds(self, phi):
        a = self._coeff_angle
        n = self._coeff_flat
        
        dr = self._axis0_r(phi)*numpy.sin(phi)/ \
             numpy.sqrt(4.0*numpy.cos(a*phi)**(2.0*n)-
                        4.0*numpy.cos(phi)*numpy.cos(a*phi)**n+1.0)
        
        dp = 2.0*numpy.cos(a*phi)**n*(2.0*numpy.cos(a*phi)**n-
                                      numpy.cos(phi))/ \
             (4.0*numpy.cos(a*phi)**(2.0*n)-
              4.0*numpy.cos(phi)*numpy.cos(a*phi)**n+1.0)
        
        ds = numpy.sqrt(dr**2+(self._axis0_r(phi)*dp)**2)
        return ds

    def _axis0_s(self, phi):
        s = scipy.integrate.quad(self._axis0_ds, -self._half_width, phi)
        return s[0]

    def btot_unit_cyl(self, r):
        return 0.0 if r > 1.0 else self._b0* \
                                   numpy.exp(-(r/self._sigma)**2/2.0)

    def b_line(self, r0, phi0, s):
        # 0. no deformations
        b = self.btot_unit_cyl(r0)
        r = numpy.ones(len(s))*r0
        phi = numpy.ones(len(s))*phi0
        # 1. twist
        # todo: add helicity and polarity
        phi = phi+s*self._twist*numpy.pi*2.0
        # 2. elongation
        z = s*self._axis0_s(self._half_width)
        # 3. taper
        r = r*self._axis0_r(self._spline_axis0_s_phi(z))* \
            self._radius_pol/self._radius_tor
        # 4. bend
        x, y, z = cs.cyl2cart(r, phi, z)
        return (x, y, z, b)

def demo():
    fr = AIFR()
    fr.set_b0(10.0)
    fr.set_h(1.0)
    fr.set_twist(1.0)
    fr.set_radius_tor(1.0)
    fr.set_radius_pol(0.15)
    fr.set_half_width(numpy.pi/6.0)
    fr.set_coeff_flat(0.5)
    fr.init_spline_axis0_s_phi()

    s = numpy.linspace(0.0, 1.0, 500)
   
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for r0 in numpy.arange(0.0, 1.0, 0.2):
        for phi0 in numpy.arange(0.0, numpy.pi/6.0*11.0, numpy.pi/6.0):
            x, y, z, b = fr.b_line(r0, phi0, s)
            ax.plot(x, y, z)
    fig.show()

def test():
    fr = AIFR()
    fr.set_b0(10.0)
    fr.set_h(1.0)
    fr.set_twist(5.0)
    fr.set_radius_tor(1.0)
    fr.set_half_width(numpy.pi/6.0)
    fr.set_coeff_flat(0.1)

    phi = numpy.linspace(-fr._half_width, fr._half_width, 500)
    r = fr._axis0_r(phi)
    s = numpy.array(map(fr._axis0_s, phi))
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(s, phi)
    fig.show()

# 1. use gaussian for absolute magnetic field
# 2. twist the magnetic field with constant (or variable) twist
