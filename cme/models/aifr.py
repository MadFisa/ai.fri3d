
import numpy
import cme.cs as cs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import fractions
import scipy.special

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

    def axis_pol(self, phi):
        return self._radius_tor* \
               numpy.cos(self._coeff_angle*phi)**self._coeff_flat

    def int_axis_pol(self, phi):
        frac = fractions.Fraction(str(self._coeff_flat))
        num = float(frac.numerator)
        denum = float(frac.denominator)
        return -denum*numpy.sign(-phi)*numpy.abs(numpy.sin(self._coeff_angle*phi))/ \
               numpy.sin(self._coeff_angle*phi)* \
               numpy.cos(self._coeff_angle*phi)**((num+denum)/denum)* \
               scipy.special.hyp2f1(0.5, (num+denum)/2.0/denum, 
                                    0.5*(num/denum+3.0), 
                                    numpy.cos(self._coeff_angle*phi)**2)/ \
               (self._coeff_angle*(num+denum))

    def btot_unit_cyl(self, r):
        return 0.0 if r > 1.0 else self._b0* \
                                   numpy.exp(-(r/self._sigma)**2/2.0)

    def b_line(self, r0, phi0, s):
        # 0. no deformations
        b = self.btot_unit_cyl(r0)
        z = s
        r = numpy.ones(len(z))*r0
        phi = numpy.ones(len(z))*phi0
        # 1. constant twist
        # todo: add helicity and polarity
        phi = phi+s*self._twist*numpy.pi*2.0
        # 2. test taper
        r = r-numpy.abs(s-0.5)*1.0
        x, y, z = cs.cyl2cart(r, phi, z)
        return (x, y, z, b)

def demo():
    fr = AIFR()
    fr.set_b0(10.0)
    fr.set_h(1.0)
    fr.set_twist(5.0)

    r0 = 0.5
    phi0 = numpy.pi/6.0
    s = numpy.linspace(0.0, 1.0, 500)

    x, y, z, b = fr.b_line(r0, phi0, s)
    print b
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z)
    fig.show()

def test():
    fr = AIFR()
    fr.set_b0(10.0)
    fr.set_h(1.0)
    fr.set_twist(5.0)
    fr.set_radius_tor(1.0)
    fr.set_half_width(numpy.pi/6.0)
    fr.set_coeff_flat(0.5)

    phi = numpy.linspace(-fr._half_width, fr._half_width, 500)
    s = fr.int_axis_pol(phi)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(phi, s)
    fig.show()

# 1. use gaussian for absolute magnetic field
# 2. twist the magnetic field with constant (or variable) twist
