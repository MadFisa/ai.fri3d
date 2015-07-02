
import numpy
import cme.cs as cs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import scipy.special
import scipy.interpolate
import scipy.integrate

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
    _panc_angle = numpy.pi/6.0
    _skew_angle = 0.0
    _lat = 0.0
    _lon = 0.0
    _tilt = 0.0
    
    
    _coeff_panc = 0.1

    _spline_axis0_s_phi = None

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

    def set_panc_angle(self, new_panc_angle):
        self._panc_angle = new_panc_angle

    def set_lat(self, new_lat):
        self._lat = new_lat

    def set_lon(self, new_lon):
        self._lon = new_lon

    def set_tilt(self, new_tilt):
        self._tilt = new_tilt

    def set_skew_angle(self, new_skew_angle):
        self._skew_angle = new_skew_angle

    def set_coeff_panc(self, new_coeff_panc):
        self._coeff_panc = new_coeff_panc

    def init_spline_axis0_s_phi(self):
        phi = numpy.linspace(-self._half_width, self._half_width, 500)
        s = numpy.array(map(self._axis0_s, phi))
        self._spline_axis0_s_phi = scipy.interpolate.interp1d(s, phi, 
                                                              kind='cubic')

    def init_magnetic_field(self):


    def _axis0_r(self, phi):
        return numpy.nan_to_num(self._radius_tor* \
               numpy.cos(self._coeff_angle*phi)**self._coeff_flat)

    def _axis0_tan(self, phi):
        return numpy.arctan(-self._coeff_angle*
                            self._coeff_flat*
                            numpy.tan(self._coeff_angle*phi))

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
        x3, y3, z3 = cs.cyl2cart(r, phi, z)
        # 4. rotate Z to X
        T = cs.mx_rot_y(-numpy.pi/2.0)
        x4 = T[0,0]*x3+T[0,1]*y3+T[0,2]*z3
        y4 = T[1,0]*x3+T[1,1]*y3+T[1,2]*z3
        z4 = T[2,0]*x3+T[2,1]*y3+T[2,2]*z3
        # 5. bend
        phi = self._spline_axis0_s_phi(x4)
        r = self._axis0_r(phi)
        t = self._axis0_tan(phi)
        x5 = r*numpy.cos(phi)+numpy.sin(t-phi-numpy.pi/2.0)*y4
        y5 = r*numpy.sin(phi)+numpy.cos(t-phi-numpy.pi/2.0)*y4
        z5 = z4
        # 6. pancaking
        r, theta, phi = cs.cart2sp(x5, y5, z5)
        theta = theta/numpy.arctan2(self._radius_pol, self._radius_tor)* \
                self._panc_angle
        x6, y6, z6 = cs.sp2cart(r, theta, phi)
        # calculate magnetic field here
        b = self.btot_unit_cyl(r0)
        # 7. orientation
        T = cs.mx_rot(self._lat, -self._lon, -self._tilt)
        x7 = T[0,0]*x6+T[0,1]*y6+T[0,2]*z6
        y7 = T[1,0]*x6+T[1,1]*y6+T[1,2]*z6
        z7 = T[2,0]*x6+T[2,1]*y6+T[2,2]*z6
        # 8. scew
        r, phi, z = cs.cart2cyl(x7, y7, z7)
        phi = phi+self._skew_angle*r/r.max()
        x8, y8, z8 = cs.cyl2cart(r, phi, z)
        # finished
        x = x8
        y = y8
        z = z8
        return (x, y, z, b)

def demo():
    fr = AIFR()
    fr.set_b0(10.0)
    fr.set_h(1.0)
    fr.set_twist(2.0)
    fr.set_radius_tor(1.0)
    fr.set_radius_pol(0.1)
    fr.set_half_width(numpy.pi/180.0*30.0)
    fr.set_coeff_flat(0.6)
    fr.set_panc_angle(numpy.pi/180.0*30.0)
    fr.set_skew_angle(-numpy.pi/180.0*0.0)
    fr.set_lat(numpy.pi/180.0*0.0)
    fr.set_lon(numpy.pi/180.0*0.0)
    fr.set_tilt(numpy.pi/180.0*0.0)
    fr.init_spline_axis0_s_phi()

    s = numpy.linspace(1.0e-6, 1.0-1.0e-6, 500)
    
    b_max = fr.btot_unit_cyl(0.0)
    b_min = fr.btot_unit_cyl(1.0)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for r0 in numpy.arange(0.0, 1.1, 0.1):
        for phi0 in numpy.arange(0.0, numpy.pi*2.0, numpy.pi/6.0):
            x, y, z, b = fr.b_line(r0, phi0, s)
            ax.plot(x, y, z, color=plt.cm.jet((b-b_min)/(b_max-b_min)))
            max_range = numpy.array([x.max()-x.min(), 
                                     y.max()-y.min(), 
                                     z.max()-z.min()]).max()/2.0
            mean_x = x.mean()
            mean_y = y.mean()
            mean_z = z.mean()
            ax.set_xlim(0.0, mean_x+max_range)
            ax.set_ylim(mean_y-max_range, mean_y+max_range)
            ax.set_zlim(mean_z-max_range, mean_z+max_range)
    fig.show()

def test():
    fr = AIFR()
    fr.set_b0(10.0)
    fr.set_h(1.0)
    fr.set_twist(2.0)
    fr.set_radius_tor(1.0)
    fr.set_radius_pol(0.2)
    fr.set_half_width(numpy.pi/180.0*30.0)
    fr.set_coeff_flat(0.6)
    fr.set_coeff_panc(0.9)
    fr.set_skew_angle(-numpy.pi/180.0*0.0)
    fr.set_lat(numpy.pi/180.0*0.0)
    fr.set_lon(numpy.pi/180.0*0.0)
    fr.set_tilt(numpy.pi/180.0*0.0)
    fr.init_spline_axis0_s_phi()

    phi = numpy.linspace(-fr._half_width, fr._half_width, 500)
    t = fr._axis0_tan(phi)
    
    fig = plt.figure()
    ax = fig.add_subplot(221)
    ax.plot(phi, numpy.pi/2.0-t)
    ax = fig.add_subplot(222)
    ax.plot(phi, -phi-(numpy.pi/2.0-t))
    ax = fig.add_subplot(223)
    ax.plot(phi, numpy.cos(-phi-(numpy.pi/2.0-t)), 
            phi, numpy.sin(-phi-(numpy.pi/2.0-t)))
    ax = fig.add_subplot(224)
    ax.plot(phi, 1.0-numpy.abs(t)/numpy.pi*2.0)
    fig.show()
