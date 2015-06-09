
import numpy

class AIFX:
    _lat = None
    _lon = None
    _tilt = None
    _half_width = None
    _coeff_angle = None
    _radius_tor = None
    _radius_pol = None
    _coeff_flat = None
    _coeff_twist = None
    _radius_corot = None
    _coeff_panc = None
    _grid_tor = None
    _grid_pol = None
    _helicity = None
    _polarity = None
    _core_field = None

    def set_radius_tor(self, new_radius_tor):
        global _radius_tor
        _radius_tor = new_radius_tor

    def set_radius_pol(self, new_radius_pol):
        global _radius_pol
        _radius_pol = new_radius_pol

    def set_half_width(self, new_half_width):
        global _half_width
        global _coeff_angle
        _half_width = new_half_width
        _coeff_angle = numpy.pi/2.0/_half_width

    def set_coeff_flat(self, new_coeff_flat):
        global _coeff_flat
        _coeff_flat = new_coeff_flat

    def set_lat(self, new_lat):
        global _lat
        _lat = new_lat

    def set_lon(self, new_lon):
        global _lon
        _lon = new_lon

    def set_tilt(self, new_tilt):
        global _tilt
        _tilt = new_tilt

    def _axis_r(self, phi):
        return _radius_tor*numpy.cos(_coeff_angle*phi)**_coeff_flat

    def _axis_tangent(self, phi):
        return numpy.arctan2(1.0, _coeff_angle*_coeff_flat*
                                  numpy.tan(_coeff_angle*phi))

    def _shell_x(self, phi, theta):
        return _axis_r(phi)*(_radius_pol/_radius_tor*numpy.cos(theta)*
                             numpy.sin(_axis_tangent(phi)-phi)+
                             numpy.cos(phi))

    def _shell_y(self, phi, theta):
        return _axis_r(phi)*(_radius_pol/_radius_tor*numpy.cos(theta)*
                             numpy.cos(_axis_tangent(phi)-phi)+
                             numpy.sin(phi))

    def _shell_z(self, phi, theta):
        return _axis_r(phi)*_radius_pol/_radius_tor*numpy.sin(theta)

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

"""
Order of transformations:
1. Flattening
2. Latitude
3. Longitude
4. Tilt
5. Twist
6. Pancaking
"""
