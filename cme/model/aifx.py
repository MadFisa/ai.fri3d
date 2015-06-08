
import numpy

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

"""
Order of transformations:
1. Flattening
2. Latitude
3. Longitude
4. Tilt
5. Twist
6. Pancaking
"""

def set_radius_tor(new_radius_tor):
    global _radius_tor
    _radius_tor = new_radius_tor

def set_radius_pol(new_radius_pol):
    global _radius_pol
    _radius_pol = new_radius_pol

def set_half_width(new_half_width):
    global _half_width
    global _coeff_angle
    _half_width = new_half_width
    _coeff_angle = numpy.pi/2.0/_half_width

def set_coeff_flat(new_coeff_flat):
    global _coeff_flat
    _coeff_flat = new_coeff_flat

def set_lat(new_lat):
    global _lat
    _lat = new_lat

def set_lon(new_lon):
    global _lon
    _lon = new_lon

def set_tilt(new_tilt):
    global _tilt
    _tilt = new_tilt

def _axis_r(phi):
    return _radius_tor*numpy.cos(_coeff_angle*phi)**_coeff_flat

def _axis_tangent(phi):
    return numpy.arctan2(1.0, _coeff_angle*_coeff_flat*
                              numpy.tan(_coeff_angle*phi))

def _axis_x():

def _axis_y():

def _axis_z():

def _axis():

def _shell_x():

def _shell_y():

def _shell_z():

def _shell():

