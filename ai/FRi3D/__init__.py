
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
from astropy import constants as c
from astropy import units as u

# definition of nT unit
u.nT = u.def_unit('nT', 1e-9*u.T)

class FRi3D:
    def __init__(
        self,
        latitude=u.deg.to(u.rad, 0.0), 
        longitude=u.deg.to(u.rad, 0.0), 
        toroidal_height=u.au.to(u.m, 1.0), 
        poloidal_height=u.au.to(u.m, 0.1), 
        half_width=u.deg.to(u.rad, 40.0), 
        tilt=u.deg.to(u.rad, 0.0), 
        flattening=0.5, 
        pancaking=u.deg.to(u.rad, 20.0), 
        skew=u.deg.to(u.rad, 0.0), 
        twist=1.0, 
        flux=5e14,
        sigma=2.05,
        polarity=1.0,
        chirality=1.0,
        spline_s_phi_kind='cubic',
        spline_s_phi_n=500):
        
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
        self.sigma = sigma
        self.polarity = polarity
        self.chirality = chirality
        self.spline_s_phi_kind = spline_s_phi_kind
        self.spline_s_phi_n = spline_s_phi_n
        self.init()

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
        if toroidal_height > 0.0:
            self._toroidal_height = toroidal_height

    @property
    def poloidal_height(self):
        return self._poloidal_height
    @poloidal_height.setter
    def poloidal_height(self, poloidal_height):
        if poloidal_height > 0.0:
            self._poloidal_height = poloidal_height

    @property
    def half_width(self):
        return self._half_width
    @half_width.setter
    def half_width(self, half_width):
        if half_width > 0.0 and half_width < np.pi*2.0:
            self._half_width = half_width

    @property
    def coeff_angle(self):
        return self._coeff_angle

    @property
    def flattening(self):
        return self._flattening
    @flattening.setter
    def flattening(self, flattening):
        if flattening > 0.0 and flattening < 1.0:
            self._flattening = flattening

    @property
    def pancaking(self):
        return self._pancaking
    @pancaking.setter
    def pancaking(self, pancaking):
        if pancaking is None:
            self._pancaking = np.arctan(
                self.poloidal_height/self.toroidal_height
            )
        elif pancaking > 0.0 and pancaking < np.pi:
            self._pancaking = pancaking

    @property
    def skew(self):
        return self._skew
    @skew.setter
    def skew(self, skew):
        if skew is None:
            self._skew = 0.0
        elif skew >= 0.0:
            self._skew = skew

    @property
    def latitude(self):
        return self._latitude
    @latitude.setter
    def latitude(self, latitude):
        if latitude >= -np.pi/2.0 and latitude <= np.pi/2.0:
            self._latitude = latitude

    @property
    def longitude(self):
        return self._longitude
    @longitude.setter
    def longitude(self, longitude):
        if longitude >= -np.pi and longitude <= np.pi:
            self._longitude = longitude

    @property
    def tilt(self):
        return self._tilt
    @tilt.setter
    def tilt(self, tilt):
        if tilt >= -np.pi and tilt <= np.pi:
            self._tilt = tilt    

    @property
    def flux(self):
        return self._flux
    @flux.setter
    def flux(self, flux):
        if flux > 0.0:
            self._flux = flux

    @property
    def sigma(self):
        return self._sigma
    @sigma.setter
    def sigma(self, sigma):
        if sigma > 0.0:
            self._sigma = sigma

    @property
    def polarity(self):
        return self._polarity
    @polarity.setter
    def polarity(self, polarity):
        if polarity == 1.0 or polarity == -1.0:
            self._polarity = polarity

    @property
    def chirality(self):
        return self._chirality
    @chirality.setter
    def chirality(self, chirality):
        if chirality == 1.0 or chirality == -1.0:
            self._chirality = chirality

    @property
    def spline_s_phi_kind(self):
        return self._spline_s_phi_kind
    @spline_s_phi_kind.setter
    def spline_s_phi_kind(self, spline_s_phi_kind):
        self._spline_s_phi_kind = spline_s_phi_kind

    @property
    def spline_s_phi_n(self):
        return self._spline_s_phi_n
    @spline_s_phi_n.setter
    def spline_s_phi_n(self, spline_s_phi_n):
        self._spline_s_phi_n = spline_s_phi_n

    def init(self):
        self._coeff_angle = np.pi/2.0/self.half_width
        self._unit_b = self.flux/(2.0*np.pi*self.sigma**2)
        self._init_spline_initial_axis_s_phi()

    # initilize spline phi(s)
    def _init_spline_initial_axis_s_phi(self):
        phi = np.linspace(
            -self.half_width, 
            self.half_width, 
            self.spline_s_phi_n
        )
        s = np.array([self._initial_axis_s(p) for p in phi])
        self._spline_initial_axis_s_phi = interp1d(
            s, phi, 
            kind=self.spline_s_phi_kind,
            bounds_error=False,
            fill_value=(-self.half_width, self.half_width)
        )

    # r(phi) for undeformed axis
    def _initial_axis_r(self, phi):
        return np.nan_to_num(
            self.toroidal_height*
            np.cos(self.coeff_angle*phi)**self.flattening
        )

    #dr/dphi for undeformed axis
    def _initial_axis_dr(self, phi):
        return (
            -self.coeff_angle*self.toroidal_height*self.flattening*
            np.cos(self.coeff_angle*phi)**(self.flattening-1.0)*
            np.sin(self.coeff_angle*phi)
        )

    # distance to undeformed axis from (r0,phi0)
    def _initial_axis_l(self, phi, r0, phi0):
        return (
            (self._initial_axis_r(phi)*np.cos(phi)-r0*np.cos(phi0))**2+
            (self._initial_axis_r(phi)*np.sin(phi)-r0*np.sin(phi0))**2
        )

    # find phi which gives the minimum distance to undeformed axis
    def _initial_axis_min_l_phi(self, r0, phi0):
        res = scipy.optimize.minimize_scalar(
            lambda phi: self._initial_axis_l(phi, r0, phi0),
            bounds=[-self.half_width, self.half_width],
            method='Brent'
        )
        return res.x

    # tangent to undeformed axis at a given phi
    def _initial_axis_tan(self, phi):
        return np.arctan(
            -self.coeff_angle*self.flattening*np.tan(self.coeff_angle*phi)
        )

    # ds/dphi of of underformed axis at a given phi
    def _initial_axis_ds(self, phi):
        return np.sqrt(
            self._initial_axis_r(phi)**2+
            self._initial_axis_dr(phi)**2
        )

    # length of axis at a given phi
    def _initial_axis_s(self, phi):
        s = quad(self._initial_axis_ds, -self.half_width, phi)
        return s[0]

    from ai.FRi3D.shell import shell
    from ai.FRi3D.line import line
