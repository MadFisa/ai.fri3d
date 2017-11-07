"""Defines the FRi3D class which describes CME structure and Evolution
class which handles its evolution.
"""
import os
import math
import pickle
import numpy as np
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator
from scipy.integrate import fixed_quad
from scipy.optimize import minimize_scalar
from astropy import constants as c
from astropy import units as u
from ai.shared import cs

class FRi3D:
    """FRi3D model class. It provides static description of the model.
    """
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=E1101
    # pylint: disable=C0103
    def __init__(self, **kwargs):
        self._latitude = None
        self._longitude = None
        self._toroidal_height = None
        self._poloidal_height = None
        self._half_width = None
        self._coeff_angle = None
        self._tilt = None
        self._flattening = None
        self._pancaking = None
        self._skew = None
        self._twist = None
        self._flux = None
        self._sigma = None
        self._unit_b = None
        self._polarity = None
        self._chirality = None
        self._interpolator_axis_length = None
        self._interpolator_axis_phi = None
        self.latitude = kwargs.get('latitude', u.deg.to(u.rad, 0))
        self.longitude = kwargs.get('longitude', u.deg.to(u.rad, 0))
        self.toroidal_height = kwargs.get('toroidal_height', u.au.to(u.m, 1))
        self.poloidal_height = kwargs.get('poloidal_height', u.au.to(u.m, 0.2))
        self.half_width = kwargs.get('half_width', u.deg.to(u.rad, 40))
        self.tilt = kwargs.get('tilt', u.deg.to(u.rad, 0))
        self.flattening = kwargs.get('flattening', 0.5)
        self.pancaking = kwargs.get(
            'pancaking',
            np.arctan(self.poloidal_height/self.toroidal_height)
        )
        self.skew = kwargs.get('skew', u.deg.to(u.rad, 0))
        self.twist = kwargs.get('twist', 1)
        self.flux = kwargs.get('flux', 5e14)
        self.sigma = kwargs.get('sigma', 2)
        self.polarity = kwargs.get('polarity', 1)
        self.chirality = kwargs.get('chirality', 1)
        self._location_interpolator_axis_length = kwargs.get(
            '_location_interpolator_axis_length',
            os.path.join(
                os.path.realpath(
                    os.path.join(os.getcwd(), os.path.dirname(__file__))
                ),
                'FRi3D__interpolator_axis_length.pkl'
            )
        )
        self._location_interpolator_axis_phi = kwargs.get(
            '_location_interpolator_axis_phi',
            os.path.join(
                os.path.realpath(
                    os.path.join(os.getcwd(), os.path.dirname(__file__))
                ),
                'FRi3D__interpolator_axis_phi.pkl'
            )
        )
        try:
            self._interpolator_axis_length = pickle.load(
                open(self._location_interpolator_axis_length, 'rb')
            )
            self._interpolator_axis_phi = pickle.load(
                open(self._location_interpolator_axis_phi, 'rb')
            )
        except FileNotFoundError:
            self._init_axis_interpolators()

    @property
    def latitude(self):
        """Get orientation latitude."""
        return self._latitude
    @latitude.setter
    def latitude(self, latitude):
        """Set orientation latitude."""
        self._latitude = subtract_period(latitude, np.pi*2)

    @property
    def longitude(self):
        """Get orientation longitude."""
        return self._longitude
    @longitude.setter
    def longitude(self, longitude):
        """Set orientation longitude."""
        self._longitude = subtract_period(longitude, np.pi*2)

    @property
    def toroidal_height(self):
        """Get toroidal height."""
        return self._toroidal_height
    @toroidal_height.setter
    def toroidal_height(self, toroidal_height):
        """Set toroidal height."""
        if toroidal_height > 0:
            self._toroidal_height = toroidal_height
        else:
            raise ValueError('Toroidal height should be positive.')

    @property
    def poloidal_height(self):
        """Get poloidal height."""
        return self._poloidal_height
    @poloidal_height.setter
    def poloidal_height(self, poloidal_height):
        """Set poloidal height."""
        if poloidal_height > 0:
            self._poloidal_height = poloidal_height
        else:
            raise ValueError('Poloidal height should be positive.')

    @property
    def half_width(self):
        """Get half width."""
        return self._half_width
    @half_width.setter
    def half_width(self, half_width):
        """Set half width explicitly and width coefficient implicitly.
        """
        if half_width > 0 and half_width < np.pi*2:
            self._half_width = half_width
            self._coeff_angle = np.pi/2/self.half_width
        else:
            raise ValueError('Half width should positive and less than 2pi.')

    @property
    def coeff_angle(self):
        """Get width coefficient."""
        return self._coeff_angle

    @property
    def tilt(self):
        """Get tilt."""
        return self._tilt
    @tilt.setter
    def tilt(self, tilt):
        """Set tilt."""
        self._tilt = subtract_period(tilt, np.pi*2)

    @property
    def flattening(self):
        """Get flattening coefficient."""
        return self._flattening
    @flattening.setter
    def flattening(self, flattening):
        """Set flattening coefficient."""
        if flattening > 0 and flattening < 1:
            self._flattening = flattening
        else:
            raise ValueError(
                'Flattening should be greater than 0 and less than 1.'
            )

    @property
    def pancaking(self):
        """Get pancaking (a.k.a., half height)."""
        return self._pancaking
    @pancaking.setter
    def pancaking(self, pancaking):
        """Set pancaking (a.k.a., half height)."""
        if pancaking > 0 and pancaking < np.pi:
            self._pancaking = pancaking
        else:
            raise ValueError(
                'Pancaking should be greater than 0 and less than pi.'
            )

    @property
    def skew(self):
        """Get skewing angle."""
        return self._skew
    @skew.setter
    def skew(self, skew):
        """Set skewing angle."""
        self._skew = skew

    @property
    def twist(self):
        """Get twist."""
        return self._twist
    @twist.setter
    def twist(self, twist):
        """Set twist."""
        if twist >= 0.0:
            self._twist = twist
        else:
            self._twist = -twist
            self._chirality *= -1

    @property
    def flux(self):
        """Get magnetic flux."""
        return self._flux
    @flux.setter
    def flux(self, flux):
        """Set magnetic flux."""
        if flux > 0:
            self._flux = flux
            if self.sigma is not None:
                self._unit_b = self.flux/(2.0*np.pi*self.sigma**2)
        else:
            raise ValueError('Flux should be positive.')

    @property
    def sigma(self):
        """Get sigma from Gaussian distribution of total magnetic field.
        """
        return self._sigma
    @sigma.setter
    def sigma(self, sigma):
        """Set sigma for Gaussian distribution of total magnetic field.
        """
        if sigma > 0:
            self._sigma = sigma
            if self.flux is not None:
                self._unit_b = self.flux/(2.0*np.pi*self.sigma**2)
        else:
            raise ValueError('Sigma should be positive.')

    @property
    def polarity(self):
        """Get polarity."""
        return self._polarity
    @polarity.setter
    def polarity(self, polarity):
        """Set polarity."""
        if polarity == 1 or polarity == -1:
            self._polarity = polarity
        else:
            raise ValueError('Polarity should be +1 or -1.')

    @property
    def chirality(self):
        """Get chirality."""
        return self._chirality
    @chirality.setter
    def chirality(self, chirality):
        """Set chirality."""
        if chirality == 1 or chirality == -1:
            self._chirality = chirality
        else:
            raise ValueError('Chirality should be +1 or -1.')

    def _vanilla_axis_height(
            self,
            phi,
            toroidal_height=None,
            coeff_angle=None,
            flattening=None):
        """Evaluate the axis function r(phi) in polar coordinates. Note
        that rotational skewing is not taken into account.
        """
        if toroidal_height is None:
            toroidal_height = self.toroidal_height
        if coeff_angle is None:
            coeff_angle = self.coeff_angle
        if flattening is None:
            flattening = self.flattening
        return toroidal_height*np.abs(np.cos(coeff_angle*phi))**flattening

    def _vanilla_axis_dheight(
            self,
            phi,
            toroidal_height=None,
            coeff_angle=None,
            flattening=None):
        """Evaluate the derivative of the axis function dr/d(phi). Note
        that rotational skewing is not taken into account.
        """
        if toroidal_height is None:
            toroidal_height = self.toroidal_height
        if coeff_angle is None:
            coeff_angle = self.coeff_angle
        if flattening is None:
            flattening = self.flattening
        return(
            flattening*coeff_angle*np.abs(np.tan(coeff_angle*phi))
            *self._vanilla_axis_height(
                phi,
                toroidal_height=toroidal_height,
                coeff_angle=coeff_angle,
                flattening=flattening
            )
        )

    def _vanilla_axis_distance(self, phi, r_sc, phi_sc):
        """Find the distance to the given point of the axis (phi) from
        a arbitrary point in space (r_sc, phi_sc)
        """
        return(
            (
                self._vanilla_axis_height(phi)*np.cos(phi)
                -r_sc*np.cos(phi_sc)
            )**2+
            (
                self._vanilla_axis_height(phi)*np.sin(phi)
                -r_sc*np.sin(phi_sc)
            )**2
        )

    def _vanilla_axis_min_distance(self, r_sc, phi_sc):
        """Estimate the minimal distance to the axis from an arbitrary
        point in space (r_sc, phi_sc).
        """
        phi = minimize_scalar(
            lambda phi: self._vanilla_axis_distance(phi, r_sc, phi_sc),
            bounds=[-self.half_width, self.half_width],
            method='Bounded'
        )
        return(self._vanilla_axis_distance(phi, r_sc, phi_sc), phi)

    def _vanilla_axis_tan(self, phi):
        """Evaluate tangent angle relative to the axis at a given
        location.
        """
        return np.arctan(
            (-1)*self.coeff_angle*self.flattening*np.tan(self.coeff_angle*phi)
        )

    def _vanilla_axis_dlength(
            self,
            phi,
            toroidal_height=None,
            coeff_angle=None,
            flattening=None):
        """Evaluate derivative of the axis length ds/d(phi). Note that
        rotational skewing is not taken into account.
        """
        if toroidal_height is None:
            toroidal_height = self.toroidal_height
        if coeff_angle is None:
            coeff_angle = self.coeff_angle
        if flattening is None:
            flattening = self.flattening
        return(
            self._vanilla_axis_height(
                phi,
                toroidal_height=toroidal_height,
                coeff_angle=coeff_angle,
                flattening=flattening
            )*np.sqrt(
                1+coeff_angle**2*flattening**2
                *np.tan(coeff_angle*phi)**2
            )
        )

    def _vanilla_axis_length(self, phi):
        """Evaluate length of the axis. It is an estimation and also
        does not take into account rotational skewing.
        """
        return(
            self.toroidal_height
            *self._interpolator_axis_length(
                (
                    self.coeff_angle*phi,
                    self.coeff_angle,
                    self.flattening
                )
            )
        )

    def _vanilla_axis_phi(self, length):
        """Evaluate polar coordinate of the axis as a function of its
        length. It is an estimation and also does not take into account
        rotational skewing.
        """
        return(
            self._interpolator_axis_phi(
                (
                    length/self._vanilla_axis_length(np.pi/2/self.coeff_angle),
                    self.coeff_angle,
                    self.flattening
                )
            )/self.coeff_angle
        )

    def _init_axis_interpolators(self, ratio=1-1e-5):
        """Initialize the axis interpolators:
        1. length=function(coeff_angle*phi, coeff_angle, flattening)
        2. coeff_angle*phi=function(
            relativelength,
            coeff_angle,
            flattening
        )
        for a curve defined in polar coordinates as
        r=cos(coeff_angle*phi)^flattening. Interpolation is performed
        along the following variable ranges:
        coeff_angle*phi = [-pi/2, pi/2]
        coeff_angle = [1, 18] (half_angle = [pi/36, pi/2])
        flattening = [0.1, 1.0]
        relative_length = [0, 1]
        """
        def integrate_length(
                coeff_angle_phi,
                coeff_angle,
                flattening,
                ratio=ratio):
            """Estimate length along axis by numerical integration."""
            if coeff_angle_phi <= -np.pi/2*ratio:
                length = self._vanilla_axis_height(
                    coeff_angle_phi/coeff_angle,
                    toroidal_height=1,
                    coeff_angle=coeff_angle,
                    flattening=flattening
                )
            elif coeff_angle_phi < np.pi/2*ratio:
                length = (
                    self._vanilla_axis_height(
                        -np.pi/2*ratio/coeff_angle,
                        toroidal_height=1,
                        coeff_angle=coeff_angle,
                        flattening=flattening
                    )+fixed_quad(
                        lambda coeff_angle_phi: self._vanilla_axis_dlength(
                            coeff_angle_phi/coeff_angle,
                            toroidal_height=1,
                            coeff_angle=coeff_angle,
                            flattening=flattening)/coeff_angle,
                        -np.pi/2*ratio,
                        coeff_angle_phi,
                        n=1000
                    )[0]
                )
            else:
                length = (
                    2*self._vanilla_axis_height(
                        -np.pi/2*ratio/coeff_angle,
                        toroidal_height=1,
                        coeff_angle=coeff_angle,
                        flattening=flattening
                    )+fixed_quad(
                        lambda coeff_angle_phi: self._vanilla_axis_dlength(
                            coeff_angle_phi/coeff_angle,
                            toroidal_height=1,
                            coeff_angle=coeff_angle,
                            flattening=flattening)/coeff_angle,
                        -np.pi/2*ratio,
                        np.pi/2*ratio,
                        n=1000
                    )[0]-self._vanilla_axis_height(
                        coeff_angle_phi/coeff_angle,
                        toroidal_height=1,
                        coeff_angle=coeff_angle,
                        flattening=flattening
                    )
                )
            return length

        coeff_angle_phi_array = np.linspace(
            -np.pi/2,
            np.pi/2,
            100
        )
        coeff_angle_array = np.linspace(1, 18, 100)
        flattening_array = np.linspace(0.1, 1, 100)

        coeff_angle_phi_grid, coeff_angle_grid, flattening_grid = np.meshgrid(
            coeff_angle_phi_array,
            coeff_angle_array,
            flattening_array,
            indexing='ij'
        )
        v_integrate_length = np.vectorize(
            integrate_length,
            otypes=[np.float64]
        )
        length_grid = v_integrate_length(
            coeff_angle_phi_grid,
            coeff_angle_grid,
            flattening_grid
        )
        self._interpolator_axis_length = RegularGridInterpolator(
            (coeff_angle_phi_array, coeff_angle_array, flattening_array),
            length_grid
        )
        relative_length_grid = length_grid/self._interpolator_axis_length(
            (
                np.pi/2,
                coeff_angle_grid,
                flattening_grid
            )
        )
        tmp_interpolator_axis_phi = LinearNDInterpolator(
            (
                np.ravel(relative_length_grid),
                np.ravel(coeff_angle_grid),
                np.ravel(flattening_grid)
            ),
            np.ravel(coeff_angle_phi_grid),
            fill_value=-np.pi/2
        )
        relative_length_array = np.linspace(0, 1, 100)
        relative_length_grid, coeff_angle_grid, flattening_grid = np.meshgrid(
            relative_length_array,
            coeff_angle_array,
            flattening_array,
            indexing='ij'
        )
        coeff_angle_phi_grid = tmp_interpolator_axis_phi(
            relative_length_grid,
            coeff_angle_grid,
            flattening_grid
        )
        self._interpolator_axis_phi = RegularGridInterpolator(
            (relative_length_array, coeff_angle_array, flattening_array),
            coeff_angle_phi_grid
        )
        with open(self._location_interpolator_axis_length, 'wb') as output:
            pickle.dump(
                self._interpolator_axis_length,
                output,
                pickle.HIGHEST_PROTOCOL
            )
        with open(self._location_interpolator_axis_phi, 'wb') as output:
            pickle.dump(
                self._interpolator_axis_phi,
                output,
                pickle.HIGHEST_PROTOCOL
            )

    def shell(
            self,
            relative_length=np.linspace(0, 1, 50),
            phi=np.linspace(0.0, np.pi*2.0, 24)):
        """Evaluate the 3D shell of the flux rope.
        relative_length defines the sampling along the axis,
        phi defines the sampling of the cross-section.
        """
        relative_length = np.array(relative_length, copy=False, ndmin=1)
        phi = np.array(phi, copy=False, ndmin=1)
        # relative_length is the length along the axis from 0 to 1
        relative_length = np.transpose(np.tile(relative_length, (phi.size, 1)))
        phi = np.tile(phi, (relative_length.shape[0], 1))
        # extend to full axis length
        z = relative_length*self._vanilla_axis_length(self.half_width)
        # apply tapering
        r = np.ones(relative_length.shape)
        r = (
            r*self.poloidal_height
            *(
                self._vanilla_axis_height(self._vanilla_axis_phi(z))
                /self.toroidal_height
            )
        )
        x, y, z = cs.cyl2cart(r, phi, z)
        # rotate towards X axis
        T = cs.mx_rot_y(np.pi/2)
        x, y, z = cs.mx_apply(T, x, y, z)
        # remove (tiny) values less than 0, numerical issues
        x[x < 0] = 0
        # apply bending
        phi = self._vanilla_axis_phi(x)
        r = self._vanilla_axis_height(phi)
        t = self._vanilla_axis_tan(phi)
        x = r*np.cos(phi)+np.sin(t-phi-np.pi/2)*y
        y = r*np.sin(phi)+np.cos(t-phi-np.pi/2)*y
        # apply pancaking
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
        phi += self.skew*(1-r/self.toroidal_height)
        x, y, z = cs.cyl2cart(r, phi, z)
        return(x, y, z)

    def line(self, r=0.0, phi=0.0, relative_length=np.linspace(0.0, 1.0, 50)):
        """Evaluate the 3D magnetic field line of the flux rope.
        relative_length defines the sampling along the axis,
        (r, phi) define the coordinates of the origin of the line.
        """
        s = np.array(relative_length, copy=False, ndmin=1)
        phi = np.ones(s.size)*phi
        # twist
        phi += s*self.twist*np.pi*2.0*self.chirality
        # elongation
        z = s*self._vanilla_axis_length(self.half_width)
        # distance to axis from origin
        R = self._vanilla_axis_height(self._vanilla_axis_phi(z))
        # cross-section radial size in the FR plane
        rx = R*self.poloidal_height/self.toroidal_height
        # cross-section radial size perp to FR plane
        ry = R*self.pancaking
        # coefficient of flux decay
        kappa = rx*ry
        # tapering
        r *= rx
        # magnetic field
        b = self._unit_b/kappa*np.exp(
            -((r/rx)**2)/2.0/self.sigma**2
        )
        x, y, z = cs.cyl2cart(r, phi, z)
        # rotation to x
        T = cs.mx_rot_y(np.pi/2.0)
        x, y, z = cs.mx_apply(T, x, y, z)
        # bending
        x[x < 0] = 0
        phi = self._vanilla_axis_phi(x)
        r = self._vanilla_axis_height(phi)
        t = self._vanilla_axis_tan(phi)
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

    # from ai.fri3d.line import line
    # from ai.fri3d.data import data
    # from ai.fri3d.impact import impact

"""
class Evolution:
    def __init__(self,
        latitude=lambda t: u.deg.to(u.rad, 0.0),
        longitude=lambda t: u.deg.to(u.rad, 0.0),
        toroidal_height=lambda t: 
            u.Unit('km/s').to(u.Unit('m/s'), 450.0)*t+u.au.to(u.m, 0.7), 
        poloidal_height=lambda t: u.au.to(u.m, 0.2), 
        half_width=lambda t: u.deg.to(u.rad, 40.0), 
        tilt=lambda t: u.deg.to(u.rad, 0.0), 
        flattening=lambda t: 0.5, 
        pancaking=lambda t: u.deg.to(u.rad, 20.0), 
        skew=lambda t: u.deg.to(u.rad, 0.0), 
        twist=lambda t: 3.0, 
        flux=lambda t: 5e14,
        sigma=lambda t: 2.0,
        polarity=1.0,
        chirality=1.0,
        spline_s_phi_kind='cubic',
        spline_s_phi_n=200):
        
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

    @property
    def tilt(self):
        return self._tilt
    @tilt.setter
    def tilt(self, tilt):
        self._tilt = tilt

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
    def twist(self):
        return self._twist
    @twist.setter
    def twist(self, twist):
        self._twist = twist

    @property
    def flux(self):
        return self._flux
    @flux.setter
    def flux(self, flux):
        self._flux = flux

    @property
    def sigma(self):
        return self._sigma
    @sigma.setter
    def sigma(self, sigma):
        self._sigma = sigma

    @property
    def polarity(self):
        return self._polarity
    @polarity.setter
    def polarity(self, polarity):
        self._polarity = polarity

    @property
    def chirality(self):
        return self._chirality
    @chirality.setter
    def chirality(self, chirality):
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

    def insitu(self, t, x, y, z):
        fr = FRi3D()
        fr.polarity = self.polarity
        fr.chirality = self.chirality
        fr.spline_s_phi_kind = self.spline_s_phi_kind
        fr.spline_s_phi_n = self.spline_s_phi_n
        b = []
        v = []
        for i, t in enumerate(t):
            fr.latitude = self.latitude(t)
            fr.longitude = self.longitude(t)
            fr.toroidal_height = self.toroidal_height(t)
            fr.poloidal_height = self.poloidal_height(t)
            fr.half_width = self.half_width(t)
            fr.tilt = self.tilt(t)
            fr.flattening = self.flattening(t)
            fr.pancaking = self.pancaking(t)
            fr.skew = self.skew(t)
            fr.twist = self.twist(t)
            fr.flux = self.flux(t)
            fr.sigma = self.sigma(t)
            if i == 0:
                fr.toroidal_height = 1.0
                fr.init()
                fr._unit_spline_initial_axis_s_phi = \
                    fr._spline_initial_axis_s_phi
                fr.toroidal_height = self.toroidal_height(t)
                fr.init()
            # valid if flattening, half width and flux stay constant
            fr._spline_initial_axis_s_phi = lambda s: \
                fr._unit_spline_initial_axis_s_phi(s/fr.toroidal_height)
            # print(
                # 'Latitude: ', u.rad.to(u.deg, fr.latitude),
                # 'Longitude: ', u.rad.to(u.deg, fr.longitude),
                # 'Toroidal height: ', u.m.to(u.au, fr.toroidal_height),
                # 'Poloidal height: ', u.m.to(u.au, fr.poloidal_height),
                # 'Half width: ', u.rad.to(u.deg, fr.half_width),
                # 'Tilt: ', u.rad.to(u.deg, fr.tilt),
            # )
            b_, c_ = fr.data(
                x(t) if callable(x) else x, 
                y(t) if callable(y) else y, 
                z(t) if callable(z) else z
            )
            if b_.size == 0:
                b_ = np.array([0.0, 0.0, 0.0])
            if c_.size == 0:
                c_ = np.array([0.0, 0.0])
            b.append(b_.ravel())
            v.append(
                c_[0]*(self.toroidal_height(t)-self.toroidal_height(t-1))+
                c_[1]*(self.poloidal_height(t)-self.poloidal_height(t-1))
            )
        return (np.array(b), np.array(v))

    def impact(self, t, x, y, z):
        fr = FRi3D()
        fr.polarity = self.polarity
        fr.chirality = self.chirality
        fr.spline_s_phi_kind = self.spline_s_phi_kind
        fr.spline_s_phi_n = self.spline_s_phi_n
        impacts = []
        times = []
        for i, t in enumerate(t):
            fr.latitude = self.latitude(t)
            fr.longitude = self.longitude(t)
            fr.toroidal_height = self.toroidal_height(t)
            fr.poloidal_height = self.poloidal_height(t)
            fr.half_width = self.half_width(t)
            fr.tilt = self.tilt(t)
            fr.flattening = self.flattening(t)
            fr.pancaking = self.pancaking(t)
            fr.skew = self.skew(t)
            fr.twist = self.twist(t)
            fr.flux = self.flux(t)
            fr.sigma = self.sigma(t)
            if i == 0:
                fr.toroidal_height = 1.0
                fr.init()
                fr._unit_spline_initial_axis_s_phi = \
                    fr._spline_initial_axis_s_phi
                fr.toroidal_height = self.toroidal_height(t)
                fr.init()
            fr._spline_initial_axis_s_phi = lambda s: \
                fr._unit_spline_initial_axis_s_phi(s/fr.toroidal_height)
            impact, _, _, _, _, _, _ = fr.impact(x, y, z)
            impacts.append(impact)
            times.append(t)
        impacts = np.array(impacts)
        times = np.array(times)
        index = np.argmin(impacts)
        return (impacts[index], times[index])

    def map(self, t, x, y, z, 
        dx=u.au.to(u.m, np.linspace(-0.2, 0.2, 100)),
        dy=u.au.to(u.m, np.linspace(-0.2, 0.2, 100))):
        
        _, t = self.impact(t, x, y, z)

        fr = FRi3D()
        fr.polarity = self.polarity
        fr.chirality = self.chirality
        fr.spline_s_phi_kind = self.spline_s_phi_kind
        fr.spline_s_phi_n = self.spline_s_phi_n
        fr.latitude = self.latitude(t)
        fr.longitude = self.longitude(t)
        fr.toroidal_height = self.toroidal_height(t)
        fr.poloidal_height = self.poloidal_height(t)
        fr.half_width = self.half_width(t)
        fr.tilt = self.tilt(t)
        fr.flattening = self.flattening(t)
        fr.pancaking = self.pancaking(t)
        fr.skew = self.skew(t)
        fr.twist = self.twist(t)
        fr.flux = self.flux(t)
        fr.sigma = self.sigma(t)
        fr.toroidal_height = 1.0
        fr.init()
        fr._unit_spline_initial_axis_s_phi = \
            fr._spline_initial_axis_s_phi
        fr.toroidal_height = self.toroidal_height(t)
        fr.init()

        _, xa, ya, za, xt, yt, zt = fr.impact(x, y, z)
        vtan = np.array([np.mean(xt), np.mean(yt), np.mean(zt)])
        if np.dot(vtan, fr.data(xa, ya, za)) < 0.0:
            vtan = -vtan
        # vtan = fr.data(xa, ya, za)
        # vtan /= np.linalg.norm(vtan)
        vsc = np.array([x, y, z])
        vsc /= np.linalg.norm(vsc)

        vmcy = np.cross(vtan, vsc)
        vmcy /= np.linalg.norm(vmcy)
        if vmcy[0] < 0.0:
            vmcy = -vmcy
        vmcx = np.cross(vmcy, vtan)
        vmcx /= np.linalg.norm(vmcx)

        print(vmcx, vmcy, vtan)

        xg = np.zeros([dx.size, dy.size])
        yg = np.zeros([dx.size, dy.size])
        zg = np.zeros([dx.size, dy.size])

        for i in range(dx.size):
            for k in range(dy.size):
                p = np.array([x, y, z])+dx[i]*vmcx+dy[k]*vmcy
                xg[i,k] = p[0]
                yg[i,k] = p[1]
                zg[i,k] = p[2]
        print(
            xg.shape, xg.flatten().shape,
            yg.shape, yg.flatten().shape,
            zg.shape, zg.flatten().shape
        )
        b = fr.data(xg.flatten(), yg.flatten(), zg.flatten())
        print(b.shape)
        bmap = np.zeros(b.shape[0])
        for i in range(b.shape[0]):
            bmap[i] = np.dot(b[i,:], vtan)
        bmap = np.reshape(bmap, [dx.size, dy.size])

        return bmap.T
"""
def subtract_period(value, period):
    """Reduce angle by period."""
    return value-math.copysign(value, 1)*(math.fabs(value)//period)*period
