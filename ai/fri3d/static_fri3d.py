"""Static FRi3D class definition.

This module defines static FRi3D class. It provides a static description
of a CME (snapshot, not varying in time).
"""
# pylint: disable=too-many-instance-attributes
# pylint: disable=E1101
# pylint: disable=C0103
# pylint: disable=C0302
import os
import pickle
import numpy as np
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator
from scipy.integrate import fixed_quad
from scipy.optimize import minimize_scalar
from astropy import units as u
from ai.shared import cs
from ai.shared.utils import subtract_period

class StaticFRi3D:
    """FRi3D model static class. It provides static description of the
    model.
    """
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
        self._reload = kwargs.get('_reload', False)
        self._n_coeff_angle_phi = kwargs.get('_n_coeff_angle_phi', 100)
        self._n_coeff_angle = kwargs.get('_n_coeff_angle', 100)
        self._n_flattening = kwargs.get('_n_flattening', 100)
        self._n_relative_length = kwargs.get('_n_relative_length', 100)
        self._ratio = kwargs.get('_ratio', 1-1e-5)
        self._location_interpolator_axis_length = kwargs.get(
            '_location_interpolator_axis_length',
            os.path.join(
                os.path.realpath(
                    os.path.join(os.getcwd(), os.path.dirname(__file__))
                ),
                '__interpolator_axis_length.pkl'
            )
        )
        self._location_interpolator_axis_phi = kwargs.get(
            '_location_interpolator_axis_phi',
            os.path.join(
                os.path.realpath(
                    os.path.join(os.getcwd(), os.path.dirname(__file__))
                ),
                '__interpolator_axis_phi.pkl'
            )
        )
        if self._reload:
            try:
                os.remove(self._location_interpolator_axis_length)
                os.remove(self._location_interpolator_axis_phi)
            except FileNotFoundError:
                pass
        try:
            self._interpolator_axis_length = pickle.load(
                open(self._location_interpolator_axis_length, 'rb')
            )
            self._interpolator_axis_phi = pickle.load(
                open(self._location_interpolator_axis_phi, 'rb')
            )
        except FileNotFoundError:
            self._init_axis_interpolators(
                n_coeff_angle_phi=self._n_coeff_angle_phi,
                n_coeff_angle=self._n_coeff_angle,
                n_flattening=self._n_flattening,
                n_relative_length=self._n_relative_length,
                ratio=self._ratio
            )

    def modify(self, **kwargs):
        """Modify the model properties."""
        allowed_attr = (
            'latitude', 'longitude', 'toroidal_height', 'poloidal_height',
            'half_width', 'tilt', 'flattening', 'pancaking', 'skew',
            'sigma', 'flux', 'polarity', 'chirality'
        )
        for k, v in kwargs.items():
            if k in allowed_attr:
                setattr(self, k, v)
            else:
                raise KeyError('Unsupported parameter encountered.')

    @property
    def latitude(self):
        """float: latitude orientation of CME [rad]."""
        return self._latitude
    @latitude.setter
    def latitude(self, latitude):
        self._latitude = subtract_period(latitude, np.pi*2)

    @property
    def longitude(self):
        """float: longitude orientation of CME [rad]."""
        return self._longitude
    @longitude.setter
    def longitude(self, longitude):
        self._longitude = subtract_period(longitude, np.pi*2)

    @property
    def toroidal_height(self):
        """float: distance from the origin (Sun) to the apex of the
        CME's axis [m].
        """
        return self._toroidal_height
    @toroidal_height.setter
    def toroidal_height(self, toroidal_height):
        if toroidal_height > 0:
            self._toroidal_height = toroidal_height
        else:
            raise ValueError('Toroidal height should be positive.')

    @property
    def poloidal_height(self):
        """float: distance from the apex of the CME's axis to its global
        apex [m].
        """
        return self._poloidal_height
    @poloidal_height.setter
    def poloidal_height(self, poloidal_height):
        if poloidal_height > 0:
            self._poloidal_height = poloidal_height
        else:
            raise ValueError('Poloidal height should be positive.')

    @property
    def half_width(self):
        """float: angular half width of the CME [rad]."""
        return self._half_width
    @half_width.setter
    def half_width(self, half_width):
        """Sets not only half width explicitly but also width
        coefficient implicitly.
        """
        if half_width > 0 and half_width < np.pi*2:
            self._half_width = half_width
            self._coeff_angle = np.pi/2/self.half_width
        else:
            raise ValueError('Half width should positive and less than 2pi.')

    @property
    def coeff_angle(self):
        """float: coefficient for the axis function,
        defined as pi/2/half_width [unitless].
        """
        return self._coeff_angle

    @property
    def tilt(self):
        """float: tilt of the CME, measured from equatorial plane using
        right-hand rule around the axis with origin in the Sun [rad].
        """
        return self._tilt
    @tilt.setter
    def tilt(self, tilt):
        self._tilt = subtract_period(tilt, np.pi*2)

    @property
    def flattening(self):
        """float: coefficient that controls the front flattening of the
        CME [unitless].
        0 corresponds to total flattening,
        1 corresponds to no flattening, i.e., circular axis.
        """
        return self._flattening
    @flattening.setter
    def flattening(self, flattening):
        if flattening > 0 and flattening < 1:
            self._flattening = flattening
        else:
            raise ValueError(
                'Flattening should be greater than 0 and less than 1.'
            )

    @property
    def pancaking(self):
        """float: angular half height of the CME, measured in the plane
        of the CME [rad].
        """
        return self._pancaking
    @pancaking.setter
    def pancaking(self, pancaking):
        if pancaking > 0 and pancaking < np.pi:
            self._pancaking = pancaking
        else:
            raise ValueError(
                'Pancaking should be greater than 0 and less than pi.'
            )

    @property
    def skew(self):
        """float: rotational skewing angle of the CME, happens due to
        rotation of the Sun [rad], corresponds to rotation angle of the
        Sun.
        """
        return self._skew
    @skew.setter
    def skew(self, skew):
        self._skew = skew

    @property
    def twist(self):
        """float: constant twist of the flux rope, measured as number of
        full rotations of magnetic fields around CME's axis [unitless].
        """
        return self._twist
    @twist.setter
    def twist(self, twist):
        """If negative twist is submitted the setter will revert the
        chirality.
        """
        if twist >= 0.0:
            self._twist = twist
        else:
            self._twist = -twist
            self._chirality *= -1

    @property
    def flux(self):
        """(float): total magnetic flux of the CME [Wb]."""
        return self._flux
    @flux.setter
    def flux(self, flux):
        """Set not only magnetic flux but also unit magnetic field if
        sigma is already defined.
        """
        if flux > 0:
            self._flux = flux
            if self.sigma is not None:
                self._unit_b = self.flux/(2.0*np.pi*self.sigma**2)
        else:
            raise ValueError('Flux should be positive.')

    @property
    def sigma(self):
        """float: sigma coefficient of the Gaussian distribution of
        total magnetic field in cross-section of CME [unitless].
        """
        return self._sigma
    @sigma.setter
    def sigma(self, sigma):
        """Set not only sigma but also unit magnetic field if magnetic
        flux is already defined.
        """
        if sigma > 0:
            self._sigma = sigma
            if self.flux is not None:
                self._unit_b = self.flux/(2.0*np.pi*self.sigma**2)
        else:
            raise ValueError('Sigma should be positive.')

    @property
    def polarity(self):
        """int: defines the polarity of the flux rope (direction of the
        axial magnetic field) [unitless].
        +1 corresponds to east-to-west direction of magnetic field from
        footpoint to footpoint.
        -1 corresponds to west-to-east direction of magnetic field from
        footpoint to footpoint.
        """
        return self._polarity
    @polarity.setter
    def polarity(self, polarity):
        if polarity == 1 or polarity == -1:
            self._polarity = polarity
        else:
            raise ValueError('Polarity should be +1 or -1.')

    @property
    def chirality(self):
        """int: defines the chirality (handedness) of the flux
        rope [unitless].
        +1 correponds to right-handed twist of magnetic field lines.
        -1 correponds to left-handed twist of magnetic field lines.
        """
        return self._chirality
    @chirality.setter
    def chirality(self, chirality):
        if chirality == 1 or chirality == -1:
            self._chirality = chirality
        else:
            raise ValueError('Chirality should be +1 or -1.')

    def vanilla_axis_height(
            self,
            phi,
            toroidal_height=None,
            coeff_angle=None,
            flattening=None):
        """Evaluate the axis function r(phi) in polar coordinates. Note
        that rotational skewing is not taken into account.

        Args:
            phi (float): angular coordinate of a point on the
                axis [rad] in polar coordinates, lies in the range
                [-half_width, half_width].
            toroidal_height (float, optional): custom toroidal height
                for the calculation [m].
            coeff_angle (float, optional): custom angle coefficient for
                the calculation [unitless].
            flattening (float, optional): custom angle coefficient for
                the calculation [unitless].

        Returns:
            (float) radial coordinate of the point of the axis in polar
                coordinates [m].
        """
        if toroidal_height is None:
            toroidal_height = self.toroidal_height
        if coeff_angle is None:
            coeff_angle = self.coeff_angle
        if flattening is None:
            flattening = self.flattening
        return toroidal_height*np.abs(np.cos(coeff_angle*phi))**flattening

    def vanilla_axis_dheight(
            self,
            phi,
            toroidal_height=None,
            coeff_angle=None,
            flattening=None):
        """Evaluate the derivative of the axis function dr/d(phi). Note
        that rotational skewing is not taken into account.

        Args:
            phi (float): angular coordinate of a point on the
                axis [rad] in polar coordinates, lies in the range
                [-half_width, half_width].
            toroidal_height (float, optional): custom toroidal height
                for the calculation [m].
            coeff_angle (float, optional): custom angle coefficient for
                the calculation [unitless].
            flattening (float, optional): custom angle coefficient for
                the calculation [unitless].

        Returns:
            (float) dr/d(phi) avulated at an angular point phi in polar
                coordinates [m/rad].
        """
        if toroidal_height is None:
            toroidal_height = self.toroidal_height
        if coeff_angle is None:
            coeff_angle = self.coeff_angle
        if flattening is None:
            flattening = self.flattening
        return(
            flattening*coeff_angle*np.abs(np.tan(coeff_angle*phi))
            *self.vanilla_axis_height(
                phi,
                toroidal_height=toroidal_height,
                coeff_angle=coeff_angle,
                flattening=flattening
            )
        )

    def vanilla_axis_distance(self, phi, r_sc, phi_sc):
        """Find the distance to the given point of the axis (phi) from
        an arbitrary point in space (r_sc, phi_sc).

        Args:
            phi (float): angular coordinate of a point on the
                axis [rad] in polar coordinates, lies in the range
                [-half_width, half_width].
            r_sc (float): radial coordinate of a point in space [m].
            phi_sc (float): radial coordinate of a point in space [rad].

        Returns:
            (float) distance from (r_sc, phi_sc) to the phi point of the
                axis [m].
        """
        return(
            (
                self.vanilla_axis_height(phi)*np.cos(phi)
                -r_sc*np.cos(phi_sc)
            )**2+
            (
                self.vanilla_axis_height(phi)*np.sin(phi)
                -r_sc*np.sin(phi_sc)
            )**2
        )

    def vanilla_axis_min_distance(self, r_sc, phi_sc):
        """Estimate the minimal distance to the axis from an arbitrary
        point in space (r_sc, phi_sc).

        Args:
            r_sc (float): radial coordinate of a point in space [m].
            phi_sc (float): radial coordinate of a point in space [rad].

        Returns:
            (float) minimal distance from (r_sc, phi_sc) to the
            axis [m].
        """
        phi = minimize_scalar(
            lambda phi: self.vanilla_axis_distance(phi, r_sc, phi_sc),
            bounds=[-self.half_width, self.half_width],
            method='Bounded'
        ).x
        return(self.vanilla_axis_distance(phi, r_sc, phi_sc), phi)

    def vanilla_axis_tan(self, phi):
        """Evaluate tangent angle relative to the axis at a given
        location.

        Args:
            phi (float): angular coordinate of a point on the
                axis [rad] in polar coordinates, lies in the range
                [-half_width, half_width].

        Returns:
            (float) tangent angle to the axis at a given angular
                location [rad].
        """
        return np.arctan(
            (-1)*self.coeff_angle*self.flattening*np.tan(self.coeff_angle*phi)
        )

    def vanilla_axis_dlength(
            self,
            phi,
            toroidal_height=None,
            coeff_angle=None,
            flattening=None):
        """Evaluate derivative of the axis length ds/d(phi). Note that
        rotational skewing is not taken into account.

        Args:
            phi (float): angular coordinate of a point on the
                axis [rad] in polar coordinates, lies in the range
                [-half_width, half_width].
            toroidal_height (float, optional): custom toroidal height
                for the calculation [m].
            coeff_angle (float, optional): custom angle coefficient for
                the calculation [unitless].
            flattening (float, optional): custom angle coefficient for
                the calculation [unitless].

        Returns:
            (float) ds/d(phi) evaluated at a phi angular location of the
                axis [m/rad].
        """
        if toroidal_height is None:
            toroidal_height = self.toroidal_height
        if coeff_angle is None:
            coeff_angle = self.coeff_angle
        if flattening is None:
            flattening = self.flattening
        return(
            self.vanilla_axis_height(
                phi,
                toroidal_height=toroidal_height,
                coeff_angle=coeff_angle,
                flattening=flattening
            )*np.sqrt(
                1+coeff_angle**2*flattening**2
                *np.tan(coeff_angle*phi)**2
            )
        )

    def vanilla_axis_length(self, phi):
        """Evaluate length of the axis. It is an estimation and also
        does not take into account rotational skewing.

        Args:
            phi (float): angular coordinate of a point on the
                axis [rad] in polar coordinates, lies in the range
                [-half_width, half_width].

        Returns:
            (float) length of the axis from origin footpoint towards the
                location defined by angular coordinate phi.
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

    def vanilla_axis_phi(self, length):
        """Evaluate polar coordinate of the axis as a function of its
        length. It is an estimation and also does not take into account
        rotational skewing.

        Args:
            length (float): length of the section of the axis from
                origin footpoint towards some point of the axis [m].

        Returns:
            (float) angular coordinate of a point of the axis, distance
                to which is equal to length [rad].
        """
        return(
            self._interpolator_axis_phi(
                (
                    length/self.vanilla_axis_length(np.pi/2/self.coeff_angle),
                    self.coeff_angle,
                    self.flattening
                )
            )/self.coeff_angle
        )

    def _init_axis_interpolators(
            self,
            n_coeff_angle_phi=100,
            n_coeff_angle=100,
            n_flattening=100,
            n_relative_length=100,
            ratio=1-1e-5):
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
        flattening = [0, 1]
        relative_length = [0, 1]

        Args:
            n_coeff_angle_phi (int, optional): number [unitless] of
                (linear) samples of the coeff_angle_phi
                range [-pi/2, pi/2].
            n_coeff_angle (int, optional): number [unitless] of
                (linear) samples of the coeff_angle range [1, 18].
            n_flattening (int, optional): number [unitless] of
                (linear) samples of the flattening range [0, 1].
            n_relative_length (int, optional): number [unitless] of
                (linear) samples of the relative_length range [0, 1].
            ratio (float, optional): numerical integration is applied in
                the range [-ratio*pi/2, ratio*pi/2]. Outside of this
                range ds/d(phi) tends to infinity and hence an
                assumption that length = height is made. Ratio
                parameter [unitless] can lie in the range [0, 1], though
                it makes sense to keep it as close to 1 as possible.

        Returns:
            nothing
        """
        def integrate_length(
                coeff_angle_phi,
                coeff_angle,
                flattening,
                ratio=ratio):
            """Estimate length along axis by numerical integration."""
            if coeff_angle_phi <= -np.pi/2*ratio:
                length = self.vanilla_axis_height(
                    coeff_angle_phi/coeff_angle,
                    toroidal_height=1,
                    coeff_angle=coeff_angle,
                    flattening=flattening
                )
            elif coeff_angle_phi < np.pi/2*ratio:
                length = (
                    self.vanilla_axis_height(
                        -np.pi/2*ratio/coeff_angle,
                        toroidal_height=1,
                        coeff_angle=coeff_angle,
                        flattening=flattening
                    )+fixed_quad(
                        lambda coeff_angle_phi: self.vanilla_axis_dlength(
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
                    2*self.vanilla_axis_height(
                        -np.pi/2*ratio/coeff_angle,
                        toroidal_height=1,
                        coeff_angle=coeff_angle,
                        flattening=flattening
                    )+fixed_quad(
                        lambda coeff_angle_phi: self.vanilla_axis_dlength(
                            coeff_angle_phi/coeff_angle,
                            toroidal_height=1,
                            coeff_angle=coeff_angle,
                            flattening=flattening)/coeff_angle,
                        -np.pi/2*ratio,
                        np.pi/2*ratio,
                        n=1000
                    )[0]-self.vanilla_axis_height(
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
            n_coeff_angle_phi
        )
        coeff_angle_array = np.linspace(1, 18, n_coeff_angle)
        flattening_array = np.linspace(0.1, 1, n_flattening)

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
        relative_length_array = np.linspace(0, 1, n_relative_length)
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
            s=np.linspace(0, 1, 50),
            phi=np.linspace(0, np.pi*2, 24)):
        """Evaluate the 3D shell of the flux rope.

        Args:
            s (numpy.ndarray, optional): defines the sampling along the
                axis in a relative sense, i.e., s goes from 0 to 1 from
                one footpoint to the other [unitless].
            phi (numpy.ndarray, optional) defines the angular sampling
                of the cross-section [rad].

        Returns:
            (numpy.ndarray, numpy.ndarray, numpy.ndarray)
                (x, y, z) coordinates of the shell points [m].
        """
        s = np.array(s, copy=False, ndmin=1)
        phi = np.array(phi, copy=False, ndmin=1)
        s = np.transpose(np.tile(s, (phi.size, 1)))
        phi = np.tile(phi, (s.shape[0], 1))
        # extend to full axis length
        z = s*self.vanilla_axis_length(self.half_width)
        # apply tapering
        r = np.ones(s.shape)
        r = (
            r*self.poloidal_height
            *(
                self.vanilla_axis_height(self.vanilla_axis_phi(z))
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
        phi = self.vanilla_axis_phi(x)
        r = self.vanilla_axis_height(phi)
        t = self.vanilla_axis_tan(phi)
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

    def line(self, r=0, phi=0, s=np.linspace(0, 1, 50)):
        """Evaluate the 3D magnetic field line of the flux rope.

        Args:
            r (float, optional): relative radial coordinate of the line
                origin in origin footpoint cross-section [m], goes from
                0 (center) to 1 (edge).
            phi (float, optional): angular coordinate of the line origin
                in origin footpoint cross-section [rad].
            s (numpy.ndarray, optional): relative sampling [unitless] of
                the line, assuming that distance along the line goes
                from 0 to 1.

        Returns:
            (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray)
                (x, y, z, b) coordinates of the line point and total
                magnetic field along the line.
        """
        s = np.array(s, copy=False, ndmin=1)
        phi = np.ones(s.size)*phi
        # twist
        phi += s*self.twist*np.pi*2.0*self.chirality
        # elongation
        z = s*self.vanilla_axis_length(self.half_width)
        # distance to axis from origin
        R = self.vanilla_axis_height(self.vanilla_axis_phi(z))
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
            -((r/rx)**2)/2/self.sigma**2
        )
        x, y, z = cs.cyl2cart(r, phi, z)
        # rotation to x
        T = cs.mx_rot_y(np.pi/2)
        x, y, z = cs.mx_apply(T, x, y, z)
        # bending
        x[x < 0] = 0
        phi = self.vanilla_axis_phi(x)
        r = self.vanilla_axis_height(phi)
        t = self.vanilla_axis_tan(phi)
        x = r*np.cos(phi)+np.sin(t-phi-np.pi/2)*y
        y = r*np.sin(phi)+np.cos(t-phi-np.pi/2)*y
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
        phi += self.skew*(1-r/self.toroidal_height)
        x, y, z = cs.cyl2cart(r, phi, z)
        return (x, y, z, b)

    def data(self, x, y, z, ds=1e-5):
        """Evaluate magnetic field measurements at a given point (or
        trajectory) in space.

        Args:
            x (float or numpy.ndarray): x coordinate(s) in space.
            y (float or numpy.ndarray): y coordinate(s) in space.
            z (float or numpy.ndarray): z coordinate(s) in space.
            ds (float, optional): length of a relaticve axis section
                used to integrate the magnetic field measurement.

        Returns:
            (numpy.ndarray, numpy.ndarray)
                magnetic field measurements (3, n) and
                coefficients used for local speed estimation (2, n)
        """
        x = np.array(x, copy=False, ndmin=1)
        y = np.array(y, copy=False, ndmin=1)
        z = np.array(z, copy=False, ndmin=1)
        # reverse skew
        r, theta, phi = cs.cart2sp(x, y, z)
        phi -= self.skew*(1-r/self.toroidal_height)
        x, y, z = cs.sp2cart(r, theta, phi)
        # reverse orientation
        T = cs.mx_rot_reverse(self.latitude, -self.longitude, -self.tilt)
        x, y, z = cs.mx_apply(T, x, y, z)
        # reverse pancaking
        r, theta, phi = cs.cart2sp(x, y, z)
        theta = (
            theta/self.pancaking*
            np.arctan2(self.poloidal_height, self.toroidal_height)
        )
        x, y, z = cs.sp2cart(r, theta, phi)
        # inside axis loop mask
        p_in = self.vanilla_axis_height(phi) >= r*np.cos(theta)
        # get r_ax and phi_ax of the closest point on axis
        v_vanilla_axis_min_distance = np.vectorize(
            self.vanilla_axis_min_distance,
            otypes=[np.float64, np.float64]
        )
        _, phi_ax = v_vanilla_axis_min_distance(r*np.cos(theta), phi)
        r_ax = self.vanilla_axis_height(phi_ax)
        # get s
        v_vanilla_axis_length = np.vectorize(
            self.vanilla_axis_length,
            otypes=[np.float64]
        )
        s = (
            v_vanilla_axis_length(phi_ax)
            /self.vanilla_axis_length(self.half_width)
        )
        x_ax, y_ax, z_ax = cs.sp2cart(r_ax, np.zeros(r_ax.size), phi_ax)
        dx = x-x_ax
        dy = y-y_ax
        dz = z-z_ax
        r_abs = np.sqrt(dx**2+dy**2+dz**2)
        r = r_abs/(r_ax*self.poloidal_height/self.toroidal_height)
        def div0(a, b):
            """Handling the division by zero by defaulting to zero.""" 
            with np.errstate(divide='ignore', invalid='ignore'):
                cc = np.true_divide(a, b)
                cc[~np.isfinite(cc)] = 0  # -inf inf NaN
            return cc
        phi = (
            np.piecewise(dz, [dz < 0, dz >= 0], [-1, 1])
            *np.arccos(div0(np.sqrt(dx**2+dy**2), r_abs))
        )
        phi[p_in] = np.pi-phi[p_in]
        # reverse twist
        phi -= s*self.twist*np.pi*2*self.chirality
        # reverse rotation to x
        phi -= np.pi/2
        # get magnetic field and speed coefficients along sc trajectory
        b = []
        vc = []
        for i in range(r.size):
            if r[i] <= 1:
                x_, y_, z_, b_ = self.line(
                    r[i],
                    phi[i],
                    [s[i]-ds, s[i]+ds]
                )
                if x_.size < 2 or y_.size < 2 or z_.size < 2:
                    b.append([np.nan, np.nan, np.nan])
                    vc.append([np.nan, np.nan])
                else:
                    vtc = r_ax[i]/self.toroidal_height
                    vpc = (
                        r_ax[i]/self.toroidal_height
                        *(
                            np.sqrt(
                                np.mean(x_)**2
                                +np.mean(y_)**2
                                +np.mean(z_)**2
                            )
                            -r_ax[i]
                        )
                        /self.poloidal_height
                        *np.cos(self.vanilla_axis_tan(phi_ax[i]))
                    )
                    dr = np.array([
                        x_[1]-x_[0],
                        y_[1]-y_[0],
                        z_[1]-z_[0]
                    ])
                    dr /= np.linalg.norm(dr)
                    b.append(dr*np.mean(b_)*self.polarity)
                    vc.append(np.array([vtc, vpc]))
            else:
                b.append([np.nan, np.nan, np.nan])
                vc.append([np.nan, np.nan])
        return (np.array(b), np.array(vc))

    def impact(self, x, y, z, dphi=1e-5):
        """Estimate the impact distance.

        Args:
            Args:
            x (float): x coordinate in space.
            y (float): y coordinate in space.
            z (float): z coordinate in space.

        Returns:
            (float): impact distance
            (np.ndarray): (x, y, z)-coordinates of the closest point on
                the axis impact  don't remember
            (np.ndarray): (x, y, z)-vector (unit) of the tangent to the
                the axis near the closest point.
        """
        x0 = x
        y0 = y
        z0 = z
        # reverse skew
        r, theta, phi = cs.cart2sp(x, y, z)
        phi -= self.skew*(1-r/self.toroidal_height)
        x, y, z = cs.sp2cart(r, theta, phi)
        # reverse orientation
        T = cs.mx_rot_reverse(self.latitude, -self.longitude, -self.tilt)
        x, y, z = cs.mx_apply(T, x, y, z)
        # reverse pancaking
        r, theta, phi = cs.cart2sp(x, y, z)
        theta = (
            theta/self.pancaking*
            np.arctan2(self.poloidal_height, self.toroidal_height)
        )
        # get r_ax and phi_ax of the closest point on axis
        phi_ax = self.vanilla_axis_min_distance(r*np.cos(theta), phi)
        r_ax = self.vanilla_axis_height(phi_ax)
        x, y, z = cs.sp2cart(r_ax, np.zeros(r_ax.size), phi_ax)
        # orientation
        T = cs.mx_rot(-self.latitude, self.longitude, self.tilt)
        x, y, z = cs.mx_apply(T, x, y, z)
        # skew
        r, theta, phi = cs.cart2sp(x, y, z)
        phi += self.skew*(1-r/self.toroidal_height)
        x, y, z = cs.sp2cart(r, theta, phi)
        # get r_ax and phi_ax of the closest delta points on axis
        phi_ax1 = phi_ax-dphi
        phi_ax2 = phi_ax+dphi
        r_ax1 = self.vanilla_axis_height(phi_ax1)
        r_ax2 = self.vanilla_axis_height(phi_ax2)
        x1, y1, z1 = cs.sp2cart(r_ax1, np.zeros(r_ax1.size), phi_ax1)
        x2, y2, z2 = cs.sp2cart(r_ax2, np.zeros(r_ax2.size), phi_ax2)
        # orientation
        T = cs.mx_rot(-self.latitude, self.longitude, self.tilt)
        x1, y1, z1 = cs.mx_apply(T, x1, y1, z1)
        x2, y2, z2 = cs.mx_apply(T, x2, y2, z2)
        # skew
        r, theta, phi = cs.cart2sp(x1, y1, z1)
        phi += self.skew*(1-r/self.toroidal_height)
        x1, y1, z1 = cs.sp2cart(r, theta, phi)
        r, theta, phi = cs.cart2sp(x2, y2, z2)
        phi += self.skew*(1-r/self.toroidal_height)
        x2, y2, z2 = cs.sp2cart(r, theta, phi)
        d = np.array([x2, y2, z1])-np.array([x1, y1, z1])
        d /= np.linalg.norm(d)
        return(
            np.linalg.norm(np.array([x-x0, y-y0, z-z0])),
            np.array([x, y, z]),
            d
        )
