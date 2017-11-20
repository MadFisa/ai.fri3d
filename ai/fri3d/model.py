"""The module defines the static (StaticFRi3D) and dynamic
(DynamicFRi3D) classes that describe FRi3D as a static snapshot
and as a propagating dynamic structure, respectively.
"""
# pylint: disable=too-many-instance-attributes
# pylint: disable=E1101
# pylint: disable=E1102
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

class BaseFRi3D:
    """Parent class for all FRi3D-related classes. Defines the common
    model properties.
    """
    def __init__(self):
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
        self._polarity = None
        self._chirality = None
        self._props = [
            p for p in dir(BaseFRi3D)
            if isinstance(getattr(BaseFRi3D, p), property)
        ]

    @property
    def latitude(self):
        """float, profile function or profile object:
        latitude orientation of CME [rad].
        """
        return self._latitude
    @latitude.setter
    def latitude(self, val):
        self._latitude = val

    @property
    def longitude(self):
        """float, profile function or profile object:
        longitude orientation of CME [rad].
        """
        return self._longitude
    @longitude.setter
    def longitude(self, val):
        self._longitude = val

    @property
    def toroidal_height(self):
        """float, profile function or profile object:
        distance from the origin (Sun) to the apex of
        the CME's axis [m].
        """
        return self._toroidal_height
    @toroidal_height.setter
    def toroidal_height(self, val):
        self._toroidal_height = val

    @property
    def poloidal_height(self):
        """float, profile function or profile object:
        distance from the apex of the CME's axis to its global apex [m].
        """
        return self._poloidal_height
    @poloidal_height.setter
    def poloidal_height(self, val):
        self._poloidal_height = val

    @property
    def half_width(self):
        """float, profile function or profile object:
        angular half width of the CME [rad].
        """
        return self._half_width
    @half_width.setter
    def half_width(self, val):
        self._half_width = val

    @property
    def tilt(self):
        """float, profile function or profile object:
        tilt of the CME, measured from equatorial plane using right-hand
        rule around the axis with origin in the Sun [rad].
        """
        return self._tilt
    @tilt.setter
    def tilt(self, val):
        self._tilt = val

    @property
    def flattening(self):
        """float, profile function or profile object:
        coefficient that controls the front flattening
        of the CME [unitless].
        0 corresponds to total flattening,
        1 corresponds to no flattening, i.e., circular axis.
        """
        return self._flattening
    @flattening.setter
    def flattening(self, val):
        self._flattening = val

    @property
    def pancaking(self):
        """float, profile function or profile object:
        angular half height of the CME, measured in the plane
        of the CME [rad].
        """
        return self._pancaking
    @pancaking.setter
    def pancaking(self, val):
        self._pancaking = val

    @property
    def skew(self):
        """float, profile function or profile object:
        rotational skewing angle of the CME, happens due to rotation
        of the Sun [rad], corresponds to rotation angle of the Sun.
        """
        return self._skew
    @skew.setter
    def skew(self, val):
        self._skew = val

    @property
    def twist(self):
        """float, profile function or profile object:
        constant twist of the flux rope, measured as number of full
        rotations of magnetic fields around CME's axis [unitless].
        """
        return self._twist
    @twist.setter
    def twist(self, val):
        self._twist = val

    @property
    def flux(self):
        """float, profile function or profile object:
        total magnetic flux of the CME [Wb].
        """
        return self._flux
    @flux.setter
    def flux(self, val):
        self._flux = val

    @property
    def sigma(self):
        """float, profile function or profile object:
        sigma coefficient of the Gaussian distribution of total magnetic
        field in cross-section of CME [unitless].
        """
        return self._sigma
    @sigma.setter
    def sigma(self, val):
        self._sigma = val

    @property
    def polarity(self):
        """float, profile function or profile object:
        defines the polarity of the flux rope (direction of the axial
        magnetic field) [unitless].
        +1 corresponds to east-to-west direction of magnetic field from
        footpoint to footpoint.
        -1 corresponds to west-to-east direction of magnetic field from
        footpoint to footpoint.
        """
        return self._polarity
    @polarity.setter
    def polarity(self, val):
        self._polarity = val

    @property
    def chirality(self):
        """float, profile function or profile object:
        defines the chirality (handedness) of the flux rope [unitless].
        +1 correponds to right-handed twist of magnetic field lines.
        -1 correponds to left-handed twist of magnetic field lines.
        """
        return self._chirality
    @chirality.setter
    def chirality(self, val):
        self._chirality = val

class StaticFRi3D(BaseFRi3D):
    """FRi3D model static class. It provides static description of the
    model.
    """
    def __init__(self, **kwargs):
        super(StaticFRi3D, self).__init__()
        self._coeff_angle = None
        self._unit_b = None
        self._interpolator_axis_length = None
        self._interpolator_axis_phi = None
        self.latitude = kwargs.get('latitude', 0)
        self.longitude = kwargs.get('longitude', 0)
        self.toroidal_height = kwargs.get('toroidal_height', u.au.to(u.m, 1))
        self.poloidal_height = kwargs.get('poloidal_height', u.au.to(u.m, 0.2))
        self.half_width = kwargs.get('half_width', u.deg.to(u.rad, 40))
        self.tilt = kwargs.get('tilt', 0)
        self.flattening = kwargs.get('flattening', 0.5)
        self.pancaking = kwargs.get(
            'pancaking',
            np.arctan(self.poloidal_height/self.toroidal_height)
        )
        self.skew = kwargs.get('skew', 0)
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
        for k, v in kwargs.items():
            if k in self._props:
                setattr(self, k, v)
            else:
                raise KeyError('Unsupported parameter encountered.')

    @BaseFRi3D.latitude.setter
    def latitude(self, val):
        self._latitude = subtract_period(val, np.pi*2)

    @BaseFRi3D.longitude.setter
    def longitude(self, val):
        self._longitude = subtract_period(val, np.pi*2)

    @BaseFRi3D.toroidal_height.setter
    def toroidal_height(self, val):
        if val > 0:
            self._toroidal_height = val
        else:
            raise ValueError('Toroidal height should be positive.')

    @BaseFRi3D.poloidal_height.setter
    def poloidal_height(self, val):
        if val > 0:
            self._poloidal_height = val
        else:
            raise ValueError('Poloidal height should be positive.')

    @BaseFRi3D.half_width.setter
    def half_width(self, val):
        """Sets not only half width explicitly but also width
        coefficient implicitly.
        """
        if val > 0 and val < np.pi*2:
            self._half_width = val
            self._coeff_angle = np.pi/2/self.half_width
        else:
            raise ValueError('Half width should positive and less than 2pi.')

    @BaseFRi3D.tilt.setter
    def tilt(self, val):
        self._tilt = subtract_period(val, np.pi*2)

    @BaseFRi3D.flattening.setter
    def flattening(self, val):
        if val > 0 and val < 1:
            self._flattening = val
        else:
            raise ValueError(
                'Flattening should be greater than 0 and less than 1.'
            )

    @BaseFRi3D.pancaking.setter
    def pancaking(self, val):
        if val > 0 and val < np.pi:
            self._pancaking = val
        else:
            raise ValueError(
                'Pancaking should be greater than 0 and less than pi.'
            )

    @BaseFRi3D.twist.setter
    def twist(self, val):
        """If negative twist is submitted the setter will revert the
        chirality.
        """
        if val >= 0.0:
            self._twist = val
        else:
            self._twist = -val
            self._chirality *= -1

    @BaseFRi3D.flux.setter
    def flux(self, val):
        """Set not only magnetic flux but also unit magnetic field if
        sigma is already defined.
        """
        if val > 0:
            self._flux = val
            if self.sigma is not None:
                self._unit_b = self.flux/(2.0*np.pi*self.sigma**2)
        else:
            raise ValueError('Flux should be positive.')

    @BaseFRi3D.sigma.setter
    def sigma(self, val):
        """Set not only sigma but also unit magnetic field if magnetic
        flux is already defined.
        """
        if val > 0:
            self._sigma = val
            if self.flux is not None:
                self._unit_b = self.flux/(2.0*np.pi*self.sigma**2)
        else:
            raise ValueError('Sigma should be positive.')

    @BaseFRi3D.polarity.setter
    def polarity(self, val):
        if val == 1 or val == -1:
            self._polarity = val
        else:
            raise ValueError('Polarity should be +1 or -1.')

    @BaseFRi3D.chirality.setter
    def chirality(self, val):
        if val == 1 or val == -1:
            self._chirality = val
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
            coeff_angle = self._coeff_angle
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
            coeff_angle = self._coeff_angle
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
            (-1)*self._coeff_angle*self.flattening
            *np.tan(self._coeff_angle*phi)
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
            coeff_angle = self._coeff_angle
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
                    self._coeff_angle*phi,
                    self._coeff_angle,
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
                    length/self.vanilla_axis_length(np.pi/2/self._coeff_angle),
                    self._coeff_angle,
                    self.flattening
                )
            )/self._coeff_angle
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

class DynamicFRi3D(BaseFRi3D):
    """FRi3D model dynamic class. It provides static description of the
    model.
    """
    def __init__(self, **kwargs):
        super(DynamicFRi3D, self).__init__()
        self.__sfr = StaticFRi3D()
        self.latitude = kwargs.get('latitude', lambda t: self.__sfr.latitude)
        self.longitude = kwargs.get(
            'longitude',
            lambda t: self.__sfr.longitude
        )
        self.toroidal_height = kwargs.get(
            'toroidal_height',
            lambda t: self.__sfr.toroidal_height
        )
        self.poloidal_height = kwargs.get(
            'poloidal_height',
            lambda t: self.__sfr.poloidal_height
        )
        self.half_width = kwargs.get(
            'half_width',
            lambda t: self.__sfr.half_width
        )
        self.tilt = kwargs.get('tilt', lambda t: self.__sfr.tilt)
        self.flattening = kwargs.get(
            'flattening',
            lambda t: self.__sfr.flattening
        )
        self.pancaking = kwargs.get(
            'pancaking',
            lambda t: self.__sfr.pancaking
        )
        self.skew = kwargs.get('skew', lambda t: self.__sfr.skew)
        self.twist = kwargs.get('twist', lambda t: self.__sfr.twist)
        self.flux = kwargs.get('flux', lambda t: self.__sfr.flux)
        self.sigma = kwargs.get('sigma', lambda t: self.__sfr.sigma)
        self.polarity = kwargs.get('polarity', lambda t: self.__sfr.polarity)
        self.chirality = kwargs.get(
            'chirality',
            lambda t: self.__sfr.chirality
        )

    def modify(self, **kwargs):
        """Modify the time profiles."""
        for k, v in kwargs.items():
            if k in self._props:
                setattr(self, k, v)
            else:
                raise KeyError('Unsupported parameter encountered.')

    @BaseFRi3D.latitude.setter
    def latitude(self, func):
        if callable(func):
            self._latitude = func
        else:
            raise ValueError('Latitude profile is expected to be a callable.')

    @BaseFRi3D.longitude.setter
    def longitude(self, func):
        if callable(func):
            self._longitude = func
        else:
            raise ValueError('Longitude profile is expected to be a callable.')

    @BaseFRi3D.toroidal_height.setter
    def toroidal_height(self, func):
        if callable(func):
            self._toroidal_height = func
        else:
            raise ValueError(
                'Toroidal height profile is expected to be a callable.'
            )

    @BaseFRi3D.poloidal_height.setter
    def poloidal_height(self, func):
        if callable(func):
            self._poloidal_height = func
        else:
            raise ValueError(
                'Poloidal height profile is expected to be a callable.'
            )

    @BaseFRi3D.half_width.setter
    def half_width(self, func):
        if callable(func):
            self._half_width = func
        else:
            raise ValueError(
                'Half width profile is expected to be a callable.'
            )

    @BaseFRi3D.tilt.setter
    def tilt(self, func):
        if callable(func):
            self._tilt = func
        else:
            raise ValueError(
                'Tilt profile is expected to be a callable.'
            )

    @BaseFRi3D.flattening.setter
    def flattening(self, func):
        if callable(func):
            self._flattening = func
        else:
            raise ValueError(
                'Flattening profile is expected to be a callable.'
            )

    @BaseFRi3D.pancaking.setter
    def pancaking(self, func):
        if callable(func):
            self._pancaking = func
        else:
            raise ValueError(
                'Pancaking profile is expected to be a callable.'
            )

    @BaseFRi3D.skew.setter
    def skew(self, func):
        if callable(func):
            self._skew = func
        else:
            raise ValueError(
                'Skew profile is expected to be a callable.'
            )

    @BaseFRi3D.twist.setter
    def twist(self, func):
        if callable(func):
            self._twist = func
        else:
            raise ValueError(
                'Twist profile is expected to be a callable.'
            )

    @BaseFRi3D.flux.setter
    def flux(self, func):
        if callable(func):
            self._flux = func
        else:
            raise ValueError(
                'Flux profile is expected to be a callable.'
            )

    @BaseFRi3D.sigma.setter
    def sigma(self, func):
        if callable(func):
            self._sigma = func
        else:
            raise ValueError(
                'Sigma profile is expected to be a callable.'
            )

    @BaseFRi3D.polarity.setter
    def polarity(self, func):
        if callable(func):
            self._polarity = func
        else:
            raise ValueError(
                'Polarity profile is expected to be a callable.'
            )

    @BaseFRi3D.chirality.setter
    def chirality(self, func):
        if callable(func):
            self._chirality = func
        else:
            raise ValueError(
                'Chirality profile is expected to be a callable.'
            )

    def snapshot(self, t):
        """Returns a snapshot of the FRi3D model at a given moment of
        time.

        Args:
            t (float): timestamp

        Returns:
            StaticFRi3D object
        """
        self.__sfr.modify(
            latitude=self.latitude(t),
            longitude=self.longitude(t),
            toroidal_height=self.toroidal_height(t),
            poloidal_height=self.poloidal_height(t),
            half_width=self.half_width(t),
            tilt=self.tilt(t),
            flattening=self.flattening(t),
            pancaking=self.pancaking(t),
            skew=self.skew(t),
            twist=self.twist(t),
            flux=self.flux(t),
            sigma=self.sigma(t),
            polarity=self.polarity(t),
            chirality=self.chirality(t)
        )
        return self.__sfr

    def insitu(self, t, x, y, z):
        """Calculate synthetic in-situ measurements.

        Args:
            t (float or np.ndarray): time (unix timestamp) for which
                the in-situ measurements are estimated. Can be a single
                time or an array of timestamps.
            x, y, z (float or func): synthetic spacecraft coordinates.
                Can be a single point in space or func(t) which describe
                the spacecraft trajectory.

        Returns:
            (np.ndarray(3), np.ndarray): magnetic field components and
                absolute speed.
        """
        t = np.array(t, copy=False, ndmin=1)
        if not callable(x):
            _x = x
            x = lambda t: _x
        if not callable(y):
            _y = y
            y = lambda t: _y
        if not callable(z):
            _z = z
            z = lambda t: _z
        b = []
        v = []
        for _t in t:
            _b, _c = self.snapshot(_t).data(x(_t), y(_t), z(_t))
            _b = _b[0, :]
            _c = _c[0, :]
            b.append(_b.ravel())
            v.append(
                _c[0]*(self.toroidal_height(_t)-self.toroidal_height(_t-1))+
                _c[1]*(self.poloidal_height(_t)-self.poloidal_height(_t-1))
            )
        return (np.array(b), np.array(v))

    # def impact(self, t, x, y, z):
    #     fr = FRi3D()
    #     fr.polarity = self.polarity
    #     fr.chirality = self.chirality
    #     fr.spline_s_phi_kind = self.spline_s_phi_kind
    #     fr.spline_s_phi_n = self.spline_s_phi_n
    #     impacts = []
    #     times = []
    #     for i, t in enumerate(t):
    #         fr.latitude = self.latitude(t)
    #         fr.longitude = self.longitude(t)
    #         fr.toroidal_height = self.toroidal_height(t)
    #         fr.poloidal_height = self.poloidal_height(t)
    #         fr.half_width = self.half_width(t)
    #         fr.tilt = self.tilt(t)
    #         fr.flattening = self.flattening(t)
    #         fr.pancaking = self.pancaking(t)
    #         fr.skew = self.skew(t)
    #         fr.twist = self.twist(t)
    #         fr.flux = self.flux(t)
    #         fr.sigma = self.sigma(t)
    #         if i == 0:
    #             fr.toroidal_height = 1.0
    #             fr.init()
    #             fr._unit_spline_initial_axis_s_phi = \
    #                 fr._spline_initial_axis_s_phi
    #             fr.toroidal_height = self.toroidal_height(t)
    #             fr.init()
    #         fr._spline_initial_axis_s_phi = lambda s: \
    #             fr._unit_spline_initial_axis_s_phi(s/fr.toroidal_height)
    #         impact, _, _, _, _, _, _ = fr.impact(x, y, z)
    #         impacts.append(impact)
    #         times.append(t)
    #     impacts = np.array(impacts)
    #     times = np.array(times)
    #     index = np.argmin(impacts)
    #     return (impacts[index], times[index])

    # def map(self, t, x, y, z,
    #     dx=u.au.to(u.m, np.linspace(-0.2, 0.2, 100)),
    #     dy=u.au.to(u.m, np.linspace(-0.2, 0.2, 100))):

    #     _, t = self.impact(t, x, y, z)

    #     fr = FRi3D()
    #     fr.polarity = self.polarity
    #     fr.chirality = self.chirality
    #     fr.spline_s_phi_kind = self.spline_s_phi_kind
    #     fr.spline_s_phi_n = self.spline_s_phi_n
    #     fr.latitude = self.latitude(t)
    #     fr.longitude = self.longitude(t)
    #     fr.toroidal_height = self.toroidal_height(t)
    #     fr.poloidal_height = self.poloidal_height(t)
    #     fr.half_width = self.half_width(t)
    #     fr.tilt = self.tilt(t)
    #     fr.flattening = self.flattening(t)
    #     fr.pancaking = self.pancaking(t)
    #     fr.skew = self.skew(t)
    #     fr.twist = self.twist(t)
    #     fr.flux = self.flux(t)
    #     fr.sigma = self.sigma(t)
    #     fr.toroidal_height = 1.0
    #     fr.init()
    #     fr._unit_spline_initial_axis_s_phi = \
    #         fr._spline_initial_axis_s_phi
    #     fr.toroidal_height = self.toroidal_height(t)
    #     fr.init()

    #     _, xa, ya, za, xt, yt, zt = fr.impact(x, y, z)
    #     vtan = np.array([np.mean(xt), np.mean(yt), np.mean(zt)])
    #     if np.dot(vtan, fr.data(xa, ya, za)) < 0.0:
    #         vtan = -vtan
    #     # vtan = fr.data(xa, ya, za)
    #     # vtan /= np.linalg.norm(vtan)
    #     vsc = np.array([x, y, z])
    #     vsc /= np.linalg.norm(vsc)

    #     vmcy = np.cross(vtan, vsc)
    #     vmcy /= np.linalg.norm(vmcy)
    #     if vmcy[0] < 0.0:
    #         vmcy = -vmcy
    #     vmcx = np.cross(vmcy, vtan)
    #     vmcx /= np.linalg.norm(vmcx)

    #     print(vmcx, vmcy, vtan)

    #     xg = np.zeros([dx.size, dy.size])
    #     yg = np.zeros([dx.size, dy.size])
    #     zg = np.zeros([dx.size, dy.size])

    #     for i in range(dx.size):
    #         for k in range(dy.size):
    #             p = np.array([x, y, z])+dx[i]*vmcx+dy[k]*vmcy
    #             xg[i,k] = p[0]
    #             yg[i,k] = p[1]
    #             zg[i,k] = p[2]
    #     print(
    #         xg.shape, xg.flatten().shape,
    #         yg.shape, yg.flatten().shape,
    #         zg.shape, zg.flatten().shape
    #     )
    #     b = fr.data(xg.flatten(), yg.flatten(), zg.flatten())
    #     print(b.shape)
    #     bmap = np.zeros(b.shape[0])
    #     for i in range(b.shape[0]):
    #         bmap[i] = np.dot(b[i,:], vtan)
    #     bmap = np.reshape(bmap, [dx.size, dy.size])

    #     return bmap.T
