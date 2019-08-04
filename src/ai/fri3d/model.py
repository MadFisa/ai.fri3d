"""The module defines the static (StaticFRi3D) and dynamic
(DynamicFRi3D) classes that describe FRi3D as a static snapshot
and as a propagating dynamic structure, respectively.
"""
# pylint: disable=too-many-instance-attributes
# pylint: disable=E1101
# pylint: disable=E1102
# pylint: disable=C0103
# pylint: disable=C0302
import math

import numpy as np
from ai import cs
from scipy import LowLevelCallable
from scipy.integrate import dblquad, quad
from scipy.optimize import minimize_scalar

from ai.fri3d import lib


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
        self._half_height = None
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
        self._props = [p for p in dir(BaseFRi3D) if isinstance(getattr(BaseFRi3D, p), property)]

    @property
    def latitude(self):
        """scalar, profile function or profile object:
        latitude orientation of CME [rad].
        """
        return self._latitude

    @latitude.setter
    def latitude(self, val):
        self._latitude = val

    @property
    def longitude(self):
        """scalar, profile function or profile object:
        longitude orientation of CME [rad].
        """
        return self._longitude

    @longitude.setter
    def longitude(self, val):
        self._longitude = val

    @property
    def toroidal_height(self):
        """scalar, profile function or profile object:
        distance from the origin (Sun) to the apex of
        the CME's axis [m].
        """
        return self._toroidal_height

    @toroidal_height.setter
    def toroidal_height(self, val):
        self._toroidal_height = val

    @property
    def half_width(self):
        """scalar, profile function or profile object:
        angular half width of the CME [rad].
        """
        return self._half_width

    @half_width.setter
    def half_width(self, val):
        self._half_width = val

    @property
    def half_height(self):
        """scalar, profile function or profile object:
        angular half height of the CME (pancaking) [rad].
        """
        return self._half_height

    @half_height.setter
    def half_height(self, val):
        self._half_height = val

    @property
    def tilt(self):
        """scalar, profile function or profile object:
        tilt of the CME, measured from equatorial plane using right-hand
        rule around the axis with origin in the Sun [rad].
        """
        return self._tilt

    @tilt.setter
    def tilt(self, val):
        self._tilt = val

    @property
    def flattening(self):
        """scalar, profile function or profile object:
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
        """scalar, profile function or profile object:
        angular half height of the CME, measured in the plane
        of the CME [rad].
        """
        return self._pancaking

    @pancaking.setter
    def pancaking(self, val):
        self._pancaking = val

    @property
    def skew(self):
        """scalar, profile function or profile object:
        rotational skewing angle of the CME, happens due to rotation
        of the Sun [rad], corresponds to rotation angle of the Sun.
        """
        return self._skew

    @skew.setter
    def skew(self, val):
        self._skew = val

    @property
    def twist(self):
        """scalar, profile function or profile object:
        constant twist of the flux rope, measured as number of full
        rotations of magnetic fields around CME's axis [unitless].
        """
        return self._twist

    @twist.setter
    def twist(self, val):
        self._twist = val

    @property
    def flux(self):
        """scalar, profile function or profile object:
        total magnetic flux of the CME [Wb].
        """
        return self._flux

    @flux.setter
    def flux(self, val):
        self._flux = val

    @property
    def sigma(self):
        """scalar, profile function or profile object:
        sigma coefficient of the Gaussian distribution of total magnetic
        field in cross-section of CME [unitless].
        """
        return self._sigma

    @sigma.setter
    def sigma(self, val):
        self._sigma = val

    @property
    def polarity(self):
        """scalar, profile function or profile object:
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
        """scalar, profile function or profile object:
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
        self.latitude = kwargs.get("latitude", 0)
        self.longitude = kwargs.get("longitude", 0)
        self.toroidal_height = kwargs.get("toroidal_height", 149597870700)
        self.half_width = kwargs.get("half_width", 40 * np.pi / 180)
        self.half_height = kwargs.get("half_height", 30 * np.pi / 180)
        self.tilt = kwargs.get("tilt", 0)
        self.flattening = kwargs.get("flattening", 0.5)
        self.pancaking = kwargs.get("pancaking", 1)
        self.skew = kwargs.get("skew", 0)
        self.twist = kwargs.get("twist", 1)
        self.flux = kwargs.get("flux", 1e13)
        self.sigma = kwargs.get("sigma", 2)
        self.polarity = kwargs.get("polarity", 1)
        self.chirality = kwargs.get("chirality", 1)

    def modify(self, **kwargs):
        """Modifies the model parameters.

        Args:
            **kwargs: Scalar model parameters. Allowed keywords:
                latitude, longitude, toroidal_height, half_width,
                half_height, tilt, flattening, pancaking, skew, twist,
                flux, sigma, polarity, chirality."""
        for k, v in kwargs.items():
            if k in self._props:
                setattr(self, k, v)
            else:
                raise KeyError("Unsupported parameter encountered.")

    @BaseFRi3D.latitude.setter
    def latitude(self, val):
        self._latitude = subtract_period(val, np.pi * 2)

    @BaseFRi3D.longitude.setter
    def longitude(self, val):
        self._longitude = subtract_period(val, np.pi * 2)

    @BaseFRi3D.toroidal_height.setter
    def toroidal_height(self, val):
        """Sets not only toroidal height explicitly but also poloidal
        height implicitly.
        """
        if val > 0:
            self._toroidal_height = val
            if self.half_height is not None:
                self._poloidal_height = self.toroidal_height * np.tan(self.half_height)
        else:
            raise ValueError("Toroidal height should be positive.")

    @BaseFRi3D.half_width.setter
    def half_width(self, val):
        """Sets not only half width explicitly but also width
        coefficient implicitly.
        """
        if val > 0 and val < np.pi * 2:
            self._half_width = val
            self._coeff_angle = np.pi / 2 / self.half_width
        else:
            raise ValueError("Half width should positive and less than 2pi.")

    @BaseFRi3D.half_height.setter
    def half_height(self, val):
        """Sets not only half height explicitly but also poloidal height
        implicitly.
        """
        if val > 0 and val < np.pi:
            self._half_height = val
            if self.toroidal_height is not None:
                self._poloidal_height = self.toroidal_height * np.tan(self.half_height)
        else:
            raise ValueError("Half height should positive and less than pi.")

    @BaseFRi3D.tilt.setter
    def tilt(self, val):
        self._tilt = subtract_period(val, np.pi * 2)

    @BaseFRi3D.flattening.setter
    def flattening(self, val):
        if val >= 0 and val <= 1:
            self._flattening = val
        else:
            raise ValueError("Flattening should be greater than 0 and less than 1.")

    @BaseFRi3D.pancaking.setter
    def pancaking(self, val):
        if val > 0 and val <= 1:
            self._pancaking = val
        else:
            raise ValueError("Pancaking should be greater than 0 and less than 1.")

    @BaseFRi3D.twist.setter
    def twist(self, val):
        """If negative twist is submitted the setter will revert the
        chirality.
        """
        if val >= 0:
            self._twist = np.absolute(val)
        else:
            raise ValueError("Twist should be positive.")

    @BaseFRi3D.flux.setter
    def flux(self, val):
        if val >= 0:
            self._flux = np.absolute(val)
        else:
            raise ValueError("Flux should be positive.")

    @BaseFRi3D.sigma.setter
    def sigma(self, val):
        if val > 0:
            self._sigma = val
        else:
            raise ValueError("Sigma should be positive.")

    @BaseFRi3D.polarity.setter
    def polarity(self, val):
        if val == 1 or val == -1:
            self._polarity = val
        else:
            raise ValueError("Polarity should be +1 or -1.")

    @BaseFRi3D.chirality.setter
    def chirality(self, val):
        if val == 1 or val == -1:
            self._chirality = val
        else:
            raise ValueError("Chirality should be +1 or -1.")

    def vanilla_axis_height(self, phi):
        """Evaluates the axis function r(phi) in polar coordinates. Note
        that rotational skewing is not taken into account.

        Args:
            phi (scalar or array_like): Angular coordinate of a point on
                the axis [rad] in polar coordinates, lies in the range
                [-half_width, half_width].

        Returns:
            scalar or array: Radial coordinate of the point of the axis
                in polar coordinates [m].
        """
        phi = np.asarray(phi)
        scalar_input = False
        if phi.ndim == 0:
            phi = phi[None]
            scalar_input = True
        res = np.ones(phi.shape) * np.nan
        mask = np.logical_and(phi >= -self.half_width, phi <= self.half_width)
        res[mask] = self.toroidal_height * np.cos(np.pi / 2 / self.half_width * phi[mask]) ** self.flattening
        if scalar_input:
            return res.squeeze()
        return res

    def vanilla_axis_distance(self, phi, r_sc, phi_sc):
        """Evaluates the distance to the given point of the axis
        (defined by `phi`) from an arbitrary point in space (defined by
        `r_sc` and `phi_sc`). Note that rotational skewing is not taken
        into account.

        Args:
            phi (scalar or array_like): Angular coordinate of a point on
                the axis [rad] in polar coordinates, lies in the range
                [-half_width, half_width].
            r_sc (scalar or array_like): Radial coordinate of a point in
                space [m].
            phi_sc (scalar or array_like): Radial coordinate of a point
                in space [rad].

        Returns:
            scalar or array: Distance from (`r_sc`, `phi_sc`) to the
                `phi` point of the axis [m].
        """
        phi = np.asarray(phi)
        scalar_input = False
        if phi.ndim == 0:
            phi = phi[None]
            scalar_input = True
        r_sc = np.asarray(r_sc)
        phi_sc = np.asarray(phi_sc)
        res = np.sqrt(
            (self.vanilla_axis_height(phi) * np.cos(phi) - r_sc * np.cos(phi_sc)) ** 2
            + (self.vanilla_axis_height(phi) * np.sin(phi) - r_sc * np.sin(phi_sc)) ** 2
        )
        if scalar_input:
            return res.squeeze()
        return res

    def vanilla_axis_min_distance(self, r_sc, phi_sc):
        """Estimates the minimal distance to the axis from an arbitrary
        point in space (defined by `r_sc`, `phi_sc`). Note that\
        rotational skewing is not taken into account.

        Args:
            r_sc (scalar): Radial coordinate of a point in space [m].
            phi_sc (scalar): Angular coordinate of a point in space
                [rad].

        Returns:
            Tuple: minimal distance from (`r_sc`, `phi_sc`) to the
            axis [m] and angle phi of the corresponding point on the
            axis [rad].
        """
        phi = minimize_scalar(
            lambda phi: self.vanilla_axis_distance(phi, r_sc, phi_sc),
            bounds=[-self.half_width, self.half_width],
            method="bounded",
        ).x
        return (self.vanilla_axis_distance(phi, r_sc, phi_sc), phi)

    def vanilla_axis_normal_angle(self, phi):
        """Evaluates the angle between normal and radial direction
            at a given location.

        Args:
            phi (scalar or array_like): Angular coordinate of a point on
                the axis [rad], lies in the range
                [-half_width, half_width].

        Returns:
            scalar or array: Angle between normal and radial direction
            at a given location [rad].
        """
        phi = np.asarray(phi)
        scalar_input = False
        if phi.ndim == 0:
            phi = phi[None]
            scalar_input = True
        res = np.abs(np.arctan(self._coeff_angle * self.flattening * np.tan(self._coeff_angle * phi)))
        if scalar_input:
            return res.squeeze()
        return res

    def vanilla_axis_length(self, phi):
        """Evaluates length of the axis. It is an approximation and also
        does not take into account rotational skewing.

        Args:
            phi (scalar or array_like): Angular coordinate of a point on
                the axis [rad] in polar coordinates, lies in the range
                [-half_width, half_width].

        Returns:
            scalar or array: Length [m] of the axis from origin
                footpoint towards the location defined by the angular
                coordinate `phi`.
        """
        phi = np.asarray(phi)
        scalar_input = False
        if phi.ndim == 0:
            phi = phi[None]
            scalar_input = True
        res = np.array(
            [
                quad(
                    LowLevelCallable.from_cython(lib, "vanilla_axis_dlength"),
                    -self.half_width,
                    p,
                    (self.toroidal_height, self.half_width, self.flattening),
                )[0]
                for p in phi
            ]
        )
        if scalar_input:
            return res.squeeze()
        return res

    def shell(self, phi=None, theta=np.linspace(0, np.pi * 2, 24)):
        """Evaluates the 3D shell of the flux rope.

        Args:
            phi (scalar or array_like, optional): defines angular
                sampling along the axis [rad].
            theta (scalar or array_like, optional) defines angular
                sampling of the cross-section [rad].

        Returns:
            tuple: (x, y, z) coordinates of the shell points [m]. Each
                element of the tuple is either a scalar or 2D array.
        """
        # Sets the default axial coordinate
        # to cover all of the shell's length
        if phi is None:
            phi = np.linspace(-self.half_width, self.half_width, 50)
        # Provides support of scalar input
        phi = np.asarray(phi)
        theta = np.asarray(theta)
        scalar_input = False
        if phi.ndim == 0 and theta.ndim == 0:
            phi = phi[None]
            theta = theta[None]
            scalar_input = True
        # Checks that all of the axial coordinates are valid
        if np.any(phi < -self.half_width) or np.any(phi > self.half_width):
            raise ValueError("phi should be in the range of angular width")
        # Defines distance to axis and normal angle for later usage
        axis_height = self.vanilla_axis_height(phi)
        axis_normal = self.vanilla_axis_normal_angle(phi)
        # Starts with a cylinder aligned with Z axis in cylindrical CS
        z = self.vanilla_axis_length(phi)
        # Tapers the cylinder
        r = np.ones(phi.shape) * self._poloidal_height * axis_height / self.toroidal_height
        # Converts coordinates to meshgrid
        z, _ = np.meshgrid(z, theta, indexing="ij")
        r, _ = np.meshgrid(r, theta, indexing="ij")
        axis_height, _ = np.meshgrid(axis_height, theta, indexing="ij")
        axis_normal, _ = np.meshgrid(axis_normal, theta, indexing="ij")
        phi, theta = np.meshgrid(phi, theta, indexing="ij")
        # Applies pancaking
        rx = np.copy(r)
        ry = np.copy(r)
        rx *= 1 - (1 - self.pancaking) / np.sqrt(
            1 + (self.flattening * self._coeff_angle * np.tan(self._coeff_angle * phi)) ** 2
        )
        r = rx * ry / np.sqrt((ry * np.cos(theta)) ** 2 + (rx * np.sin(theta)) ** 2)
        # Bends the cylinder to FR shape
        x = axis_height * np.cos(phi) + r * np.cos(theta) * np.cos(axis_normal + np.abs(phi))
        y = axis_height * np.sin(phi) + r * np.cos(theta) * np.sin(axis_normal + np.abs(phi)) * np.sign(phi)
        z = r * np.sin(theta)
        # Applies correction for radial expansion
        r, _, _ = cs.cart2cyl(x, y, z)
        _, theta, phi = cs.cart2sp(x, y, z)
        # TODO: implement pancaking transformation
        x, y, z = cs.sp2cart(r, theta, phi)
        # Orients the FR direction and tilt
        T = cs.mx_rot(-self.latitude, self.longitude, self.tilt)
        x, y, z = cs.mx_apply(T, x, y, z)
        # Applies rotational skew deformation to the FR
        r, phi, z = cs.cart2cyl(x, y, z)
        phi += self.skew * (1 - r / self.toroidal_height)
        x, y, z = cs.cyl2cart(r, phi, z)
        # Handles the scalar output
        if scalar_input:
            return (x.squeeze(), y.squeeze(), z.squeeze())
        return (x, y, z)

    def line(self, r=0, phi=None, theta=0):
        """Evaluates the 3D magnetic field line of the flux rope.

        Args:
            r (scalar, optional): Relative radial coordinate of the line
                origin in the origin footpoint cross-section [m],
                goes from 0 (center) to 1 (edge).
            phi (scalar or array_like, optional): Angular coordinate
                along the axis [rad].
            theta (scalar, optional): angular coordinate
                in the cross-section [rad].

        Returns:
            tuple: (x, y, z, b) scalars or arrays with coordinates of
                line points and scalar or array with total magnetic
                field along the line.
        """
        # Sets the default axial coordinate
        # to cover all of the shell's length
        if phi is None:
            phi = np.linspace(-self.half_width, self.half_width, 50)
        # Provides support of scalar input
        phi = np.asarray(phi)
        scalar_input = False
        if phi.ndim == 0 and theta.ndim == 0:
            phi = phi[None]
            scalar_input = True
        # Checks that all of the axial coordinates are valid
        if np.any(phi < -self.half_width) or np.any(phi > self.half_width):
            raise ValueError("phi should be in the range of angular width")
        # Checks that the radial coordinate is valid
        if r < 0 or r > 1:
            raise ValueError("r should be in the range [0, 1]")
        # Defines distance to axis and normal angle for later usage
        axis_height = self.vanilla_axis_height(phi)
        axis_normal = self.vanilla_axis_normal_angle(phi)
        # Applies twist
        twist = np.array(
            [
                self.twist
                / quad(
                    LowLevelCallable.from_cython(lib, "vanilla_axis_height"),
                    -self.half_width,
                    self.half_width,
                    (self.toroidal_height, self.half_width, self.flattening),
                )[0]
                * quad(
                    LowLevelCallable.from_cython(lib, "vanilla_axis_height"),
                    -self.half_width,
                    p,
                    (self.toroidal_height, self.half_width, self.flattening),
                )[0]
                for p in phi
            ]
        )
        theta = twist * np.pi * 2.0 * self.chirality + np.ones(phi.size) * theta
        # Applies tapering and pancaking
        rx = axis_height * self._poloidal_height / self.toroidal_height
        ry = axis_height * self._poloidal_height / self.toroidal_height
        pancaking = 1 - (1 - self.pancaking) / np.sqrt(
            1 + (self.flattening * self._coeff_angle * np.tan(self._coeff_angle * phi)) ** 2
        )
        rx *= pancaking
        theta = np.arctan2(np.sin(theta), np.cos(theta) * pancaking)
        rtot = rx * ry / np.sqrt((ry * np.cos(theta)) ** 2 + (rx * np.sin(theta)) ** 2)
        r *= rtot
        # Estimates magnetic field
        b_ax = np.array(
            [
                self.flux
                / dblquad(
                    LowLevelCallable.from_cython(lib, "vanilla_axis_rdflux"),
                    0,
                    2 * np.pi,
                    lambda theta: 0,
                    lambda theta: rx[i] * ry[i] / np.sqrt((ry[i] * np.cos(theta)) ** 2 + (rx[i] * np.sin(theta)) ** 2),
                    (
                        phi[i],
                        self.toroidal_height,
                        self.half_width,
                        self.half_height,
                        self.flattening,
                        self.pancaking,
                        self.twist,
                        self.sigma,
                        quad(
                            LowLevelCallable.from_cython(lib, "vanilla_axis_height"),
                            -self.half_width,
                            self.half_width,
                            (self.toroidal_height, self.half_width, self.flattening),
                        )[0],
                    ),
                )[0]
                for i in range(phi.size)
            ]
        )
        # sigmax = self.sigma*pancaking
        sigmay = self.sigma
        b = b_ax * np.exp(
            -(r * np.cos(theta) / ry) ** 2 / 2 / sigmay ** 2 - (r * np.sin(theta) / ry) ** 2 / 2 / sigmay ** 2
        )
        # Bends the cylinder to FR shape
        x = axis_height * np.cos(phi) + r * np.cos(theta) * np.cos(axis_normal + np.abs(phi))
        y = axis_height * np.sin(phi) + r * np.cos(theta) * np.sin(axis_normal + np.abs(phi)) * np.sign(phi)
        z = r * np.sin(theta)
        # Applies correction for radial expansion
        r, _, _ = cs.cart2cyl(x, y, z)
        _, theta, phi = cs.cart2sp(x, y, z)
        x, y, z = cs.sp2cart(r, theta, phi)
        # Orients the FR direction and tilt
        T = cs.mx_rot(-self.latitude, self.longitude, self.tilt)
        x, y, z = cs.mx_apply(T, x, y, z)
        # Applies rotational skew deformation to the FR
        r, phi, z = cs.cart2cyl(x, y, z)
        phi += self.skew * (1 - r / self.toroidal_height)
        x, y, z = cs.cyl2cart(r, phi, z)
        # Handles the scalar output
        if scalar_input:
            return (x.squeeze(), y.squeeze(), z.squeeze(), b.squeeze())
        return (x, y, z, b)

    def data(self, x, y, z, dphi=1e-5):
        """Evaluates magnetic field measurements at a given point (or
        trajectory) in space.

        Args:
            x (scalar or array_like): X-component of coordinate in
                space.
            y (scalar or array_like): Y-component of coordinate in
                space.
            z (scalar or array_like): Z-component of coordinate in
                space.
            dphi (scalar, optional): axis angle step used to integrate
                the magnetic field measurement.

        Returns:
            tuple: (array, array), magnetic field measurements array of
                shape (3) or (3, n) and array of coefficients used for
                local speed estimation of shape (2) or (2, n).
        """
        # Provides support of scalar input
        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)
        scalar_input = False
        if x.ndim == 0 and y.ndim == 0 and z.ndim == 0:
            x = x[None]
            y = y[None]
            z = z[None]
            scalar_input = True
        # Reverses the skew deformation
        r, theta, phi = cs.cart2sp(x, y, z)
        phi -= self.skew * (1 - r / self.toroidal_height)
        x, y, z = cs.sp2cart(r, theta, phi)
        # Reverses the orientation
        T = cs.mx_rot_reverse(self.latitude, -self.longitude, -self.tilt)
        x, y, z = cs.mx_apply(T, x, y, z)
        # Reverses correction for radial expansion
        r, theta, phi = cs.cart2sp(x, y, z)
        x, y, z = cs.cyl2cart(r, phi, z)
        # Here r, phi, z are cylindrical coordinates
        # inside axis loop mask
        with np.errstate(invalid="ignore"):
            mask_inside = self.vanilla_axis_height(phi) >= r
        # Finds the closest point on axis
        v_vanilla_axis_min_distance = np.vectorize(self.vanilla_axis_min_distance, otypes=[np.float64, np.float64])
        _, phi_ax = v_vanilla_axis_min_distance(r, phi)
        r_ax = self.vanilla_axis_height(phi_ax)
        # Estimates relative radius and polar angle inside cross-section
        x_ax, y_ax, z_ax = cs.cyl2cart(r_ax, phi_ax, np.zeros(r_ax.size))
        dx = x - x_ax
        dy = y - y_ax
        dz = z - z_ax
        r_abs = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        theta = np.arctan2(dz, np.sqrt(dx ** 2 + dy ** 2))
        theta[mask_inside] = np.pi - theta[mask_inside]
        rx = r_ax * self._poloidal_height / self.toroidal_height
        ry = r_ax * self._poloidal_height / self.toroidal_height
        pancaking = 1 - (1 - self.pancaking) / np.sqrt(
            1 + (self.flattening * self._coeff_angle * np.tan(self._coeff_angle * phi_ax)) ** 2
        )
        rx *= pancaking
        r_tot = rx * ry / np.sqrt((ry * np.cos(theta)) ** 2 + (rx * np.sin(theta)) ** 2)
        r = r_abs / r_tot
        theta = np.arctan2(r_abs * np.sin(theta), r_abs * np.cos(theta) / pancaking)
        # Reverses twist
        twist = np.array(
            [
                self.twist
                / quad(
                    LowLevelCallable.from_cython(lib, "vanilla_axis_height"),
                    -self.half_width,
                    self.half_width,
                    (self.toroidal_height, self.half_width, self.flattening),
                )[0]
                * quad(
                    LowLevelCallable.from_cython(lib, "vanilla_axis_height"),
                    -self.half_width,
                    p,
                    (self.toroidal_height, self.half_width, self.flattening),
                )[0]
                for p in phi_ax
            ]
        )
        theta -= twist * np.pi * 2.0 * self.chirality
        # Estimates magnetic field and speed coefficients along sc trajectory
        b_list = []
        vc_list = []
        for i in range(r.size):
            if r[i] <= 1:
                x, y, z, b = self.line(r[i], [phi_ax[i] - dphi, phi_ax[i] + dphi], theta[i])
                if b.size != 2:
                    b_list.append([np.nan, np.nan, np.nan])
                    vc_list.append([np.nan, np.nan])
                else:
                    vtc = r_ax[i] / self.toroidal_height
                    vpc = (
                        r_ax[i]
                        / self.toroidal_height
                        * (np.sqrt(np.mean(x) ** 2 + np.mean(y) ** 2 + np.mean(z) ** 2) - r_ax[i])
                        / self._poloidal_height
                        * self.pancaking
                        * np.cos(self.vanilla_axis_normal_angle(phi_ax[i]))
                    )
                    dr = np.array([x[1] - x[0], y[1] - y[0], z[1] - z[0]])
                    dr /= np.linalg.norm(dr)
                    # print(
                    #     np.mean(b),
                    #     vars(self)
                    # )
                    b_list.append(dr * np.mean(b) * self.polarity)
                    vc_list.append(np.array([vtc, vpc]))
            else:
                b_list.append([np.nan, np.nan, np.nan])
                vc_list.append([np.nan, np.nan])
        if scalar_input:
            return (np.array(b_list).squeeze(), np.array(vc_list).squeeze())
        return (np.array(b_list), np.array(vc_list))

    def axis_min_distance(self, x, y, z, dphi=1e-5):
        """Estimates the distance to the axis.

        Args:
            x (scalar): X-component of coordinate of the point in space.
            y (scalar): Y-component of coordinate of the point in space.
            z (scalar): Z-component of coordinate of the point in space.
            dphi (scalar, optional): Angle section used to deduce
                axis direction at its closest point.

        Returns:
            tuple: (scalar, array, array), impact distance [m],
                (x, y, z)-coordinates of the closest point on the axis,
                (x, y, z) unit vector tangent to the the axis near the
                closest point.
        """
        x0 = x
        y0 = y
        z0 = z
        # reverse skew
        r, theta, phi = cs.cart2sp(x, y, z)
        phi -= self.skew * (1 - r / self.toroidal_height)
        x, y, z = cs.sp2cart(r, theta, phi)
        # reverse orientation
        T = cs.mx_rot_reverse(self.latitude, -self.longitude, -self.tilt)
        x, y, z = cs.mx_apply(T, x, y, z)
        # reverse pancaking
        r, theta, phi = cs.cart2sp(x, y, z)
        theta = theta / self.pancaking * self.half_height
        # get r_ax and phi_ax of the closest point on axis
        _, phi_ax = self.vanilla_axis_min_distance(r * np.cos(theta), phi)
        r_ax = self.vanilla_axis_height(phi_ax)
        x, y, z = cs.sp2cart(r_ax, np.zeros(r_ax.size), phi_ax)
        # orientation
        T = cs.mx_rot(-self.latitude, self.longitude, self.tilt)
        x, y, z = cs.mx_apply(T, x, y, z)
        # skew
        r, theta, phi = cs.cart2sp(x, y, z)
        phi += self.skew * (1 - r / self.toroidal_height)
        x, y, z = cs.sp2cart(r, theta, phi)
        # get r_ax and phi_ax of the closest delta points on axis
        phi_ax1 = phi_ax - dphi
        phi_ax2 = phi_ax + dphi
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
        phi += self.skew * (1 - r / self.toroidal_height)
        x1, y1, z1 = cs.sp2cart(r, theta, phi)
        r, theta, phi = cs.cart2sp(x2, y2, z2)
        phi += self.skew * (1 - r / self.toroidal_height)
        x2, y2, z2 = cs.sp2cart(r, theta, phi)
        d = np.array([x2, y2, z2]) - np.array([x1, y1, z1])
        d /= np.linalg.norm(d)
        return (np.linalg.norm(np.array([x - x0, y - y0, z - z0])), np.array([x, y, z]).squeeze(), d.squeeze())

    def map(
        self,
        x,
        y,
        z,
        xmc,
        ymc,
        xgrid=np.linspace(-0.5, 0.5, 100) * 149597870700,
        ygrid=np.linspace(-0.5, 0.5, 100) * 149597870700,
    ):
        """Calculates magnetic field map, i.e., cross-section of the
        flux rope, in any plane.

        Args:
            x (scalar): X-component of coordinate of the point in space.
            y (scalar): Y-component of coordinate of the point in space.
            z (scalar): Z-component of coordinate of the point in space.
            xmc (array_like): basis unit vector for X axis in coordinate
                system of magnetic cloud, i.e., flux-rope cross-section,
                of size (3)
            ymc (array_like): basis unit vector for Y axis in coordinate
                system of magnetic cloud, i.e., flux-rope cross-section,
                of size (3)
            xgrid (array_like, optional): map grid in X direction.
            ygrid (array_like, optional): map grid in Y direction.

        Returns:
            array: transverse magnetic field 2D array in all the points
                of the provided grid.
        """
        xmc = np.asarray(xmc)
        ymc = np.asarray(ymc)
        zmc = np.cross(xmc, ymc)
        xg = np.zeros([xgrid.size, ygrid.size])
        yg = np.zeros([xgrid.size, ygrid.size])
        zg = np.zeros([xgrid.size, ygrid.size])
        for i in range(xgrid.size):
            for k in range(ygrid.size):
                p = np.array([x, y, z]) + xgrid[i] * xmc + ygrid[k] * ymc
                xg[i, k] = p[0]
                yg[i, k] = p[1]
                zg[i, k] = p[2]
        b, _ = self.data(xg.flatten(), yg.flatten(), zg.flatten())
        bmap = np.array([np.dot(b[i, :], zmc) for i in range(b.shape[0])])
        # bmap = np.array([np.linalg.norm(b[i, :]) for i in range(b.shape[0])])
        bmap = np.reshape(bmap, [xgrid.size, ygrid.size]).T
        return bmap

    def forcemap(
        self,
        x,
        y,
        z,
        xmc,
        ymc,
        xgrid=np.linspace(-0.05, 0.05, 100) * 149597870700,
        ygrid=np.linspace(-0.05, 0.05, 100) * 149597870700,
    ):
        """Calculates force map, i.e., |jxB| of the flux rope, in any
        plane.

        Args:
            x (scalar): X-component of coordinate of the point in space.
            y (scalar): Y-component of coordinate of the point in space.
            z (scalar): Z-component of coordinate of the point in space.
            xmc (array_like): basis unit vector for X axis in coordinate
                system of magnetic cloud, i.e., flux-rope cross-section,
                of size (3)
            ymc (array_like): basis unit vector for Y axis in coordinate
                system of magnetic cloud, i.e., flux-rope cross-section,
                of size (3)
            xgrid (array_like, optional): map grid in X direction.
            ygrid (array_like, optional): map grid in Y direction.

        Returns:
            array: force 2D array in all the points of the provided
                grid.
        """
        xmc = np.asarray(xmc)
        ymc = np.asarray(ymc)
        zmc = np.cross(xmc, ymc)
        xg = np.zeros([xgrid.size, ygrid.size])
        yg = np.zeros([xgrid.size, ygrid.size])
        zg = np.zeros([xgrid.size, ygrid.size])
        forcemap = np.zeros([xgrid.size, ygrid.size])
        import numdifftools as nd

        def b(pos):
            return self.data(pos[0], pos[1], pos[2])[0]

        for i in range(xgrid.size):
            for k in range(ygrid.size):
                p = np.array([x, y, z]) + xgrid[i] * xmc + ygrid[k] * ymc
                jac = nd.Jacobian(b)(p)
                # j = np.array([
                #     jac[2, 1]-jac[1, 2],
                #     jac[0, 2]-jac[2, 0],
                #     jac[1, 0]-jac[0, 1]
                # ])/1.25663706e-06
                # j /= np.linalg.norm(j)
                # b_ = b(p)
                # b_ /= np.linalg.norm(b_)
                # forcemap[i, k] = np.linalg.norm(np.cross(j, b_))
                # forcemap[i, k] = np.arccos(np.dot(j, b_))
                # forcemap[i, k] = np.linalg.norm(
                #     np.cross(
                #         np.array([
                #             jac[2, 1]-jac[1, 2],
                #             jac[0, 2]-jac[2, 0],
                #             jac[1, 0]-jac[0, 1]
                #         ])/1.25663706e-06,
                #         b(p)
                #     )
                # )
                j = np.array([jac[2, 1] - jac[1, 2], jac[0, 2] - jac[2, 0], jac[1, 0] - jac[0, 1]]) / 1.25663706e-06
                b_ = b(p)
                b_ /= np.linalg.norm(b_)
                jpar = np.dot(j, b_) * b_
                jperp = j - jpar
                forcemap[i, k] = np.linalg.norm(jperp) / np.linalg.norm(jpar)
        return forcemap.T


class DynamicFRi3D(BaseFRi3D):
    """FRi3D model dynamic class. It provides dynamic description of the
    model.
    """

    def __init__(self, **kwargs):
        super(DynamicFRi3D, self).__init__()
        self.__sfr = StaticFRi3D()
        self.latitude = kwargs.get("latitude", lambda t: self.__sfr.latitude)
        self.longitude = kwargs.get("longitude", lambda t: self.__sfr.longitude)
        self.toroidal_height = kwargs.get("toroidal_height", lambda t: self.__sfr.toroidal_height)
        self.half_width = kwargs.get("half_width", lambda t: self.__sfr.half_width)
        self.half_height = kwargs.get("half_height", lambda t: self.__sfr.half_height)
        self.tilt = kwargs.get("tilt", lambda t: self.__sfr.tilt)
        self.flattening = kwargs.get("flattening", lambda t: self.__sfr.flattening)
        self.pancaking = kwargs.get("pancaking", lambda t: self.__sfr.pancaking)
        self.skew = kwargs.get("skew", lambda t: self.__sfr.skew)
        self.twist = kwargs.get("twist", lambda t: self.__sfr.twist)
        self.flux = kwargs.get("flux", lambda t: self.__sfr.flux)
        self.sigma = kwargs.get("sigma", lambda t: self.__sfr.sigma)
        self.polarity = kwargs.get("polarity", lambda t: self.__sfr.polarity)
        self.chirality = kwargs.get("chirality", lambda t: self.__sfr.chirality)

    def modify(self, **kwargs):
        """Modifies the model parameters.

        Args:
            **kwargs: Callable model parameters. Allowed keywords:
                latitude, longitude, toroidal_height, half_width,
                half_height, tilt, flattening, pancaking, skew, twist,
                flux, sigma, polarity, chirality. Each parameter should
                be function of time."""
        for k, v in kwargs.items():
            if k in self._props:
                setattr(self, k, v)
            else:
                raise KeyError("Unsupported parameter encountered.")

    @BaseFRi3D.latitude.setter
    def latitude(self, func):
        if callable(func):
            self._latitude = func
        else:
            raise ValueError("Latitude profile is expected to be a callable.")

    @BaseFRi3D.longitude.setter
    def longitude(self, func):
        if callable(func):
            self._longitude = func
        else:
            raise ValueError("Longitude profile is expected to be a callable.")

    @BaseFRi3D.toroidal_height.setter
    def toroidal_height(self, func):
        if callable(func):
            self._toroidal_height = func
        else:
            raise ValueError("Toroidal height profile is expected to be a callable.")

    @BaseFRi3D.half_width.setter
    def half_width(self, func):
        if callable(func):
            self._half_width = func
        else:
            raise ValueError("Half width profile is expected to be a callable.")

    @BaseFRi3D.half_height.setter
    def half_height(self, func):
        if callable(func):
            self._half_height = func
        else:
            raise ValueError("Half height profile is expected to be a callable.")

    @BaseFRi3D.tilt.setter
    def tilt(self, func):
        if callable(func):
            self._tilt = func
        else:
            raise ValueError("Tilt profile is expected to be a callable.")

    @BaseFRi3D.flattening.setter
    def flattening(self, func):
        if callable(func):
            self._flattening = func
        else:
            raise ValueError("Flattening profile is expected to be a callable.")

    @BaseFRi3D.pancaking.setter
    def pancaking(self, func):
        if callable(func):
            self._pancaking = func
        else:
            raise ValueError("Pancaking profile is expected to be a callable.")

    @BaseFRi3D.skew.setter
    def skew(self, func):
        if callable(func):
            self._skew = func
        else:
            raise ValueError("Skew profile is expected to be a callable.")

    @BaseFRi3D.twist.setter
    def twist(self, func):
        if callable(func):
            self._twist = func
        else:
            raise ValueError("Twist profile is expected to be a callable.")

    @BaseFRi3D.flux.setter
    def flux(self, func):
        if callable(func):
            self._flux = func
        else:
            raise ValueError("Flux profile is expected to be a callable.")

    @BaseFRi3D.sigma.setter
    def sigma(self, func):
        if callable(func):
            self._sigma = func
        else:
            raise ValueError("Sigma profile is expected to be a callable.")

    @BaseFRi3D.polarity.setter
    def polarity(self, func):
        if callable(func):
            self._polarity = func
        else:
            raise ValueError("Polarity profile is expected to be a callable.")

    @BaseFRi3D.chirality.setter
    def chirality(self, func):
        if callable(func):
            self._chirality = func
        else:
            raise ValueError("Chirality profile is expected to be a callable.")

    def snapshot(self, t):
        """Returns a snapshot of the FRi3D model at a given moment of
        time.

        Args:
            t (scalar): timestamp [s].

        Returns:
            StaticFRi3D: a static model object for a given time.
        """
        self.__sfr.modify(
            latitude=self.latitude(t),
            longitude=self.longitude(t),
            toroidal_height=self.toroidal_height(t),
            half_width=self.half_width(t),
            half_height=self.half_height(t),
            tilt=self.tilt(t),
            flattening=self.flattening(t),
            pancaking=self.pancaking(t),
            skew=self.skew(t),
            twist=self.twist(t),
            flux=self.flux(t),
            sigma=self.sigma(t),
            polarity=self.polarity(t),
            chirality=self.chirality(t),
        )
        return self.__sfr

    def insitu(self, t, x, y, z):
        """Calculates synthetic in-situ measurements for given time
        interval and a given point of space.

        Args:
            t (scalar or array_like): time (unix timestamp), for which
                the in-situ measurements are estimated. Can be a single
                timestamp or an array of timestamps.
            x (scalar or callable): X-component of synthetic spacecraft
                coordinates. Can be a single point in space or func(t),
                which describe the spacecraft trajectory.
            y (scalar or callable): Y-component of synthetic spacecraft
                coordinates. Can be a single point in space or func(t),
                which describe the spacecraft trajectory.
            z (scalar or callable): Z-component of synthetic spacecraft
                coordinates. Can be a single point in space or func(t),
                which describe the spacecraft trajectory.

        Returns:
            tuple: (array, scalar or array): magnetic field components
                array of shape (3) or (3, n) and absolute speed, which
                can be a scalar or an array of shape (n).
        """
        t = np.asarray(t)
        scalar_input = False
        if t.ndim == 0:
            t = t[None]
            scalar_input = True
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
        vt = []
        for _t in t:
            _b, _c = self.snapshot(_t).data(x(_t), y(_t), z(_t))
            _b = _b[:]
            _c = _c[:]
            b.append(_b.ravel())
            vt.append(
                _c[0] * (self.toroidal_height(_t) - self.toroidal_height(_t - 1))
                + _c[1]
                * (
                    self.toroidal_height(_t) * np.tan(self.half_height(_t))
                    - self.toroidal_height(_t - 1) * np.tan(self.half_height(_t - 1))
                )
            )
        b = np.array(b)
        vt = np.array(vt)
        if scalar_input:
            return (b.squeeze(), vt.squeeze())
        return (b, vt)

    def impact(self, t, x, y, z):
        """Estimates the impact distance for a given time interval and
            at a given point in space (or trajectory).

        Args:
            t (scalar or array_like): time (unix timestamp), for which
                the impact distance is estimated. Can be a single
                timestamp or an array of timestamps.
            x (scalar or callable): X-component of synthetic spacecraft
                coordinates. Can be a single point in space or func(t),
                which describe the spacecraft trajectory.
            y (scalar or callable): Y-component of synthetic spacecraft
                coordinates. Can be a single point in space or func(t),
                which describe the spacecraft trajectory.
            z (scalar or callable): Z-component of synthetic spacecraft
                coordinates. Can be a single point in space or func(t),
                which describe the spacecraft trajectory.

        Returns:
            tuple: (float, int): impact distance and the timestamp of
                the closest approach.
        """
        t = np.asarray(t)
        if t.ndim == 0:
            t = t[None]
        if not callable(x):
            _x = x
            x = lambda t: _x
        if not callable(y):
            _y = y
            y = lambda t: _y
        if not callable(z):
            _z = z
            z = lambda t: _z
        res = minimize_scalar(
            lambda _t: self.snapshot(_t).axis_min_distance(x(_t), y(_t), z(_t))[0],
            bounds=(t[0], t[-1]),
            method="bounded",
        )
        return (res.fun, res.x)


def subtract_period(value, period):
    """Reduces angle by period.

    Args:
        value (scalar): initial angle [rad].
        period (scalar): period [rad].

    Returns:
        scalar: angle reduced by correct number of periods.
    """
    return value - math.copysign(value, 1) * (math.fabs(value) // period) * period
