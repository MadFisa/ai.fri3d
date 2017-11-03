"""Defines the FRi3D class which describes CME structure and Evolution
class which handles its evolution.
"""
import math
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import fixed_quad
from scipy.optimize import minimize_scalar
from astropy import constants as c
from astropy import units as u

class FRi3D:
    """FRi3D model class. It provides static description of the model.
    """
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=E1101
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
        self._polarity = None
        self._chirality = None
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
        self.init()

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

    def _vanilla_axis_r(
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
        return toroidal_height*np.cos(coeff_angle*phi)**flattening

    def _vanilla_axis_dr(
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
            (-1)*flattening*coeff_angle*np.tan(coeff_angle*phi)
            *self._vanilla_axis_r(
                phi,
                toroidal_height=toroidal_height,
                coeff_angle=coeff_angle,
                flattening=flattening)
        )

    def _vanilla_axis_ds(
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
            self._vanilla_axis_r(
                phi,
                toroidal_height=toroidal_height,
                coeff_angle=coeff_angle,
                flattening=flattening
            )*np.sqrt(
                1.0+coeff_angle**2*flattening**2
                *np.tan(coeff_angle*phi)**2
            )
        )

    def _vanilla_axis_s(self, phi):
        """Evaluate length of the axis. It is an estimation and also
        does not take into account rotational skewing.
        """
        return(
            self.toroidal_height
            *self._interpolator_axis_s(
                self.coeff_angle*phi,
                self.coeff_angle,
                self.flattening
            )
        )

    def _init_interpolator_axis_s(self, ratio=1-1e-5):
        """Initialize the interpolator
        s=length(coeff_angle*phi, coeff_angle, flattening) for a curve
        defined in polar coordinates as
        r=cos(coeff_angle*phi)^flattening. Interpolation is performed
        along the following variable ranges:
        coeff_angle*phi = [-pi/2, pi/2]
        coeff_angle = [1, 18] (half_angle = [pi/36, pi/2])
        flattening = [0.1, 1.0]
        """
        coeff_angle_phi_array = np.linspace(-np.pi/2, np.pi/2, 100)
        coeff_angle_array = np.linspace(1, 18, 100)
        flattening_array = np.linspace(0.1, 1.0, 100)
        def integrate_s(coeff_angle_phi, coeff_angle, flattening, ratio=ratio):
            """Estimate length along axis by numerical integration."""
            if coeff_angle_phi <= -np.pi/2*ratio:
                return self._vanilla_axis_r(
                    coeff_angle_phi/coeff_angle,
                    toroidal_height=1,
                    coeff_angle=coeff_angle,
                    flattening=flattening
                )
            elif coeff_angle_phi < np.pi/2*ratio:
                return(
                    self._vanilla_axis_r(
                        -np.pi/2*ratio/coeff_angle,
                        toroidal_height=1,
                        coeff_angle=coeff_angle,
                        flattening=flattening
                    )+fixed_quad(
                        lambda coeff_angle_phi: self._vanilla_axis_ds(
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
                return(
                    2*_self._vanilla_axis_r(
                        -np.pi/2*ratio/coeff_angle,
                        toroidal_height=1,
                        coeff_angle=coeff_angle,
                        flattening=flattening
                    )+fixed_quad(
                        lambda coeff_angle_phi: self._vanilla_axis_ds(
                            coeff_angle_phi/coeff_angle,
                            toroidal_height=1,
                            coeff_angle=coeff_angle,
                            flattening=flattening)/coeff_angle,
                        -np.pi/2*ratio,
                        np.pi/2*ratio,
                        n=1000
                    )[0]-self._vanilla_axis_r(
                        coeff_angle_phi/coeff_angle,
                        toroidal_height=1,
                        coeff_angle=coeff_angle,
                        flattening=flattening
                    )
                )
        data = integrate_s(
            *np.meshgrid(
                coeff_angle_phi_array,
                coeff_angle_array,
                flattening_array,
                indexing='ij',
                sparse=True
            )
        )
        self._interpolator_axis_s = RegularGridInterpolator(
            (coeff_angle_phi_array, coeff_angle_array, flattening_array),
            data
        )

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
        m = np.isfinite(s)
        s = s[m]
        phi = phi[m]
        self._spline_initial_axis_s_phi = interp1d(
            s, phi,
            kind=self.spline_s_phi_kind,
            bounds_error=False,
            # fill_value='extrapolate'
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
        res = minimize_scalar(
            lambda phi: self._initial_axis_l(phi, r0, phi0),
            bounds=[-self.half_width, self.half_width],
            method='Bounded'
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
        # print(quad(self._initial_axis_ds, -self.half_width, phi))
        # # print(fixed_quad(self._initial_axis_ds, -self.half_width, phi, n=100))
        # print(fixed_quad(
        #     self._initial_axis_ds,
        #     -self.half_width,
        #     phi,
        #     n=1000
        # ))
        # for i in range(10000):
        # s = self.toroidal_height*self.flattening*np.sqrt
        s = fixed_quad(self._initial_axis_ds, -self.half_width, phi, n=1000)
        # s = quad(self._initial_axis_ds, -self.half_width, phi)
        # s = quadrature(
        #     self._initial_axis_ds,
        #     -self.half_width,
        #     phi,
        #     rtol=1e-5,
        #     maxiter=200
        # )
        # print(s)
        return s[0]

    from ai.fri3d.shell import shell
    from ai.fri3d.line import line
    from ai.fri3d.data import data
    from ai.fri3d.impact import impact

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

def subtract_period(value, period):
    return value-math.copysign(value, 1)*(math.fabs(value)//period)*period
