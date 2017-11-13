from .static_fri3d import StaticFRi3D
from .dynamic_fri3d import DynamicFRi3D
# from .base_fri3d_fit import BaseFRi3DFit
# from .fri3d_fit_cor import FRi3DFitCor
# from .fri3d_fit_insitu import FRi3DFitInSitu

"""
class Evolution:
    # Evolution class provides dynamic description of the FRi3D model.

    def __init__(self, **kwargs):
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
        chirality=1.0):
        
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
