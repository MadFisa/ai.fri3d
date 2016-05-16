
import time

import numpy as np
from ai import cs

import scipy.special
import scipy.interpolate
import scipy.integrate
import scipy.optimize
from scipy.spatial.distance import euclidean


from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.preprocessing import scale

from fastdtw import fastdtw

AU_KM = 149597870.7
RS_KM = 6.957e5
AU_RS = AU_KM/RS_KM
RS_AU = RS_KM/AU_KM

db_prev = np.inf

BLIND_PALETTE = {
    'orange': 
        (0.901960784, 0.623529412, 0.0),
    'sky-blue': 
        (0.337254902, 0.705882353, 0.91372549),
    'bluish-green': 
        (0.0, 0.619607843, 0.450980392),
    'yellow': 
        (0.941176471, 0.894117647, 0.258823529),
    'blue': 
        (0.0, 0.447058824, 0.698039216),
    'vermillion': 
        (0.835294118, 0.368627451, 0.0),
    'reddish-purple': 
        (0.8, 0.474509804, 0.654901961)
}


class FRi3D:
    def __init__(
        self,
        latitude=0.0, 
        longitude=0.0, 
        toroidal_height=1.0, 
        poloidal_height=0.1, 
        half_width=np.pi/6.0, 
        tilt=0.0, 
        flattening=0.5, 
        pancaking=np.pi/6.0, 
        skew=0.0, 
        tapering=1.0, 
        twist=1.0, 
        flux=5e14,
        sigma=2.05,
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
        self.tapering = tapering
        self.twist = twist
        self.flux = flux
        self.sigma = sigma
        self.polarity = polarity
        self.chirality = chirality

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
        if flattening >= 0.0:
            self._flattening = flattening

    @property
    def pancaking(self):
        return self._pancaking

    @pancaking.setter
    def pancaking(self, pancaking):
        if pancaking is not None and pancaking > 0.0 and pancaking < np.pi:
            self._pancaking = pancaking
        else:
            self._pancaking = None

    @property
    def skew(self):
        return self._skew

    @skew.setter
    def skew(self, skew):
        if skew >= 0.0:
            self._skew = skew

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
    def tilt(self):
        return self._tilt

    @tilt.setter
    def tilt(self, tilt):
        self._tilt = tilt    

    @property
    def tapering(self):
        return self._tapering

    @tapering.setter
    def tapering(self, tapering):
        self._tapering = tapering

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

    def init(self):
        self._coeff_angle = np.pi/2.0/self.half_width
        self._unit_b = self.flux/(2.0*np.pi*self.sigma**2)
        self._init_spline_initial_axis_s_phi()

    # initilize spline phi(s)
    def _init_spline_initial_axis_s_phi(self):
        phi = np.linspace(-self.half_width, self.half_width, 500)
        s = np.array([self._initial_axis_s(p) for p in phi])
        self._spline_initial_axis_s_phi = scipy.interpolate.interp1d(
            s, phi, kind='cubic',
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
        s = scipy.integrate.quad(self._initial_axis_ds, -self.half_width, phi)
        return s[0]

    # model shell
    def shell(self, 
        s=np.linspace(0.0, 1.0, 50), 
        phi=np.linspace(0.0, np.pi*2.0, 24)):
        
        s = np.array(s, copy=False, ndmin=1)
        phi = np.array(phi, copy=False, ndmin=1)

        # start the FR from the solar surface
        s_max = self._initial_axis_s(self.half_width)
        s[s < RS_AU/s_max] = RS_AU/s_max
        s[s > 1.0-RS_AU/s_max] = 1.0-RS_AU/s_max
        s = np.unique(s)

        s = np.transpose(np.tile(s, (phi.size, 1)))
        phi = np.tile(phi, (s.shape[0], 1))

        # extension to full axis length
        r = np.ones(s.shape)
        z = s*self._initial_axis_s(self.half_width)
        
        # tapering
        r = (
            r*self.poloidal_height*
            (
                self._initial_axis_r(self._spline_initial_axis_s_phi(z))/
                self.toroidal_height
            )**self.tapering
        )
        x_, y_, z_ = cs.cyl2cart(r, phi, z)

        # rotation to x axis
        T = cs.mx_rot_y(-np.pi/2.0)
        x = T[0,0]*x_+T[0,1]*y_+T[0,2]*z_
        y = T[1,0]*x_+T[1,1]*y_+T[1,2]*z_
        z = T[2,0]*x_+T[2,1]*y_+T[2,2]*z_

        # bending
        phi = self._spline_initial_axis_s_phi(x)
        r = self._initial_axis_r(phi)
        t = self._initial_axis_tan(phi)
        x_ = r*np.cos(phi)+np.sin(t-phi-np.pi/2.0)*y
        y_ = r*np.sin(phi)+np.cos(t-phi-np.pi/2.0)*y
        z_ = z

        if self.pancaking is not None:
            # pancaking
            r, theta, phi = cs.cart2sp(x_, y_, z_)
            theta = (
                theta/np.arctan2(self.poloidal_height, self.toroidal_height)*
                self.pancaking
            )
            x_, y_, z_ = cs.sp2cart(r, theta, phi)

        # orientation
        T = cs.mx_rot(self.latitude, -self.longitude, -self.tilt)
        x = T[0,0]*x_+T[0,1]*y_+T[0,2]*z_
        y = T[1,0]*x_+T[1,1]*y_+T[1,2]*z_
        z = T[2,0]*x_+T[2,1]*y_+T[2,2]*z_

        # skew
        r, phi, z = cs.cart2cyl(x, y, z)
        phi += self.skew*(1.0-r/self.toroidal_height)
        x, y, z = cs.cyl2cart(r, phi, z)
        
        return (x, y, z)

    def line(self, r=0.0, phi=0.0, s=np.linspace(0.0, 1.0, 50)):
        s = np.array(s, copy=False, ndmin=1)
        
        s_max = self._initial_axis_s(self.half_width)
        s[s < RS_AU/s_max] = RS_AU/s_max
        s[s > 1.0-RS_AU/s_max] = 1.0-RS_AU/s_max
        s = np.unique(s)
        
        phi = np.ones(s.size)*phi

        # twist
        phi += s*self.twist*np.pi*2.0*self.chirality
        # elongation
        z = s*self._initial_axis_s(self.half_width)

        # distance to axis
        R = self._initial_axis_r(self._spline_initial_axis_s_phi(z))
        # cross-section radial size in the FR plane
        rx = R*self.poloidal_height/self.toroidal_height
        # cross-section radial size perp to FR plane
        ry = R*self.pancaking
        # coefficient of flux decay
        kappa = rx*ry*(AU_KM*1e3)**2
        # taper
        r *= rx
        # magnetic field
        b = self._unit_b/kappa*np.exp(
            -((r/rx)**2)/2.0/self.sigma**2
        )
        x_, y_, z_ = cs.cyl2cart(r, phi, z)

        # rotation to x
        T = cs.mx_rot_y(-np.pi/2.0)
        x = T[0,0]*x_+T[0,1]*y_+T[0,2]*z_
        y = T[1,0]*x_+T[1,1]*y_+T[1,2]*z_
        z = T[2,0]*x_+T[2,1]*y_+T[2,2]*z_

        # bend
        phi = self._spline_initial_axis_s_phi(x)
        r = self._initial_axis_r(phi)
        t = self._initial_axis_tan(phi)
        x_ = r*np.cos(phi)+np.sin(t-phi-np.pi/2.0)*y
        y_ = r*np.sin(phi)+np.cos(t-phi-np.pi/2.0)*y
        z_ = z

        # pancake
        r, theta, phi = cs.cart2sp(x_, y_, z_)
        theta = (
            theta/np.arctan2(self.poloidal_height, self.toroidal_height)*
            self.pancaking
        )
        x_, y_, z_ = cs.sp2cart(r, theta, phi)

        # orientation
        T = cs.mx_rot(self.latitude, -self.longitude, -self.tilt)
        x = T[0,0]*x_+T[0,1]*y_+T[0,2]*z_
        y = T[1,0]*x_+T[1,1]*y_+T[1,2]*z_
        z = T[2,0]*x_+T[2,1]*y_+T[2,2]*z_

        # skew
        r, phi, z = cs.cart2cyl(x, y, z)
        phi += self.skew*(1.0-r/self.toroidal_height)
        x, y, z = cs.cyl2cart(r, phi, z)

        # convert to nT
        b *= 1e9

        return (x, y, z, b)

    def cut1d(self, x, y, z):
        x = np.array(x, copy=False, ndmin=1)
        y = np.array(y, copy=False, ndmin=1)
        z = np.array(z, copy=False, ndmin=1)

        # reverse skew
        r, theta, phi = cs.cart2sp(x, y, z)
        phi -= self.skew*(1.0-r/self.toroidal_height)
        x, y, z = cs.sp2cart(r, theta, phi)

        # reverse orientation
        T = cs.mx_rot_reverse(-self.latitude, self.longitude, self.tilt)
        x_ = T[0,0]*x+T[0,1]*y+T[0,2]*z
        y_ = T[1,0]*x+T[1,1]*y+T[1,2]*z
        z_ = T[2,0]*x+T[2,1]*y+T[2,2]*z

        # reverse pancaking
        r, theta, phi = cs.cart2sp(x_, y_, z_)
        theta = (
            theta/self.pancaking*
            np.arctan2(self.poloidal_height, self.toroidal_height)
        )
        x, y, z = cs.sp2cart(r, theta, phi)

        # inside axis loop mask
        p_in = self._initial_axis_r(phi) >= r*np.cos(theta)
        # outside axis loop mask
        p_out = np.logical_not(p_in)
        # get r_ax and phi_ax of the closest point on axis
        v_initial_axis_min_l_phi = np.vectorize(self._initial_axis_min_l_phi)
        phi_ax = v_initial_axis_min_l_phi(r*np.cos(theta), phi)
        r_ax = self._initial_axis_r(phi_ax)
        # get s
        v_initial_axis_s = np.vectorize(self._initial_axis_s)
        s = v_initial_axis_s(phi_ax)/self._initial_axis_s(self.half_width)
        # get r[0,1] and phi[0,2pi] params
        x_ax, y_ax, z_ax = cs.sp2cart(r_ax, np.zeros(r_ax.size), phi_ax)
        dx = x-x_ax
        dy = y-y_ax
        dz = z-z_ax
        r_abs = np.sqrt(dx**2+dy**2+dz**2)
        r = r_abs/(r_ax*self.poloidal_height/self.toroidal_height)
        phi = (
            np.piecewise(dz, [dz < 0, dz >= 0], [-1, 1])*
            np.arccos(np.sqrt(dx**2+dy**2)/r_abs)
        )
        phi[p_in] = np.pi-phi[p_in]
        # reverse twist
        phi -= s*self.twist*np.pi*2.0*self.chirality
        # reverse rotation to x
        phi -= np.pi/2.0
        # only inside FR
        mask = r <= 1.0
        r = r[mask]
        phi = phi[mask]
        s = s[mask]
        
        # get magnetic field along sc trajectory
        b = []
        for i in range(r.size):
            x_, y_, z_, b_ = self.line(
                r[i],
                phi[i],
                [s[i]-1e-5, s[i]+1e-5]
            )
            dr = np.array([
                x_[1]-x_[0],
                y_[1]-y_[0],
                z_[1]-z_[0]
            ])
            dr /= np.linalg.norm(dr)
            b.append(np.insert(dr*np.mean(b_)*self.polarity, 0, np.mean(b_)))

        return np.array(b)

    def evocut1d(self, x, y, z, 
        toroidal_height=0.8):

        self.toroidal_height = 1.0
        self.init()
        self._unit_spline_initial_axis_s_phi = self._spline_initial_axis_s_phi

        th = toroidal_height
        dth = 0.02
        maxth = toroidal_height*4.0
        crossed = False
        b = []
        while th <= maxth:
            self._spline_initial_axis_s_phi = lambda s: self._unit_spline_initial_axis_s_phi(s/th)
            self.toroidal_height = th
            # self.init()
            b_ = self.cut1d(x, y, z)
            if crossed == True and b_.size == 0:
                break
            elif b_.size > 0:
                b.append(b_.ravel())
                crossed = True
            th += dth
        return np.array(b)

    def fit2insitu(self, 
        t, b, bx, by, bz,
        latitude=[-np.pi/180.0*15.0, np.pi/180.0*15.0], 
        longitude=[-np.pi/180.0*30.0, np.pi/180.0*0.0], 
        toroidal_height=0.7, 
        poloidal_height=[0.1, 0.3], 
        half_width=np.pi/180.0*40.0, 
        tilt=[-np.pi/180.0*15.0, np.pi/180.0*15.0], 
        flattening=[0.4, 0.6], 
        pancaking=np.pi/180.0*20.0, 
        skew=np.pi/180.0*0.0, 
        twist=[1.0, 10.0], 
        flux=1e13,
        sigma=2.05,
        polarity=-1.0,
        chirality=1.0):

        resample = True

        t = np.array([time.mktime(x.timetuple()) for x in t])

        if resample:
            n = 300
            t0 = t[0]+(t[-1]-t[0])*np.linspace(0.0, 1.0, n)
            b0 = np.interp(t0, t, b)
            bx0 = np.interp(t0, t, bx)
            by0 = np.interp(t0, t, by)
            bz0 = np.interp(t0, t, bz)
        else:
            t0 = t
            b0 = b
            bx0 = bx
            by0 = by
            bz0 = bz

        self.toroidal_height = toroidal_height
        self.half_width = half_width
        self.pancaking = pancaking
        self.skew = skew
        self.flux = flux
        self.sigma = sigma
        self.polarity = polarity
        self.chirality = chirality

        b0_mean = np.mean(b0)

        db_prev = np.inf

        def F(x):
            global db_prev
            self.latitude = x[0]
            self.longitude = x[1]
            self.poloidal_height = x[2]
            self.tilt = x[3]
            self.flattening = x[4]
            self.twist = x[5]
            # self.sigma = x[6]
            
            b_ = self.evocut1d(1.0, 0.0, 0.0, 
                toroidal_height=toroidal_height
            )
            
            if b_.size > 0:

                t = t0[-1]-(t0[-1]-t0[0])*x[6]*np.linspace(1.0, 0.0, b_.shape[0])
                # t = t0[0]+(t0[-1]-t0[0])*x[6]*np.linspace(0.0, 1.0, b_.shape[0])
                b = b_[:,0]
                bx = b_[:,1]
                by = b_[:,2]
                bz = b_[:,3]
                t1 = t0
                b1 = np.interp(t1, t, b)
                bx1 = np.interp(t1, t, bx)
                by1 = np.interp(t1, t, by)
                bz1 = np.interp(t1, t, bz)

                b1_mean = np.mean(b1)

                coeff = b0_mean/b1_mean
                b1 *= coeff
                bx1 *= coeff
                by1 *= coeff
                bz1 *= coeff
                
                db = np.mean([euclidean(
                    [bx0[i], by0[i], bz0[i]],
                    [bx1[i], by1[i], bz1[i]]
                ) for i in range(t0.size)])

                # db, _ = fastdtw(
                #     np.stack([bx, by, bz], axis=-1),
                #     np.stack([bbx, bby, bbz], axis=-1),
                #     dist=euclidean
                # )
                # db, _ = fastdtw(
                #     np.stack([t, b], axis=-1),
                #     np.stack([tt, bb], axis=-1),
                #     dist=euclidean
                # )
                # dbx, _ = fastdtw(
                #     np.stack([t, bx], axis=-1),
                #     np.stack([tt, bbx], axis=-1),
                #     dist=euclidean
                # )
                # dby, _ = fastdtw(
                #     np.stack([t, by], axis=-1),
                #     np.stack([tt, bby], axis=-1),
                #     dist=euclidean
                # )
                # dbz, _ = fastdtw(
                #     np.stack([t, bz], axis=-1),
                #     np.stack([tt, bbz], axis=-1),
                #     dist=euclidean
                # )
                # d = np.amax([dbx, dby, dbz])
                if db < db_prev:
                    db_prev = db
                    x[0] *= 180.0/np.pi
                    x[1] *= 180.0/np.pi
                    x[3] *= 180.0/np.pi
                    print(x)
                    print(db)
                # plt.plot(t, b, tt, bb)
                # plt.plot(t, bx, tt, bbx)
                # plt.plot(t, by, tt, bby)
                # plt.plot(t, bz, tt, bbz)
                # plt.show()
                return db
            else:
                return np.inf

        res = scipy.optimize.differential_evolution(
            F,
            bounds=[
                (latitude[0], latitude[1]), 
                (longitude[0], longitude[1]), 
                (poloidal_height[0], poloidal_height[1]),
                (tilt[0], tilt[1]), 
                (flattening[0], flattening[1]),
                (twist[0], twist[1]),
                (0.7, 1.0)
            ],
        )

        # res = scipy.optimize.minimize(
        #     F,
        #     np.array([
        #         np.mean([latitude[0], latitude[1]]), 
        #         np.mean([longitude[0], longitude[1]]), 
        #         np.mean([poloidal_height[0], poloidal_height[1]]), 
        #         np.mean([half_width[0], half_width[1]]), 
        #         np.mean([tilt[0], tilt[1]]), 
        #         np.mean([flattening[0], flattening[1]]), 
        #         np.mean([twist[0], twist[1]])
        #     ]), 
        #     bounds=[
        #         (latitude[0], latitude[1]), 
        #         (longitude[0], longitude[1]), 
        #         (poloidal_height[0], poloidal_height[1]),
        #         (half_width[0], half_width[1]), 
        #         (tilt[0], tilt[1]), 
        #         (flattening[0], flattening[1]),
        #         (twist[0], twist[1])
        #     ],
        #     method='L-BFGS-B'
        # )
        print(res.x)

    def fit2remote(self):
        x0, y0, z0 = self.shell()
        fig = plt.figure()

        DB = 1.079771
        DA = 0.960188
        DS = 1.0
        RB = DB*AU_RS*np.tan(4.0*np.pi/180.0)
        RA = DA*AU_RS*np.tan(4.0*np.pi/180.0)
        RS = 32.0

        gs = gridspec.GridSpec(2, 3)
        gs.update(wspace=0.0, hspace=0.0)

        ax = plt.subplot(gs[0])
        ax.imshow(
            plt.imread('/media/data/Documents/Articles/2016_Isavnin_FRi3D/20130106_103900_dbc2B_opt.png'),
            zorder=0,
            extent=[-RB+0.05, RB+0.05, -RB-0.1, RB-0.1]
        )
        ax.set_xlim([-RB+0.05, RB+0.05])
        ax.set_ylim([-RB-0.1, RB-0.1])
        ax.set_axis_bgcolor('black')
        plt.axis('off')

        ax = plt.subplot(gs[1])
        ax.imshow(
            plt.imread('/media/data/Documents/Articles/2016_Isavnin_FRi3D/20130106_1042_c3_1024_opt.png'),
            zorder=0,
            extent=[-32.0+0.3, 32.0+0.3, -32.0+1.33, 32.0+1.33]
        )
        ax.set_xlim([-32.0+0.3, 32.0+0.3])
        ax.set_ylim([-32.0+1.33, 32.0+1.33])
        ax.set_axis_bgcolor('black')
        plt.axis('off')

        ax = plt.subplot(gs[2])
        ax.imshow(
            plt.imread('/media/data/Documents/Articles/2016_Isavnin_FRi3D/20130106_103900_dbc2A_opt.png'),
            zorder=0,
            extent=[-RA, RA, -RA+0.04, RA+0.04]
        )
        ax.set_xlim([-RA, RA])
        ax.set_ylim([-RA+0.04, RA+0.04])
        ax.set_axis_bgcolor('black')
        plt.axis('off')

        
        ax = plt.subplot(gs[3])
        ax.imshow(
            plt.imread('/media/data/Documents/Articles/2016_Isavnin_FRi3D/20130106_103900_dbc2B_opt.png'),
            zorder=0,
            extent=[-RB-0.03, RB-0.03, -RB-0.0, RB-0.0]
        )
        # ax.plot([0.0], [0.0], '.y', markersize=5.0)
        T = cs.mx_rot_z(-np.pi/180.0*(128.919+132.550))
        x = T[0,0]*x0+T[0,1]*y0+T[0,2]*z0
        y = T[1,0]*x0+T[1,1]*y0+T[1,2]*z0
        z = T[2,0]*x0+T[2,1]*y0+T[2,2]*z0
        y = DB/(DB-x)*y
        z = DB/(DB-x)*z
        ax.scatter(y*AU_RS, z*AU_RS, 3, color=BLIND_PALETTE['yellow'], marker='.')
        ax.set_xlim([-RB-0.03, RB-0.03])
        ax.set_ylim([-RB-0.0, RB-0.0])
        ax.set_axis_bgcolor('black')
        # ax.text(-RA+1, -RA+1, 
        #     'COR2B 2013-01-06 10:39', 
        #     fontsize=24,
        #     color=BLIND_PALETTE['yellow']
        # )
        plt.axis('off')

        ax = plt.subplot(gs[4])
        ax.imshow(
            plt.imread('/media/data/Documents/Articles/2016_Isavnin_FRi3D/20130106_1042_c3_1024_opt.png'),
            zorder=0,
            extent=[-32.0+0.37, 32.0+0.37, -32.0+1.19, 32.0+1.19]
        )
        # ax.plot([0.0], [0.0], '.y', markersize=5.0)
        T = cs.mx_rot_z(-np.pi/180.0*128.919)
        x = T[0,0]*x0+T[0,1]*y0+T[0,2]*z0
        y = T[1,0]*x0+T[1,1]*y0+T[1,2]*z0
        z = T[2,0]*x0+T[2,1]*y0+T[2,2]*z0
        y = DS/(DS-x)*y
        z = DS/(DS-x)*z
        ax.scatter(y*AU_RS, z*AU_RS, 3, color=BLIND_PALETTE['yellow'], marker='.')
        ax.set_xlim([-32.0+0.37, 32.0+0.37])
        ax.set_ylim([-32.0+1.19, 32.0+1.19])
        ax.set_axis_bgcolor('black')
        # ax.text(-RA+1, -RA+1, 
        #     'C3 2013-01-06 10:42', 
        #     fontsize=24,
        #     color=BLIND_PALETTE['yellow']
        # )
        plt.axis('off')

        
        ax = plt.subplot(gs[5])
        ax.imshow(
            plt.imread('/media/data/Documents/Articles/2016_Isavnin_FRi3D/20130106_103900_dbc2A_opt.png'),
            zorder=0,
            extent=[-RA+0.1, RA+0.1, -RA+0.04, RA+0.04]
        )
        # ax.plot([0.0], [0.0], '.y', markersize=5.0)
        y = DA/(DA-x0)*y0
        z = DA/(DA-x0)*z0
        ax.scatter(y*AU_RS, z*AU_RS, 3, color=BLIND_PALETTE['yellow'], marker='.')
        ax.set_xlim([-RA+0.1, RA+0.1])
        ax.set_ylim([-RA+0.04, RA+0.04])
        ax.set_axis_bgcolor('black')
        ax.patch.set_facecolor('black')
        # ax.text(-RA+1, -RA+1, 
        #     'COR2A 2013-01-06 10:39', 
        #     fontsize=24,
        #     color=BLIND_PALETTE['yellow']
        # )

        plt.axis('off')

        plt.show()

