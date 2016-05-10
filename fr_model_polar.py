import numpy
import pylab
from matplotlib import pyplot as plt

from matplotlib.transforms import Affine2D
from matplotlib.projections import PolarAxes
from mpl_toolkits.axisartist import angle_helper
from mpl_toolkits.axisartist.grid_finder import MaxNLocator
from mpl_toolkits.axisartist.floating_axes import GridHelperCurveLinear, FloatingSubplot

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

def fractional_polar_axes(f, thlim=(0, 180), rlim=(0, 1), step=(30, 0.2),
    thlabel='theta', rlabel='r', ticklabels=True):
    """Return polar axes that adhere to desired theta (in deg) and r limits. steps for theta
    and r are really just hints for the locators. Using negative values for rlim causes
    problems for GridHelperCurveLinear for some reason"""
    th0, th1 = thlim # deg
    r0, r1 = rlim
    thstep, rstep = step

    # scale degrees to radians:
    tr_scale = Affine2D().scale(numpy.pi/180., 1.)
    tr = tr_scale + PolarAxes.PolarTransform()
    theta_grid_locator = angle_helper.LocatorDMS((th1-th0) // thstep)
    r_grid_locator = MaxNLocator((r1-r0) // rstep)
    theta_tick_formatter = angle_helper.FormatterDMS()
    grid_helper = GridHelperCurveLinear(tr,
                                        extremes=(th0, th1, r0, r1),
                                        grid_locator1=theta_grid_locator,
                                        grid_locator2=r_grid_locator,
                                        tick_formatter1=theta_tick_formatter,
                                        tick_formatter2=None)

    a = FloatingSubplot(f, 111, grid_helper=grid_helper)
    f.add_subplot(a)

    # adjust x axis (theta):
    a.axis["bottom"].set_visible(False)
    a.axis["top"].set_axis_direction("bottom") # tick direction
    a.axis["top"].toggle(ticklabels=ticklabels, label=bool(thlabel))
    a.axis["top"].major_ticklabels.set_axis_direction("top")
    a.axis["top"].label.set_axis_direction("top")

    # adjust y axis (r):
    a.axis["left"].set_axis_direction("bottom") # tick direction
    a.axis["right"].set_axis_direction("top") # tick direction
    a.axis["left"].toggle(ticklabels=ticklabels, label=bool(rlabel))

    # add labels:
    a.axis["top"].label.set_text(thlabel)
    a.axis["left"].label.set_text(rlabel)

    # create a parasite axes whose transData is theta, r:
    auxa = a.get_aux_axes(tr)
    # make aux_ax to have a clip path as in a?:
    auxa.patch = a.patch 
    # this has a side effect that the patch is drawn twice, and possibly over some other
    # artists. So, we decrease the zorder a bit to prevent this:
    a.patch.zorder = -2

    # add sector lines for both dimensions:
    thticks = grid_helper.grid_info['lon_info'][0]
    rticks = grid_helper.grid_info['lat_info'][0]
    for th in thticks[1:-1]: # all but the first and last
        auxa.plot([th, th], [r0, r1], '--', c='grey', zorder=-1)
    for ri, r in enumerate(rticks):
        # plot first r line as axes border in solid black only if it isn't at r=0
        if ri == 0 and r != 0:
            ls, lw, color = 'solid', 2, 'black'
        else:
            ls, lw, color = 'dashed', 1, 'grey'
        # From http://stackoverflow.com/a/19828753/2020363
        auxa.add_artist(plt.Circle([0, 0], radius=r, ls=ls, lw=lw, color=color, fill=False,
                        transform=auxa.transData._b, zorder=-1))
    return auxa

def fr_model(phi, phi0, Rt, n):
    return Rt*numpy.cos(phi/phi0*numpy.pi/2.0)**n

def fr_model_dif(phi, phi0, Rt, n):
    return -numpy.pi/2.0/phi0*n*Rt*numpy.cos(phi/phi0*numpy.pi/2.0)**(n-1)*numpy.sin(phi/phi0*numpy.pi/2.0)

def fr_tangent(phi, phi0, n):
    return numpy.arctan(-1.0/(numpy.pi/2.0/phi0*n*numpy.tan(phi/phi0*numpy.pi/2.0)))

def fr_model_outer(phi, phi0, Rt, n, Rp):
    return numpy.sqrt((fr_model(phi, phi0, Rt, n)*(1.0+Rp/Rt*numpy.abs(numpy.sin(fr_tangent(phi, phi0, n)))))**2.0+(Rp/Rt*fr_model(phi, phi0, Rt, n)*numpy.cos(fr_tangent(phi, phi0, n)))**2.0)

def fr_model_inner(phi, phi0, Rt, n, Rp):
    return numpy.sqrt((fr_model(phi, phi0, Rt, n)*(1.0-Rp/Rt*numpy.abs(numpy.sin(fr_tangent(phi, phi0, n)))))**2.0+(Rp/Rt*fr_model(phi, phi0, Rt, n)*numpy.cos(fr_tangent(phi, phi0, n)))**2.0)

def fr_dphi(phi, phi0, Rt, n, Rp):
    return numpy.arctan(numpy.cos(fr_tangent(phi, phi0, n))*Rp/Rt)

phi0 = numpy.pi/4.0
phi = numpy.linspace(-phi0, phi0, 100)
Rt = 1.0
Rp = 0.15
n = 0.32

fig = plt.figure(figsize=(9,9))

axes = fractional_polar_axes(fig, (-90.0, 90.0), (0.0, 1.2), thlabel=r"$\varphi$")
r = map(lambda x: fr_model(x, phi0, Rt, n), phi)
axes.plot(
    phi*180.0/numpy.pi, 
    r, 
    color=BLIND_PALETTE['bluish-green'], 
    linewidth=3,
    linestyle='--',
    zorder=1
)
r1 = numpy.array(map(lambda x: fr_model_outer(x, phi0, Rt, n, Rp), phi))
phi1 = numpy.array(map(lambda x: x+numpy.sign(x)*fr_dphi(x, phi0, Rt, n, Rp), phi))
axes.plot(
    phi1*180.0/numpy.pi, 
    r1, 
    color=BLIND_PALETTE['vermillion'], 
    linewidth=3,
    zorder=2
)
r2 = numpy.array(map(lambda x: fr_model_inner(x, phi0, Rt, n, Rp), phi))
phi2 = numpy.array(map(lambda x: x-numpy.sign(x)*fr_dphi(x, phi0, Rt, n, Rp), phi))
axes.plot(
    phi2*180.0/numpy.pi, 
    r2, 
    color=BLIND_PALETTE['vermillion'], 
    linewidth=3,
    zorder=2
)
# axes.set_rmax(Rt+Rp)
# axes.grid(True)

plt.show()
