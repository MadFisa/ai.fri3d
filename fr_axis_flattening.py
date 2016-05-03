from __future__ import division

import numpy
import scikits.bvp_solver
import pylab
import sys



import matplotlib.pyplot as plt

from matplotlib.transforms import Affine2D
from matplotlib.projections import PolarAxes
from mpl_toolkits.axisartist import angle_helper
from mpl_toolkits.axisartist.grid_finder import MaxNLocator
from mpl_toolkits.axisartist.floating_axes import GridHelperCurveLinear, FloatingSubplot

flattening = 0.42

fr_parameter = 1.0
fr_height = 1.0
fr_radius_max = 0.1*fr_height
phi0 = numpy.pi/3.0
n_points = 50

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

def diameter(x, radius_max):
    return 2.0*radius_max*numpy.cos(x/phi0*numpy.pi/2.0)**0.42

def function(x, y, p):
    fr_parameter = p[0]
    # print 2.0*y[1]**2, y[0], 2.0*y[1]**2/y[0]
    return numpy.array([y[1],
                        -fr_parameter*diameter(x, fr_radius_max)*(y[0]**2+y[1]**2)/(0.3 if y[0] < 0.3 else y[0])**4-y[0]-2.0*y[1]**2/(0.3 if y[0] < 0.3 else y[0])])

def boundary_conditions(y_a, y_b, p):
    condition_a = numpy.array([y_a[0]-fr_height, y_a[1]])
    condition_b = numpy.array([y_b[0]])

    return condition_a, condition_b

def guess(x):
    return numpy.array([fr_height*numpy.cos(x/phi0*numpy.pi/2.0)**flattening,
                        -fr_height/phi0*numpy.pi/2.0*numpy.sin(x/phi0*numpy.pi/2.0)])

problem = scikits.bvp_solver.ProblemDefinition(num_ODE = 2,
                                               num_parameters = 1,
                                               num_left_boundary_conditions = 2,
                                               boundary_points = (0.0, phi0),
                                               function = function,
                                               boundary_conditions = boundary_conditions)

phi = numpy.linspace(problem.boundary_points[0], problem.boundary_points[1], n_points)

fig = plt.figure()
axes = fractional_polar_axes(fig, (-90.0, 90.0), (0.0, 1.1), thlabel=r"$\varphi$")
# axes = pylab.subplot(111, polar=True)

flattening = 0.2
guessed_r = map(lambda x: guess(x)[0]/guess(0.0)[0]*1.0, phi)
axes.plot(
    phi*180.0/numpy.pi, 
    guessed_r, 
    color='r', 
    linewidth=2,
    zorder=3
)
guessed_r = map(lambda x: guess(x)[0]/guess(0.0)[0]*1.0, -phi)
axes.plot(
    -phi*180.0/numpy.pi, 
    guessed_r, 
    color='r', 
    linewidth=2, 
    label='n = 0.2',
    zorder=3
)

flattening = 0.4
guessed_r = map(lambda x: guess(x)[0]/guess(0.0)[0]*1.0, phi)
axes.plot(
    phi*180.0/numpy.pi, 
    guessed_r, 
    color='b', 
    linewidth=2,
    linestyle='dashed',
    zorder=2
)
guessed_r = map(lambda x: guess(x)[0]/guess(0.0)[0]*1.0, -phi)
axes.plot(
    -phi*180.0/numpy.pi, 
    guessed_r, 
    color='b', 
    linewidth=2, 
    label='n = 0.4',
    linestyle='dashed',
    zorder=2
)

# flattening = 0.6
# guessed_r = map(lambda x: guess(x)[0]/guess(0.0)[0]*1.0, phi)
# axes.plot(phi*180.0/numpy.pi, guessed_r, color='r', linewidth=2,
#     zorder=2)
# guessed_r = map(lambda x: guess(x)[0]/guess(0.0)[0]*1.0, -phi)
# axes.plot(-phi*180.0/numpy.pi, guessed_r, color='r', linewidth=2, label='',
#     zorder=2)

flattening = 0.8
guessed_r = map(lambda x: guess(x)[0]/guess(0.0)[0]*1.0, phi)
axes.plot(
    phi*180.0/numpy.pi, 
    guessed_r, 
    color='g', 
    linewidth=2,
    linestyle=':',
    zorder=1
)
guessed_r = map(lambda x: guess(x)[0]/guess(0.0)[0]*1.0, -phi)
axes.plot(
    -phi*180.0/numpy.pi, 
    guessed_r, 
    color='g', 
    linewidth=2, 
    label='n = 0.8',
    linestyle=':',
    zorder=1
)

# axes.set_rmax(fr_height+0.1)
# axes.set_xlim(-numpy.pi/2.0,numpy.pi/2.0)

axes.grid(True)

pylab.legend(loc='upper center')

pylab.show()





