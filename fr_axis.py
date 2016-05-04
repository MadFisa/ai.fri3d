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

fr_parameter = 1.0
fr_height = 1.0
fr_radius_max = 0.1*fr_height
phi0 = numpy.pi/3.0
n_points = 50

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

flattening = 0.44

def diameter(x, radius_max):
    return 2.0*radius_max*numpy.cos(x/phi0*numpy.pi/2.0)**flattening

def function(x, y, p):
    fr_parameter = p[0]
    # print 2.0*y[1]**2, y[0], 2.0*y[1]**2/y[0]
    return numpy.array([
        y[1],
        fr_parameter*
        # diameter(x, fr_radius_max)*
        (y[0]**2+y[1]**2)/(0.3 if y[0] < 0.3 else y[0])**3-
        y[0]-
        2.0*y[1]**2/(0.3 if y[0] < 0.3 else y[0])
    ])

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

solution = scikits.bvp_solver.solve(problem,
                                    solution_guess = guess,
                                    parameter_guess = numpy.array([1.0]),
                                    initial_mesh = phi,
                                    method = 6,
                                    trace = 2)

print(solution.parameters)

fig = plt.figure(figsize=(5,9))
axes = fractional_polar_axes(fig, (-90.0, 90.0), (0.0, 1.1), thlabel=r"$\varphi$")
# axes = pylab.subplot(111, polar=True)

mask = solution.mesh < 0.995*phi0
axes.plot(
  solution.mesh[mask]*180.0/numpy.pi, 
  (solution.solution[0,:]/solution.solution[0,0]*1.0)[mask], 
  's',
  color=BLIND_PALETTE['vermillion'], 
  linewidth=3,
  markersize=5,
  label='numerical solution',
  zorder=2
)
axes.plot(
  -solution.mesh[mask]*180.0/numpy.pi, 
  (solution.solution[0,:]/solution.solution[0,0]*1.0)[mask], 
  's',
  color=BLIND_PALETTE['vermillion'], 
  linewidth=3,
  markersize=5,
  zorder=2
)

guessed_r = map(lambda x: guess(x)[0]/guess(0.0)[0]*1.0, phi)
axes.plot(
    phi*180.0/numpy.pi, 
    guessed_r, 
    color=BLIND_PALETTE['blue'], 
    linewidth=3,
    zorder=1
)
guessed_r = map(lambda x: guess(x)[0]/guess(0.0)[0]*1.0, -phi)
axes.plot(
    -phi*180.0/numpy.pi, 
    guessed_r, 
    color=BLIND_PALETTE['blue'], 
    linewidth=3, 
    label='approximate\nanalytical solution',
    zorder=1
)

# axes.set_rmax(fr_height+0.1)
# axes.set_xlim(-numpy.pi/2.0,numpy.pi/2.0)

axes.grid(True)

pylab.legend(loc='upper right')

pylab.show()





