import numpy
import scikits.bvp_solver
import pylab
import sys

fr_parameter = 1.0
fr_height = 1.0
fr_radius_max = 0.1*fr_height
phi0 = numpy.pi/3.0
n_points = 100

def diameter(x, radius_max):
    return 2.0*radius_max*numpy.cos(x/phi0*numpy.pi/2.0)

def function(x, y, p):
    fr_parameter = p[0]
    # print 2.0*y[1]**2, y[0], 2.0*y[1]**2/y[0]
    return numpy.array([y[1],
                        -fr_parameter*diameter(x, fr_radius_max)*(y[0]**2+y[1]**2)-y[0]-2.0*y[1]**2/(0.3 if y[0] < 0.3 else y[0])])

def boundary_conditions(y_a, y_b, p):
    condition_a = numpy.array([y_a[0]-fr_height, y_a[1]])
    condition_b = numpy.array([y_b[0]])

    return condition_a, condition_b

def guess(x):
    return numpy.array([fr_height*numpy.cos(x/phi0*numpy.pi/2.0)**0.25,
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

print solution.parameters

axes = pylab.subplot(111, polar=True)
#axes.plot(solution.mesh, solution.solution[0,:]/solution.solution[0,0]*1.0, color='r', linewidth=3)
#axes.plot(-solution.mesh, solution.solution[0,:]/solution.solution[0,0]*1.0, color='r', linewidth=3)
guessed_r = map(lambda x: guess(x)[0]/guess(0.0)[0]*1.0, phi)
axes.plot(phi, guessed_r, color='g', linewidth=3)
guessed_r = map(lambda x: guess(x)[0]/guess(0.0)[0]*1.0, -phi)
axes.plot(-phi, guessed_r, color='g', linewidth=3)
axes.set_rmax(fr_height)
axes.grid(True)

pylab.show()
