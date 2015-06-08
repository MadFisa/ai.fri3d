import numpy
import pylab
from mpl_toolkits.mplot3d import Axes3D

def fr_axis(phi, a, n, Rt):
    return Rt*numpy.cos(a*phi)**n

def fr_tangent(phi, a, n):
    return numpy.arctan2(1.0, a*n*numpy.tan(a*phi))

def fr_x(phi, theta, a, n, Rt, Rp):
    return fr_axis(phi, a, n, Rt)*(Rp/Rt*numpy.cos(theta)*numpy.sin(fr_tangent(phi, a, n)-phi)+numpy.cos(phi))

def fr_y(phi, theta, a, n, Rt, Rp):
    return fr_axis(phi, a, n, Rt)*(Rp/Rt*numpy.cos(theta)*numpy.cos(fr_tangent(phi, a, n)-phi)+numpy.sin(phi))

def fr_z(phi, theta, a, n, Rt, Rp):
    return fr_axis(phi, a, n, Rt)*Rp/Rt*numpy.sin(theta)

phi0 = numpy.pi/6.0
Rt = 1.0
Rp = 0.15
n = 0.4
a = numpy.pi/2.0/phi0


n_phi = 80
n_theta = 60

phi, theta = numpy.meshgrid(numpy.linspace(-phi0, phi0, n_phi), numpy.linspace(0.0, 2.0*numpy.pi, n_theta))

x = fr_x(phi, theta, a, n, Rt, Rp)
y = fr_y(phi, theta, a, n, Rt, Rp)
z = fr_z(phi, theta, a, n, Rt, Rp)

# twist deformation
Ra = 0.05
b = 0.6

r = numpy.sqrt(x**2+y**2+z**2)
theta = numpy.arcsin(z/r)
phi = numpy.arctan2(y, x)
phi = phi+b*(r-Ra).clip(min=0)

print b*(Rt-Ra)*180.0/numpy.pi

x = r*numpy.cos(theta)*numpy.cos(phi)
y = r*numpy.cos(theta)*numpy.sin(phi)
z = r*numpy.sin(theta)


axes = pylab.subplot(111, projection='3d', adjustable='box', aspect=1.0)
axes.plot_wireframe(x, y, z)

max_range = numpy.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max()/2.0
mean_x = x.mean()
mean_y = y.mean()
mean_z = z.mean()
axes.set_xlim(0.0, mean_x+max_range)
axes.set_ylim(mean_y-max_range, mean_y+max_range)
axes.set_zlim(mean_z-max_range, mean_z+max_range)

axes.contour(x, y, z, zdir='x', offset = 0.0)
axes.contour(x, y, z, zdir='y', offset = mean_y+max_range)
axes.contour(x, y, z, zdir='z', offset = mean_z-max_range)

axes.view_init(elev = 35.0, azim = -45.0)

pylab.show()
