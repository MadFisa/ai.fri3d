import numpy
import pylab

def fr_axis(phi, a, n, Rt):
    return Rt*numpy.cos(a*phi)**n

def fr_dphi(phi, a, n, Rt, Ra, b):
    return b*(fr_axis(phi, a, n, Rt)-Ra).clip(min=0)

phi0 = numpy.pi/3.0
a = numpy.pi/2.0/phi0
phi = numpy.linspace(-phi0, phi0, 10000)
Rt = 1.0
n = 0.3
Ra = 0.05
b = 1.1

print b*(Rt-Ra)*180.0/numpy.pi

dphi = fr_dphi(phi, a, n, Rt, Ra, b)

axes = pylab.subplot(111, polar=True)
r = map(lambda x: fr_axis(x, a, n, Rt), phi)
axes.plot(phi+dphi, r, color='g', linewidth=3)
axes.set_rmax(Rt)
axes.grid(True)

pylab.show()
