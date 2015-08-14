import numpy
import pylab

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

phi0 = numpy.pi/3.0
phi = numpy.linspace(-phi0, phi0, 100)
Rt = 1.0
Rp = 0.2
n = 0.3

axes = pylab.subplot(111, polar=True)
r = map(lambda x: fr_model(x, phi0, Rt, n), phi)
axes.plot(phi, r, color='g', linewidth=3)
r1 = map(lambda x: fr_model_outer(x, phi0, Rt, n, Rp), phi)
phi1 = map(lambda x: x+numpy.sign(x)*fr_dphi(x, phi0, Rt, n, Rp), phi)
axes.plot(phi1, r1, color='red', linewidth=3)
r2 = map(lambda x: fr_model_inner(x, phi0, Rt, n, Rp), phi)
phi2 = map(lambda x: x-numpy.sign(x)*fr_dphi(x, phi0, Rt, n, Rp), phi)
axes.plot(phi2, r2, color='red', linewidth=3)
axes.set_rmax(Rt+Rp)
axes.grid(True)

pylab.show()
