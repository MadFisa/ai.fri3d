
import numpy

def cart2sp(x, y, z):
    r = numpy.sqrt(x**2+y**2+z**2)
    theta = numpy.arcsin(z/r)
    phi = numpy.arctan2(y, x)
    return (r, theta, phi)

def sp2cart(r, theta, phi):
    x = r*numpy.cos(theta)*numpy.cos(phi)
    y = r*numpy.cos(theta)*numpy.sin(phi)
    z = r*numpy.sin(theta)
    return (x, y, z)

def cart2cyl(x, y, z):
    r = numpy.sqrt(x**2+y**2)
    phi = numpy.arctan2(y, x)
    return (r, phi, z)

def cyl2cart(r, phi, z):
    x = r*numpy.cos(phi)
    y = r*numpy.sin(phi)
    return (x, y, z)
