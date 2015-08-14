
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


def mx_rot_x(gamma):
    return numpy.matrix([[1.0, 0.0, 0.0], 
                         [0.0, numpy.cos(gamma), numpy.sin(gamma)], 
                         [0.0, -numpy.sin(gamma), numpy.cos(gamma)]])

def mx_rot_y(theta):
    return numpy.matrix([[numpy.cos(theta), 0.0, -numpy.sin(theta)],
                         [0.0, 1.0, 0.0],
                         [numpy.sin(theta), 0.0, numpy.cos(theta)]])

def mx_rot_z(phi):
    return numpy.matrix([[numpy.cos(phi), numpy.sin(phi), 0.0],
                         [-numpy.sin(phi), numpy.cos(phi), 0.0],
                         [0.0, 0.0, 1.0]])

def mx_rot(theta, phi, gamma):
        return numpy.dot(mx_rot_z(phi), 
                         numpy.dot(mx_rot_y(theta), 
                                   mx_rot_x(gamma)))
