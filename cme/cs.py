
import numpy as np

def cart2sp(x, y, z):
    r = np.sqrt(x**2+y**2+z**2)
    theta = np.arcsin(z/r)
    phi = np.arctan2(y, x)
    return (r, theta, phi)

def sp2cart(r, theta, phi):
    x = r*np.cos(theta)*np.cos(phi)
    y = r*np.cos(theta)*np.sin(phi)
    z = r*np.sin(theta)
    return (x, y, z)

def cart2cyl(x, y, z):
    r = np.sqrt(x**2+y**2)
    phi = np.arctan2(y, x)
    return (r, phi, z)

def cyl2cart(r, phi, z):
    x = r*np.cos(phi)
    y = r*np.sin(phi)
    return (x, y, z)


def mx_rot_x(gamma):
    return np.matrix([
        [1.0, 0.0, 0.0], 
        [0.0, np.cos(gamma), np.sin(gamma)], 
        [0.0, -np.sin(gamma), np.cos(gamma)]
    ])

def mx_rot_y(theta):
    return np.matrix([
        [np.cos(theta), 0.0, -np.sin(theta)],
        [0.0, 1.0, 0.0],
        [np.sin(theta), 0.0, np.cos(theta)]
    ])

def mx_rot_z(phi):
    return np.matrix([
        [np.cos(phi), np.sin(phi), 0.0],
        [-np.sin(phi), np.cos(phi), 0.0],
        [0.0, 0.0, 1.0]
    ])

def mx_rot(theta, phi, gamma):
    return np.dot(
        mx_rot_z(phi), 
        np.dot(
            mx_rot_y(theta), 
            mx_rot_x(gamma)
        )
    )
