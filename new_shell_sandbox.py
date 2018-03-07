import numpy as np
from matplotlib import pyplot as plt

def axis_r(
        axis_phi,
        toroidal_height,
        half_width,
        flattening):
    return toroidal_height*np.cos(np.pi/2/half_width*axis_phi)**flattening

def axis_dr_dphi(
        axis_phi,
        toroidal_height,
        half_width,
        flattening):
    return (
        -np.pi/2/half_width*flattening
        *np.tan(np.pi/2/half_width*axis_phi)
        *axis_r(axis_phi, toroidal_height, half_width, flattening)
    )

def axis_d2r_dphi2(
        axis_phi,
        toroidal_height,
        half_width,
        flattening):
    return (
        -np.pi/2/half_width*flattening
        *np.tan(np.pi/2/half_width*axis_phi)
        *axis_dr_dphi(axis_phi, toroidal_height, half_width, flattening)
        -(np.pi/2/half_width)**2*flattening
        /np.cos(np.pi/2/half_width*axis_phi)**2
        *axis_r(axis_phi, toroidal_height, half_width, flattening)
    )

def axis_curvature_radius(
        axis_phi,
        toroidal_height,
        half_width,
        flattening):
    return (
        (
            axis_r(axis_phi, toroidal_height, half_width, flattening)**2
            +axis_dr_dphi(axis_phi, toroidal_height, half_width, flattening)**2
        )**(3/2)
        /np.abs(
            axis_r(axis_phi, toroidal_height, half_width, flattening)**2
            +2*axis_dr_dphi(axis_phi, toroidal_height, half_width, flattening)**2
            -axis_r(axis_phi, toroidal_height, half_width, flattening)
            *axis_d2r_dphi2(axis_phi, toroidal_height, half_width, flattening)
        )
    )

def cs_r(
        axis_phi,
        toroidal_height,
        poloidal_height,
        half_width,
        flattening):
    return poloidal_height/toroidal_height*axis_r(
        axis_phi,
        toroidal_height,
        half_width,
        flattening
    )

def cs_dr_dlength(
        axis_phi,
        toroidal_height,
        poloidal_height,
        half_width,
        flattening):
    return (
        poloidal_height/toroidal_height
        *axis_dr_dphi(axis_phi, toroidal_height, half_width, flattening)
        /np.sqrt(
            axis_r(axis_phi, toroidal_height, half_width, flattening)**2
            +axis_dr_dphi(axis_phi, toroidal_height, half_width, flattening)**2
        )
    )

def axis_ds_dphi(
        axis_phi,
        toroidal_height,
        half_width,
        flattening):
    return np.sqrt(
        axis_r(axis_phi, toroidal_height, half_width, flattening)**2
        +axis_dr_dphi(axis_phi, toroidal_height, half_width, flattening)
    )

def plt_axis():
    ax = plt.subplot(111, projection='polar')
    toroidal_height = 1
    half_width = np.pi/3
    flattening = 0.5
    axis_phi = np.linspace(-half_width, half_width, 100)
    ax.plot(
        axis_phi,
        axis_r(axis_phi, toroidal_height, half_width, flattening)
    )
    ax.set_rmax(1.1)
    ax.grid(True)
    plt.show()

def plt_axis_opened():
    toroidal_height = 1
    poloidal_height = 0.2
    half_width = np.pi/4
    flattening = 0.5
    phi = np.linspace(-half_width, half_width, 100)
    r = axis_r(phi, toroidal_height, half_width, flattening)
    plt.plot(phi*toroidal_height, r)
    plt.plot(
        phi*toroidal_height
        *(half_width*toroidal_height+poloidal_height)
        /(half_width*toroidal_height),
        r*(toroidal_height+poloidal_height)/toroidal_height
    )
    plt.axis('equal')
    plt.show()

def flat_axis(x, toroidal_height, half_width, flattening):
    return toroidal_height*np.cos(np.pi/2/toroidal_height*x)**flattening



# def shell(
#         s=np.linspace(0, 1, 50),
#         phi=np.linspace(0, np.pi*2, 24)):
#     s_mesh, phi_mesh = np.meshgrid(s, phi)
#     z = s*self.vanilla_axis_length(self.half_width)

# from sympy import *
# phi, y0, a, n = symbols('phi y0 a n')
# print(
#     sqrt(
#         (
#             diff(
#                 sqrt(
#                     (cos(a*phi)**n*cos(phi))**2
#                     +(cos(a*phi)**n*sin(a*phi)-y0)**2
#                 ),
#                 phi
#             )
#             /diff(
#                 atan(
#                     (cos(a*phi)**n*sin(phi)-y0)
#                     /(cos(a*phi)**n*cos(phi))
#                 ),
#                 phi
#             )
#         )**2
#         +(
#             (cos(a*phi)**n*cos(phi))**2
#             +(cos(a*phi)**n*sin(a*phi)-y0)**2
#         )
#     )
#     *diff(
#         atan(
#             (cos(a*phi)**n*sin(phi)-y0)
#             /(cos(a*phi)**n*cos(phi))
#         ),
#         phi
#     )
# )

def length_integral(phi, a, n, y0):
    return(
        (
            a*n*(-y0+np.sin(phi)*np.cos(a*phi)**n)
            *np.sin(a*phi)*np.cos(a*phi)**(-n)
            /(np.cos(phi)*np.cos(a*phi))
            +(-y0+np.sin(phi)*np.cos(a*phi)**n)
            *np.sin(phi)*np.cos(a*phi)**(-n)/np.cos(phi)**2
            +(
                -a*n*np.sin(phi)*np.sin(a*phi)*np.cos(a*phi)**n/np.cos(a*phi)
                +np.cos(phi)*np.cos(a*phi)**n
            )
            *np.cos(a*phi)**(-n)/np.cos(phi)
        )
        *np.sqrt(
            (-y0+np.sin(a*phi)*np.cos(a*phi)**n)**2
            +(
                (-y0+np.sin(phi)*np.cos(a*phi)**n)**2
                *np.cos(a*phi)**(-2*n)/np.cos(phi)**2+1
            )**2
            *(
                -a*n*np.sin(a*phi)*np.cos(phi)**2
                *np.cos(a*phi)**(2*n)/np.cos(a*phi)
                +(-y0+np.sin(a*phi)*np.cos(a*phi)**n)
                *(
                    -2*a*n*np.sin(a*phi)**2*np.cos(a*phi)**n/np.cos(a*phi)
                    +2*a*np.cos(a*phi)*np.cos(a*phi)**n
                )/2
                -np.sin(phi)*np.cos(phi)*np.cos(a*phi)**(2*n)
            )**2
            /(
                (
                    (-y0+np.sin(a*phi)*np.cos(a*phi)**n)**2
                    +np.cos(phi)**2*np.cos(a*phi)**(2*n)
                )
                *(
                    a*n*(-y0+np.sin(phi)*np.cos(a*phi)**n)
                    *np.sin(a*phi)*np.cos(a*phi)**(-n)
                    /(np.cos(phi)*np.cos(a*phi))
                    +(-y0+np.sin(phi)*np.cos(a*phi)**n)
                    *np.sin(phi)*np.cos(a*phi)**(-n)/np.cos(phi)**2
                    +(
                        -a*n*np.sin(phi)*np.sin(a*phi)
                        *np.cos(a*phi)**n/np.cos(a*phi)
                        +np.cos(phi)*np.cos(a*phi)**n
                    )*np.cos(a*phi)**(-n)/np.cos(phi)
                )**2
            )
            +np.cos(phi)**2*np.cos(a*phi)**(2*n)
        )
        /(
            (-y0+np.sin(phi)*np.cos(a*phi)**n)**2
            *np.cos(a*phi)**(-2*n)/np.cos(phi)**2+1
        )
    )

from scipy.integrate import fixed_quad, quad

toroidal_height = 1
poloidal_height = 0.1
half_width = np.pi/3
coeff_angle = np.pi/2/half_width
flattening = 0.2

# from ai.fri3d.model import StaticFRi3D

# sfr = StaticFRi3D(
#     toroidal_height=toroidal_height,
#     poloidal_height=poloidal_height,
#     half_width=half_width,
#     flattening=flattening
# )

# half_width = np.pi/10
# res = fixed_quad(
#     lambda phi: length_integral(phi, coeff_angle, flattening, 0.),
#     -half_width,
#     0,
#     n=10
# )
# print(res[0]/np.pi)

# print(sfr.vanilla_axis_length(0)/np.pi)

# res = quad(
#     lambda phi: np.sqrt(
#         axis_r(phi, toroidal_height, half_width, flattening)**2
#         +axis_dr_dphi(phi, toroidal_height, half_width, flattening)**2
#     ),
#     -half_width,
#     0,
#     # n=5,
# )
# print(res[0]/np.pi)

from numba import cfunc

def ds(phi):
    return np.sqrt(
        (toroidal_height*np.cos(np.pi/2/half_width*phi)**flattening)**2
        +(
            -np.pi/2/half_width*flattening
            *np.tan(np.pi/2/half_width*phi)
            *toroidal_height*np.cos(np.pi/2/half_width*phi)**flattening
        )**2
    )

nb_ds = cfunc("float64(float64)")(ds)

res = quad(
    nb_ds.ctypes,
    -half_width,
    0
)
print(res[0])

from scipy.optimize import brenth

res = brenth(
    lambda phi: quad(
        nb_ds.ctypes,
        -half_width,
        phi
    )[0]-1,
    -half_width,
    half_width
)
print(res)

from timeit import Timer



# t = Timer(
#     lambda: quad(
#         lambda phi: np.sqrt(
#             axis_r(phi, toroidal_height, half_width, flattening)**2
#             +axis_dr_dphi(phi, toroidal_height, half_width, flattening)**2
#         ),
#         -half_width,
#         -half_width*0.8
#     )[0]/np.pi
# )
# print(t.timeit(number=1000))

# t = Timer(
#     lambda: fixed_quad(
#         lambda phi: np.sqrt(
#             axis_r(phi, toroidal_height, half_width, flattening)**2
#             +axis_dr_dphi(phi, toroidal_height, half_width, flattening)**2
#         ),
#         -half_width,
#         -half_width*0.8,
#         n=1000
#     )[0]/np.pi
# )
# print(t.timeit(number=1000))

# t = Timer(
#     lambda: sfr.vanilla_axis_length(-half_width*0.8)/np.pi
# )
# print(t.timeit(number=1000))

t = Timer(
    lambda: quad(
        nb_ds.ctypes,
        -half_width,
        half_width
    )
)
print(t.timeit(number=1000))

t = Timer(
    lambda: brenth(
        lambda phi: quad(
            nb_ds.ctypes,
            -half_width,
            phi
        )[0]-1.5,
        -half_width,
        half_width
    )
)
print(t.timeit(number=1000))


# from pynverse import inversefunc

# vquad = np.vectorize(
#     lambda phi: quad(
#         nb_ds.ctypes,
#         -half_width,
#         phi
#     )[0]
# )

# length2angle = inversefunc(
#     lambda phi: vquad(phi),
#     domain=[-half_width, half_width]
# )

# print(
#     quad(
#         nb_ds.ctypes,
#         -half_width,
#         half_width
#     )[0]
# )

# t = Timer(
#     lambda: length2angle(np.linspace(1, 2.5, 10))
# )
# print(t.timeit(number=1000))

# print(
#     length2angle(0)*180/np.pi
# )


# def flat_axis_length(x, toroidal_height, half_width, flattening):
#     return toroidal_height**2*

# from scipy.special import hyp2f1



# s=np.linspace(0, 1, 50),
# phi=np.linspace(0, np.pi*2, 24)

# z = s*self.vanilla_axis_length(self.half_width)