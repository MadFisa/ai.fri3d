

import numpy as np
from scipy.integrate import quad, fixed_quad, dblquad
from matplotlib import pyplot as plt
from sympy import pprint, symbols, cos, sin, atan, exp, pi, integrate, Function
from sympy.solvers.solveset import linsolve
from sympy.solvers.pde import pdsolve
from astropy import units as u
from astropy import constants as c

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

def z0(
        axis_phi,
        toroidal_height,
        half_width,
        flattening):
    return (
        axis_r(axis_phi, toroidal_height, half_width, flattening)
        *axis_ds_dphi(axis_phi, toroidal_height, half_width, flattening)
        /axis_dr_dphi(axis_phi, toroidal_height, half_width, flattening)
    )

def Bz(
        r,
        phi,
        axis_phi,
        toroidal_height,
        half_width,
        flattening,
        twist,
        B0=1):
    Rc = axis_curvature_radius(axis_phi, toroidal_height, half_width, flattening)
    a = twist
    b = np.cos(phi)/Rc
    return (
        B0
        *(1-r*b)
        *((1-r*b)**2+a**2*r**2)**(-(2*a**2+b**2)/2/(a**2+b**2))
        *np.exp(
            -a*b/(a**2+b**2)*np.arctan(((a**2+b**2)*r-b)/a)
            +a*b/(a**2+b**2)*np.arctan(-b/a)
        )
    )

def Bphi(
        r,
        phi,
        axis_phi,
        toroidal_height,
        half_width,
        flattening,
        twist):
    return twist*r/(1-r*np.cos(phi)/axis_curvature_radius(
        axis_phi,
        toroidal_height,
        half_width,
        flattening
    ))*Bz(r, phi, axis_phi, toroidal_height, half_width, flattening, twist)

def Br(
        r,
        phi,
        axis_phi,
        toroidal_height,
        half_width,
        flattening,
        twist):
    return (
        r/z0(axis_phi, toroidal_height, half_width, flattening)
        /(
            1-r*np.cos(phi)/axis_curvature_radius(
                axis_phi,
                toroidal_height,
                half_width,
                flattening
            )
        )*Bz(r, phi, axis_phi, toroidal_height, half_width, flattening, twist)
    )
def solveTorus():
    r, phi, T0, Rc, Bz, Bzr, Bzphi = symbols('r phi T0 Rc Bz Bzr Bzphi')
    eqns = [
        -Bzr-(T0*(2-r*cos(phi)/Rc)/(1-r*cos(phi)/Rc)**2*Bz+T0*r/(1-r*cos(phi)/Rc)*Bzr)*T0*r/(1-r*cos(phi)/Rc),
        Bzphi
    ]
    sol = linsolve(eqns, [Bzr, Bzphi])
    pprint(sol)

def integrateFlux():
    r, phi, T0, Rc, B0, Rp = symbols('r phi T0 Rc B0 Rp')
    sol = integrate(
        B0
        *(1-r*cos(phi)/Rc)
        *((1-r*cos(phi)/Rc)**2+T0**2*r**2)**(-(2*T0**2+(cos(phi)/Rc)**2)/2/(T0**2+(cos(phi)/Rc)**2))
        *exp(
            -T0*cos(phi)/Rc/(T0**2+(cos(phi)/Rc)**2)*atan(((T0**2+(cos(phi)/Rc)**2)*r-cos(phi)/Rc)/T0)
            +T0*cos(phi)/Rc/(T0**2+(cos(phi)/Rc)**2)*atan(-cos(phi)/Rc/T0)
        )
        *r
        *2*pi,
        (phi, 0, 2*pi),
        (r, 0, Rp)
    )
    pprint(sol)

def integrateFluxNumerical(
        axis_phi,
        toroidal_height,
        poloidal_height,
        half_width,
        flattening,
        twist,
        B0):
    flux = dblquad(
        lambda phi, r: Bz(r, phi, axis_phi, toroidal_height, half_width, flattening, twist, B0)*r*np.pi*2,
        0, cs_r(axis_phi, toroidal_height, poloidal_height, half_width, flattening),
        lambda r: 0,
        lambda r: 2*np.pi
    )
    print(flux[0])
    return flux[0]

def checkBrBphi(
        axis_phi,
        toroidal_height,
        half_width,
        flattening,
        twist):
    plt.plot(axis_phi, 1/z0(axis_phi, toroidal_height, half_width, flattening)/twist)
    plt.show()

def test_b_quiver(
        axis_phi,
        toroidal_height,
        half_width,
        flattening,
        twist,
        x=u.au.to(u.m, np.linspace(-0.1, 0.1, 50)),
        y=u.au.to(u.m, np.linspace(-0.1, 0.1, 50))):
    
    x, y = np.meshgrid(x, y)

    r = np.sqrt(x**2+y**2)
    phi = np.arctan2(y, x)

    b = np.array([
        Br(r, phi, axis_phi, toroidal_height, half_width, flattening, twist),
        Bphi(r, phi, axis_phi, toroidal_height, half_width, flattening, twist),
        Bz(r, phi, axis_phi, toroidal_height, half_width, flattening, twist, 50)
    ])
    # b /= np.linalg.norm(b)
    bt = np.sqrt(b[0, :]**2+b[1, :]**2+b[2, :]**2)

    plt.pcolormesh(x, y, bt)
    plt.colorbar()
    plt.show()

    plt.pcolormesh(x, y, b[0, :])
    plt.colorbar()
    plt.show()

    plt.pcolormesh(x, y, b[1, :])
    plt.colorbar()
    plt.show()

    plt.pcolormesh(x, y, b[2, :])
    plt.colorbar()
    plt.show()

    plt.quiver(
        x,
        y,
        b[0, :]*np.cos(phi)-b[1, :]*np.sin(phi),
        b[0, :]*np.sin(phi)+b[1, :]*np.cos(phi)
    )
    plt.show()

# def test_b_const(
#         r,
#         phi,
#         phi_axis,
#         toroidal_height,
#         poloidal_height,
#         half_width,
#         flattening,
#         twist):
#     b = np.array([
#         dradius_dlength(
#             phi_axis,
#             toroidal_height,
#             poloidal_height,
#             half_width,
#             flattening
#         )
#         /radius(
#             phi_axis,
#             toroidal_height,
#             poloidal_height,
#             half_width,
#             flattening
#         )
#         *r,
#         2*np.pi*twist*r
#         /(
#             1
#             -r*np.cos(phi)
#             /axis_curvature_radius(
#                 phi_axis, toroidal_height, half_width, flattening
#             )
#         ),
#         1
#     ])
#     b /= np.linalg.norm(b)
#     return b

# twist = 0.1*2*np.pi/2.5
# print(twist)
# test_b_quiver(
#     -np.pi/180*0,
#     1,
#     np.pi/180*60,
#     0.5,
#     twist,
#     x=np.linspace(-0.1, 0.1, 50),
#     y=np.linspace(-0.1, 0.1, 50)
# )

# print(b)

# solve()
# solveTorus()
# integrateFlux()
for i in range(10):
    integrateFluxNumerical(
        0,
        u.au.to(u.m, 1),
        u.au.to(u.m, 0.1),
        np.pi/4,
        0.5,
        2*2*np.pi/u.au.to(u.m, 2.5),
        20e-9
    )

# checkBrBphi(
#     np.linspace(-np.pi/6, np.pi/6, 100),
#     u.au.to(u.m, 1),
#     np.pi/3,
#     0.5,
#     1*np.pi*2/c.au
# )

# data = []
# for r in np.linspace(0, 2, 40):
#     data.append(B(r, 0*np.pi, 1e3, 1*2*np.pi, 1e3)[2])
#     # data.append([
#     #     lnBzr(r, np.pi, 1e10, 1*2*np.pi, 1e3),
#     #     lnBzphi(r, np.pi, 1e10, 1*2*np.pi, 1e3),
#     #     lnBzz(r, np.pi, 1e10, 1*2*np.pi, 1e3),
#     #     lnBzr(r, np.pi, 1e10, 1*2*np.pi, 1e3)
#     #     +lnBzphi(r, np.pi, 1e10, 1*2*np.pi, 1e3)
#     #     +lnBzz(r, np.pi, 1e10, 1*2*np.pi, 1e3)
#     # ])
# plt.plot(data)
# plt.show()

# print(
#     B(
#         r=0,
#         phi=0*np.pi/180,
#         z=10,
#         T0=0*2*np.pi,
#         Rc=10
#     )
# )
