

import numpy as np
from scipy.integrate import quad, fixed_quad
from matplotlib import pyplot as plt
from sympy import pprint, symbols, cos, sin, integrate, Function
from sympy.solvers.solveset import linsolve
from sympy.solvers.pde import pdsolve
from astropy import units as u

def axis_r(
        phi,
        toroidal_height,
        half_width,
        flattening):
    return toroidal_height*np.cos(np.pi/2/half_width*phi)**flattening

def axis_dr_dphi(
        phi,
        toroidal_height,
        half_width,
        flattening):
    return (
        -np.pi/2/half_width*flattening
        *np.tan(np.pi/2/half_width*phi)
        *axis_r(phi, toroidal_height, half_width, flattening)
    )

def axis_d2r_dphi2(
        phi,
        toroidal_height,
        half_width,
        flattening):
    return (
        -np.pi/2/half_width*flattening
        *np.tan(np.pi/2/half_width*phi)
        *axis_dr_dphi(phi, toroidal_height, half_width, flattening)
        -(np.pi/2/half_width)**2*flattening
        /np.cos(np.pi/2/half_width*phi)**2
        *axis_r(phi, toroidal_height, half_width, flattening)
    )

def axis_curvature_radius(
        phi,
        toroidal_height,
        half_width,
        flattening):
    return (
        (
            axis_r(phi, toroidal_height, half_width, flattening)**2
            +axis_dr_dphi(phi, toroidal_height, half_width, flattening)**2
        )**(3/2)
        /np.abs(
            axis_r(phi, toroidal_height, half_width, flattening)**2
            +2*axis_dr_dphi(phi, toroidal_height, half_width, flattening)**2
            -axis_r(phi, toroidal_height, half_width, flattening)
            *axis_d2r_dphi2(phi, toroidal_height, half_width, flattening)
        )
    )

def radius(phi, toroidal_height, poloidal_height, half_width, flattening):
    return (
        axis_r(phi, toroidal_height, half_width, flattening)
        /toroidal_height
        *poloidal_height
    )

def dradius_dlength(
        phi,
        toroidal_height,
        poloidal_height,
        half_width,
        flattening):
    return (
        poloidal_height/toroidal_height
        *axis_dr_dphi(phi, toroidal_height, half_width, flattening)
        /np.sqrt(
            axis_r(phi, toroidal_height, half_width, flattening)**2
            +axis_dr_dphi(phi, toroidal_height, half_width, flattening)**2
        )
    )

def solveTaper():
    r, phi, z, T0, Rc, Bz, Bzr, Bzphi, Bzz = symbols('r phi z T0 Rc Bz Bzr Bzphi Bzz')
    eqns = [
        # (-r/z**2*Bz+r/z*Bzz-Bzr)-(2*T0*Bz+T0*r*Bzr-Bzphi/z)*T0*r,
        # (Bzphi/r-T0*r*Bzz)-(2*T0*Bz+T0*r*Bzr-Bzphi/z)*r/z,
        # (Bzphi/r-T0*r*Bzz)*T0*r-(-r/z**2*Bz+r/z*Bzz-Bzr)*r/z
        (-r/z**2*Bz+r/z*Bzz-Bzr),
        (-r/z**2*Bz+r/z*Bzz-Bzr)*r/z
    ]
    sol = linsolve(eqns, [Bzr, Bzz])
    pprint(sol)
    solveDiffTaper()

def solve():
    r, phi, z, T0, Rc, Bz, Bzr, Bzphi, Bzz, f, g = symbols('r phi z T0 Rc Bz Bzr Bzphi Bzz f g')
    eqns = [
        -r/z**2/(1-r*cos(phi)/Rc)*Bz+r/z/(1-r*cos(phi)/Rc)*Bzz-Bzr-(T0*(2-r*cos(phi)/Rc)/(1-r*cos(phi)/Rc)**2*Bz+T0*r/(1-r*cos(phi)/Rc)*Bzr+1/z*r/Rc*sin(phi)/(1-r*cos(phi)/Rc)**2*Bz-1/z/(1-r*cos(phi)/Rc)*Bzphi)*T0*r/(1-r*cos(phi)/Rc),
        Bzphi/r-T0*r/(1-r*cos(phi)/Rc)*Bzz-(T0*(2-r*cos(phi)/Rc)/(1-r*cos(phi)/Rc)**2*Bz+T0*r/(1-r*cos(phi)/Rc)*Bzr+1/z*r/Rc*sin(phi)/(1-r*cos(phi)/Rc)**2*Bz-1/z/(1-r*cos(phi)/Rc)*Bzphi)*r/z/(1-r*cos(phi)/Rc),
        (Bzphi/r-T0*r/(1-r*cos(phi)/Rc)*Bzz)*T0*r/(1-r*cos(phi)/Rc)-(-r/z**2/(1-r*cos(phi)/Rc)*Bz+r/z/(1-r*cos(phi)/Rc)*Bzz-Bzr)*r/z/(1-r*cos(phi)/Rc),
        # Bzz+f/z**3
        # (2-r*cos(phi)/Rc)/(1-r*cos(phi)/Rc)**2/z*Bz+r/z/(1-r*cos(phi)/Rc)*Bzr-T0*r/Rc*sin(phi)/(1-r*cos(phi)/Rc)**2*Bz+T0/(1-r*cos(phi)/Rc)*Bzphi+Bzz
        #-r/z**2*Bz+r/z*Bzz-Bzr-(T0*(2-r*cos(phi)/Rc)/(1-r*cos(phi)/Rc)**2*Bz+T0*r/(1-r*cos(phi)/Rc)*Bzr-Bzphi/z)*T0*r/(1-r*cos(phi)/Rc),
        #Bzphi/r-T0*r/(1-r*cos(phi)/Rc)*Bzz-(T0*(2-r*cos(phi)/Rc)/(1-r*cos(phi)/Rc)**2*Bz+T0*r/(1-r*cos(phi)/Rc)*Bzr-Bzphi/z)*r/z,
        #(Bzphi/r-T0*r/(1-r*cos(phi)/Rc)*Bzz)*T0/(1-r*cos(phi)/Rc)-(-r/z**2*Bz+r/z*Bzz-Bzr)/z
    ]
    sol = linsolve(eqns, [Bzr, Bzphi, Bzz])
    pprint(sol)
    # pprint(integrate(sol.args[0][0]/Bz, r))
    # pprint(integrate(sol.args[0][1]/Bz, phi))
    # pprint(integrate(sol.args[0][2]/Bz, z))

def solveDiff():
    r, phi, T0, Rc, A = symbols('r phi T0, Rc A')
    f = Function('f')
    u = f(r, phi)
    ur = u.diff(r)
    uphi = u.diff(phi)
    pprint(
        pdsolve(
            ur+uphi+r/(1-r*cos(phi)/Rc)*(1-(A-1)*T0*r)/A*u
        )
    )

def solveDiffTaper():
    r, z = symbols('r z')
    f = Function('f')
    u = f(r, z)
    ur = u.diff(r)
    uz = u.diff(z)
    pprint(
        pdsolve(
            ur-r/z*uz+r/z**2*u
        )
    )

def lnBzr(r, phi, z, T0, Rc):
    # return -Rc*r*(Rc**2*T0**2*r**2 + 3*Rc**2 - Rc*T0*r*z*np.sin(phi) - 3*Rc*r*np.cos(phi) + r**2*np.cos(phi)**2)/((Rc - r*np.cos(phi))*(Rc**2*T0**2*r**2*z**2 + Rc**2*r**2 + z**2*(Rc - r*np.cos(phi))**2))
    return -Rc*r*(Rc**2*T0**2*r**2 + 3*Rc**2 - Rc*T0*r*z*np.sin(phi) - 3*Rc*r*np.cos(phi) + r**2*np.cos(phi)**2)/((Rc - r*np.cos(phi))*(Rc**2*T0**2*r**2*z**2 + Rc**2*r**2 + z**2*(Rc - r*np.cos(phi))**2))

def lnBzphi(r, phi, z, T0, Rc):
    # return Rc**2*T0*r**2*(Rc*r**2 + T0*r*z**3*np.sin(phi) - z**2*(2*Rc - r*np.cos(phi)))/(z*(Rc - r*np.cos(phi))*(Rc**2*T0**2*r**2*z**2 + Rc**2*r**2 + z**2*(Rc - r*np.cos(phi))**2))
    return Rc**2*T0*r**2*(Rc*r**2 + T0*r*z**3*np.sin(phi) - z**2*(2*Rc - r*np.cos(phi)))/(z*(Rc - r*np.cos(phi))*(Rc**2*T0**2*r**2*z**2 + Rc**2*r**2 + z**2*(Rc - r*np.cos(phi))**2))

def lnBzz(r, phi, z, T0, Rc):
    # return Rc*(Rc*r**2 + T0*r*z**3*np.sin(phi) - z**2*(2*Rc - r*np.cos(phi)))/(z*(Rc**2*T0**2*r**2*z**2 + Rc**2*r**2 + z**2*(Rc - r*np.cos(phi))**2))
    return Rc*(Rc*r**2 + T0*r*z**3*np.sin(phi) - z**2*(2*Rc - r*np.cos(phi)))/(z*(Rc**2*T0**2*r**2*z**2 + Rc**2*r**2 + z**2*(Rc - r*np.cos(phi))**2))


def Bz(r, phi, z, T0, Rc):
    return np.exp(
        fixed_quad(lambda var_r: lnBzr(var_r, phi, z, T0, Rc), 0, r)[0]
        +fixed_quad(lambda var_phi: lnBzphi(r, var_phi, z, T0, Rc), 0, phi)[0]
        +fixed_quad(lambda var_z: lnBzz(r, phi, var_z, T0, Rc), 0, z)[0]
    )

def B(r, phi, z, T0, Rc):
    b_z = Bz(r, phi, z, T0, Rc)
    b_r = r/z/(1-r*np.cos(phi)/Rc)*b_z
    b_phi = T0*r/(1-r*np.cos(phi)/Rc)*b_z
    return (b_r, b_phi, b_z)

def test_b_quiver(
        phi_axis,
        toroidal_height,
        poloidal_height,
        half_width,
        flattening,
        twist):
    
    x = np.linspace(-1, 1, 10)
    y = np.linspace(-1, 1, 10)
    x, y = np.meshgrid(x, y)

    r = np.sqrt(x**2+y**2)
    phi = np.arctan2(y, x)

    # r = np.linspace(0, 1, 100)
    # phi = np.linspace(0, 2*np.pi, 100)

    r *= radius(
        phi_axis,
        toroidal_height,
        poloidal_height,
        half_width,
        flattening
    )
    # r, phi = np.meshgrid(r, phi)
    # x = r*np.cos(phi)
    # y = r*np.sin(phi)
    b = np.array([
        dradius_dlength(
            phi_axis,
            toroidal_height,
            poloidal_height,
            half_width,
            flattening
        )
        /radius(
            phi_axis,
            toroidal_height,
            poloidal_height,
            half_width,
            flattening
        )
        *r
        *bz
        /(
            1
            -r*np.cos(phi)
            /axis_curvature_radius(
                phi_axis, toroidal_height, half_width, flattening
            )
        ),
        2*np.pi*twist*r
        /(
            1
            -r*np.cos(phi)
            /axis_curvature_radius(
                phi_axis, toroidal_height, half_width, flattening
            )
        )
        *bz,
        np.ones(r.shape)
        *bz
    ])
    b /= np.linalg.norm(b)

    plt.quiver(
        x,
        y,
        b[0, :]*np.cos(phi)-b[1, :]*np.sin(phi),
        b[0, :]*np.sin(phi)+b[1, :]*np.cos(phi)
    )
    plt.show()
    return b

def test_b_const(
        r,
        phi,
        phi_axis,
        toroidal_height,
        poloidal_height,
        half_width,
        flattening,
        twist):
    b = np.array([
        dradius_dlength(
            phi_axis,
            toroidal_height,
            poloidal_height,
            half_width,
            flattening
        )
        /radius(
            phi_axis,
            toroidal_height,
            poloidal_height,
            half_width,
            flattening
        )
        *r,
        2*np.pi*twist*r
        /(
            1
            -r*np.cos(phi)
            /axis_curvature_radius(
                phi_axis, toroidal_height, half_width, flattening
            )
        ),
        1
    ])
    b /= np.linalg.norm(b)
    return b

# b = test_b_quiver(
#     0,
#     1,
#     0.1,
#     np.pi/4,
#     0.5,
#     5
# )

# print(b)

solveTaper()
# solveDiff()
# solve()



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
