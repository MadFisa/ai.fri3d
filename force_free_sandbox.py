
import numpy as np
from matplotlib import pyplot as plt
from sympy import symbols, cos
from sympy.solvers.solveset import linsolve
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

def solve():
    r, phi, z, T0, Rc, Bz, Bzr, Bzphi, Bzz = symbols('r phi z T0 Rc Bz Bzr Bzphi Bzz')
    eqns = [
        -r/z**2*Bz+r/z*Bzz-Bzr-(T0*(2-r*cos(phi)/Rc)/(1-r*cos(phi)/Rc)**2*Bz+T0*r/(1-r*cos(phi)/Rc)*Bzr-Bzphi/z)*T0*r/(1-r*cos(phi)/Rc),
        Bzphi/r-T0*r/(1-r*cos(phi)/Rc)*Bzz-(T0*(2-r*cos(phi)/Rc)/(1-r*cos(phi)/Rc)**2*Bz+T0*r/(1-r*cos(phi)/Rc)*Bzr-Bzphi/z)*r/z,
        (Bzphi/r-T0*r/(1-r*cos(phi)/Rc)*Bzz)*T0/(1-r*cos(phi)/Rc)-(-r/z**2*Bz+r/z*Bzz-Bzr)/z
    ]
    sol = linsolve(eqns, [Bzr, Bzphi, Bzz])
    print(sol)

def test_b_quiver(
        phi_axis,
        toroidal_height,
        poloidal_height,
        half_width,
        flattening,
        twist):
    
    bz = 1e-9
    x = np.linspace(-1, 1, 40)
    y = np.linspace(-1, 1, 40)
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
        *bz,
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

solve()
