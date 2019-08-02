import numpy as np
from numba.pycc import CC

cc = CC("nb")
cc.verbose = True


@cc.export("_nb_vanilla_axis_height", "f8(f8, f8, f8, f8)")
def _nb_vanilla_axis_height(args0, args1, args2, args3):
    """Evaluates height of the axis. Note that rotational skewing is not
    taken into account.

    Args:
        _ (scalar): number of elements in args array,
            which is always equal to 4 [unitless].
        args[0] (scalar): Angular coordinate of a point on
            the axis [rad] in polar coordinates, lies in the range
            [-half_width, half_width] [rad].
        args[1] (scalar): Toroidal height [m].
        args[2] (scalar): Half width agnle [rad].
        args[3] (scalar): Flattening coefficient [unitless].

    Returns:
        scalar: height evaluated at `phi` angular location
            of the axis [m/rad].
    """
    coeff_angle = np.pi / 2 / args2
    res = args1 * np.cos(coeff_angle * args0) ** args3
    return res


@cc.export("_nb_vanilla_axis_rdflux", "f8(f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8)")
def _nb_vanilla_axis_rdflux(args0, args1, args2, args3, args4, args5, args6, args7, args8, args9, args10):
    # args[0] = r
    # args[1] = theta
    # args[2] = phi
    # args[3] = toroidal_height
    # args[4] = half_width
    # args[5] = half_height
    # args[6] = flattening
    # args[7] = pancaking
    # args[8] = twist
    # args[9] = sigma
    # args[10] = intrdphi

    coeff_angle = np.pi / 2 / args4
    axis_height = args3 * np.cos(coeff_angle * args2) ** args6
    axis_dheight = -coeff_angle * args6 * np.tan(coeff_angle * args2) * axis_height
    axis_dlength = np.sqrt(axis_height ** 2 + axis_dheight ** 2)
    ry = axis_height * np.tan(args5) * args3 / args3
    sigmay = args9
    return (
        np.exp(
            -(args0 * np.cos(args1) / ry) ** 2 / 2 / sigmay ** 2 - (args0 * np.sin(args1) / ry) ** 2 / 2 / sigmay ** 2
        )
        * np.sin(
            np.arctan2(axis_dlength * args10, axis_height ** 2 * 2 * np.pi * args8 / (1 - (1 - args7) * np.sin(args1)))
        )
        * args0
    )


@cc.export("_nb_vanilla_axis_dlength", "f8(f8, f8, f8, f8)")
def _nb_vanilla_axis_dlength(args0, args1, args2, args3):
    """Evaluates derivative of the axis length ds/d(phi). Note that
    rotational skewing is not taken into account.

    Args:
        _ (scalar): number of elements in args array,
            which is always equal to 4 [unitless].
        args[0] (scalar): Angular coordinate of a point on
            the axis [rad] in polar coordinates, lies in the range
            [-half_width, half_width] [rad].
        args[1] (scalar): Toroidal height [m].
        args[2] (scalar): Half width agnle [rad].
        args[3] (scalar): Flattening coefficient [unitless].

    Returns:
        scalar: ds/d(phi) evaluated at `phi` angular location
            of the axis [m/rad].
    """
    coeff_angle = np.pi / 2 / args2
    res = (
        args1
        * np.cos(coeff_angle * args0) ** args3
        * np.sqrt(coeff_angle ** 2 * args3 ** 2 * np.tan(coeff_angle * args0) ** 2 + 1)
    )
    return res


if __name__ == "__main__":
    cc.compile()
