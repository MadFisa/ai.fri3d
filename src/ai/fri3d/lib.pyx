from libc.math cimport cos, pi, tan, sqrt, exp, atan2, sin

cdef api double vanilla_axis_height(int n, double *xx):
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
    cdef double coeff_angle = pi / 2 / xx[2]
    cdef double res = xx[1] * cos(coeff_angle * xx[0]) ** xx[3]
    return res

cdef api double vanilla_axis_dlength(int n, double *xx):
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
    cdef double coeff_angle = pi / 2 / xx[2]
    cdef double res = (
        xx[1]
        * cos(coeff_angle * xx[0]) ** xx[3]
        * sqrt(coeff_angle ** 2 * xx[3] ** 2 * tan(coeff_angle * xx[0]) ** 2 + 1)
    )
    return res

cdef api double vanilla_axis_rdflux(int n, double *xx):
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

    cdef double coeff_angle = pi / 2 / xx[4]
    cdef double axis_height = xx[3] * cos(coeff_angle * xx[2]) ** xx[6]
    cdef double axis_dheight = -coeff_angle * xx[6] * tan(coeff_angle * xx[2]) * axis_height
    cdef double axis_dlength = sqrt(axis_height ** 2 + axis_dheight ** 2)
    cdef double ry = axis_height * tan(xx[5]) * xx[3] / xx[3]
    cdef double sigmay = xx[9]
    cdef res = (
        exp(
            -(xx[0] * cos(xx[1]) / ry) ** 2 / 2 / sigmay ** 2
            - (xx[0] * sin(xx[1]) / ry) ** 2 / 2 / sigmay ** 2
        )
        * sin(
            atan2(
                axis_dlength * xx[10], axis_height ** 2 * 2 * pi * xx[8] / (1 - (1 - xx[7]) * sin(xx[1]))
            )
        )
        * xx[0]
    )
    return res
