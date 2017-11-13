"""Helper functions."""

import math

def subtract_period(value, period):
    """Reduce angle by period.

    Args:
        value (float): initial angle [rad].
        period (float): period [rad].

    Returns:
        (float): angle reduced by correct number of periods.
    """
    return value-math.copysign(value, 1)*(math.fabs(value)//period)*period
