# pylint: disable=E1101
# pylint: disable=C0103
from datetime import datetime
import calendar
import numpy as np
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from astropy import units as u
from ai import cdas
from ai.shared.data import getSTA
from ai.fri3d.optimize import PolyProfile, fit2insitu


# data_b = cdas.get_data(
#     'sp_phys',
#     'STA_L1_MAG_RTN',
#     datetime(2010, 12, 15, 10, 20),
#     datetime(2010, 12, 16, 4),
#     ['BFIELD'],
#     cdf=True
# )



# t = np.array([datetime.utcfromtimestamp(t) for t in data_b['Epoch']])
# b = data_b['BFIELD']
# f = interp1d(data_v['epoch'], data['proton_bulk_speed'], kind='linear', axis=0)
# v = f(t)

d, b, _, p = getSTA(
    datetime(2010, 12, 15, 10, 20),
    datetime(2010, 12, 16, 4),
)
t = np.array([calendar.timegm(x.timetuple()) for x in d])

cdas.set_cache(True, './data')
data_v = cdas.get_data(
    'sp_phys',
    'STA_L2_PLA_1DMAX_1MIN',
    datetime(2010, 12, 15, 10, 20),
    datetime(2010, 12, 16, 4),
    ['proton_bulk_speed'],
    cdf=True
)

m = data_v['proton_bulk_speed'] < 0
data_v['proton_bulk_speed'][m] = np.nan

f = interp1d(
    np.array([calendar.timegm(x.timetuple()) for x in data_v['epoch']]),
    data_v['proton_bulk_speed'],
    kind='linear',
    bounds_error=False,
    fill_value=np.nan
)

vt = f(t)

vt = u.Unit('km/s').to(u.Unit('m/s'), vt)

# fig = plt.figure()
# ax = fig.add_subplot(211)
# ax.plot(t, b)
# ax = fig.add_subplot(212)
# ax.plot(t, vt)
# plt.show()

dfr = fit2insitu(
    np.mean(p[:, 0]),
    np.mean(p[:, 1]),
    np.mean(p[:, 2]),
    t,
    b,
    vt,
    latitude=PolyProfile(bounds=[u.deg.to(u.rad, [-15, 5]).tolist()]),
    longitude=PolyProfile(bounds=[u.deg.to(u.rad, [40, 70]).tolist()]),
    toroidal_height=PolyProfile(bounds=[
        (-0.5, 0),
        (400e3, 600e3),
        (u.au.to(u.m, 0.5), u.au.to(u.m, 1.2))
    ]),
    poloidal_height=PolyProfile(bounds=[
        (0, 50e3),
        (u.au.to(u.m, 0.05), u.au.to(u.m, 0.2))
    ]),
    half_width=PolyProfile(bounds=[u.deg.to(u.rad, [45, 65]).tolist()]),
    tilt=PolyProfile(bounds=[u.deg.to(u.rad, [0, 30]).tolist()]),
    flattening=PolyProfile(bounds=[(0.2, 0.8)]),
    pancaking=PolyProfile(bounds=[u.deg.to(u.rad, [10, 30]).tolist()]),
    skew=PolyProfile(params=[0]),
    twist=PolyProfile(bounds=[(0, 3)]),
    flux=PolyProfile(bounds=[(1e14, 1e15)]),
    sigma=PolyProfile(bounds=[(1.8, 2.2)]),
    polarity=PolyProfile(params=[-1]),
    chirality=PolyProfile(params=[1]),
    weights={'t': 1, 'b': 1, 'vt': 0.5},
    verbose=True
)
