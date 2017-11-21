# pylint: disable=E1101
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

cdas.set_cache(True, './data')
# data_b = cdas.get_data(
#     'sp_phys',
#     'STA_L1_MAG_RTN',
#     datetime(2010, 12, 15, 10, 20),
#     datetime(2010, 12, 16, 4),
#     ['BFIELD'],
#     cdf=True
# )

data_v = cdas.get_data(
    'sp_phys',
    'STA_L2_PLA_1DMAX_1MIN',
    datetime(2010, 12, 15, 10, 20),
    datetime(2010, 12, 16, 4),
    ['proton_bulk_speed'],
    cdf=True
)

# m = data_v['proton_bulk_speed'] > 0
# data_v['epoch'] = data_v['epoch'][m]
# data_v['proton_bulk_speed'] = data_v['proton_bulk_speed'][m]

# t = np.array([datetime.utcfromtimestamp(t) for t in data_b['Epoch']])
# b = data_b['BFIELD']
# f = interp1d(data_v['epoch'], data['proton_bulk_speed'], kind='linear', axis=0)
# v = f(t)

d, b, _, p = getSTA(
    datetime(2010, 12, 15, 10, 20),
    datetime(2010, 12, 16, 4),
)

# f = interp1d(
#     data_v['epoch'],
#     data_v['proton_bulk_speed'],
#     kind='linear',
#     axis=0
# )
# v = f(d)

# print(v)

t = [calendar.timegm(x.timetuple()) for x in d]
dfr = fit2insitu(
    t,
    np.mean(p[:, 0]),
    np.mean(p[:, 1]),
    np.mean(p[:, 2]),
    b,
    v=None,
    latitude=PolyProfile(bounds=[u.deg.to(u.rad, [-15, 5]).tolist()]),
    longitude=PolyProfile(bounds=[u.deg.to(u.rad, [40, 70]).tolist()]),
    toroidal_height=PolyProfile(bounds=[
        (400e3, 600e3),
        (u.au.to(u.m, 0.5), u.au.to(u.m, 1.2))
    ]),
    poloidal_height=PolyProfile(bounds=[u.au.to(u.m, [0.01, 0.2]).tolist()]),
    half_width=PolyProfile(params=u.deg.to(u.rad, [55]).tolist()),
    tilt=PolyProfile(bounds=[u.deg.to(u.rad, [0, 30]).tolist()]),
    flattening=PolyProfile(bounds=[(0.2, 0.8)]),
    pancaking=PolyProfile(bounds=[u.deg.to(u.rad, [10, 30]).tolist()]),
    skew=PolyProfile(params=[0]),
    twist=PolyProfile(bounds=[(0, 3)]),
    flux=PolyProfile(bounds=[(1e14, 1e15)]),
    sigma=PolyProfile(params=[2]),
    polarity=PolyProfile(params=[-1]),
    chirality=PolyProfile(params=[1]),
    verbose=True
)

# t0
# Rt = V*(t-t0)+
# Rt = V*(t-t0)+0.5 = V*t - V*t0+0.5
# 0 = V*t0 + R0
# 0.5 = V*t0 + R0
# R0 = 0.5 - V*t0
# fig = plt.figure()
# ax = fig.add_subplot(211)
# ax.plot(t, b)
# ax = fig.add_subplot(212)
# ax.plot(t, v)
# plt.show()
