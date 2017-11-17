
from datetime import datetime
import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from ai import cdas
from ai import ssc
from ai.fri3d.optimize import PolyProfile, fit2insitu

# cdas.set_cache(True, './data')
# data_b = cdas.get_data(
#     'sp_phys',
#     'STA_L1_MAG_RTN',
#     datetime(2010, 12, 15, 10, 20),
#     datetime(2010, 12, 16, 4),
#     ['BFIELD'],
#     cdf=True
# )

# data_v = cdas.get_data(
#     'sp_phys',
#     'STA_L2_PLA_1DMAX_1MIN',
#     datetime(2010, 12, 15, 10, 20),
#     datetime(2010, 12, 16, 4),
#     ['proton_bulk_speed'],
#     cdf=True
# )

# m = data_v['proton_bulk_speed'] > 0
# data_v['epoch'] = data_v['epoch'][m]
# data_v['proton_bulk_speed'] = data_v['proton_bulk_speed'][m]

# t = np.array([datetime.utcfromtimestamp(t) for t in data_b['Epoch']])
# b = data_b['BFIELD']
# f = interp1d(data_v['epoch'], data['proton_bulk_speed'], kind='linear', axis=0)
# v = f(t)

# t, b, v, p = getSTA(
#     datetime(2010, 12, 15, 10, 20),
#     datetime(2010, 12, 16, 4),
# )

# print(v)

# dfr = fit2insitu(t, b, v)

fig = plt.figure()
ax = fig.add_subplot(211)
ax.plot(t, b)
ax = fig.add_subplot(212)
ax.plot(t, v)
plt.show()
