
import numpy as np
from matplotlib import pyplot as plt
from astropy import units as u
from datetime import datetime
import calendar

Rt0 = u.R_sun.to(u.m, 12.0)
aRt = 0.000142307095103
v0Rt = u.Unit('km/s').to(u.Unit('m/s'), 1555.85227047)
dRt = u.R_sun.to(u.m, 28.0197506113)
v1Rt = u.Unit('km/s').to(u.Unit('m/s'), 1055.21504775)

d0 = datetime(2011, 6, 4, 8, 54)
t0 = calendar.timegm(d0.timetuple())

di = datetime(2011, 6, 5, 11, 30)
ti = calendar.timegm(di.timetuple())

d1_sta = datetime(2011, 6, 6, 14, 10)
t1_sta = calendar.timegm(d1_sta.timetuple())

# Rt = lambda t: (
#     (1.0-np.exp(-aRt*(t-t0)))*(v0Rt*(t-t0)+(dRt-Rt0))+Rt0
#     if t <= ti else
#     (1.0-np.exp(-aRt*(ti-t0)))*(v0Rt*(ti-t0)+(dRt-Rt0))+Rt0+v1Rt*(t-ti)
# )
Rt = lambda t: (
    (v0Rt-v1Rt)/aRt*(1.0-np.exp(-aRt*(t-t0)))+v1Rt*(t-t0)+Rt0
)

t = np.linspace(t0, t1_sta)

plt.plot(
    t,
    u.m.to(u.au, np.array([Rt(x) for x in t]))
)

Rt0 = u.R_sun.to(u.m, 12.0)
aRt = 9.31364197379e-05
v0Rt = u.Unit('km/s').to(u.Unit('m/s'), 1730.01821538)
dRt = u.R_sun.to(u.m, 61.000228854)
v1Rt = u.Unit('km/s').to(u.Unit('m/s'), 1297.42507118)

d0 = datetime(2011, 6, 4, 22, 54)
t0 = calendar.timegm(d0.timetuple())

di = datetime(2011, 6, 5, 11, 30)
ti = calendar.timegm(di.timetuple())

d1_sta = datetime(2011, 6, 7, 1)
t1_sta = calendar.timegm(d1_sta.timetuple())

# Rt = lambda t: (
#     (1.0-np.exp(-aRt*(t-t0)))*(v0Rt*(t-t0)+(dRt-Rt0))+Rt0
#     if t <= ti else
#     (1.0-np.exp(-aRt*(ti-t0)))*(v0Rt*(ti-t0)+(dRt-Rt0))+Rt0+v1Rt*(t-ti)
# )
Rt = lambda t: (
    (v0Rt-v1Rt)/aRt*(1.0-np.exp(-aRt*(t-t0)))+v1Rt*(t-t0)+Rt0
)

t = np.linspace(t0, t1_sta)

plt.plot(
    t,
    u.m.to(u.au, np.array([Rt(x) for x in t]))
)

plt.axvline(ti)
plt.axhline(0.39)
plt.axhline(0.72)
plt.axhline(1.01)

plt.show()


theta0 = u.deg.to(u.rad, 34.0)
theta1 = u.deg.to(u.rad, -2.40919340271)
atheta = 0.000200128388844
theta = lambda t: (
    (theta0-theta1)*np.exp(-atheta*(t-t0))+theta1
)

plt.plot(
    t,
    u.rad.to(u.deg, np.array([theta(x) for x in t]))
)

theta0 = u.deg.to(u.rad, 22.0)
theta1 = u.deg.to(u.rad, 14.0631687451)
atheta = 0.000415996235746
theta = lambda t: (
    (theta0-theta1)*np.exp(-atheta*(t-t0))+theta1
)

plt.plot(
    t,
    u.rad.to(u.deg, np.array([theta(x) for x in t]))
)


plt.axvline(ti)
plt.show()
