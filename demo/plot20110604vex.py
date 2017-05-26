
# COR
d0_cor = datetime(2011, 6, 4, 8, 54)
t0_cor = calendar.timegm(d0_cor.timetuple())
latitude_cor = u.deg.to(u.rad, 30.0)
longitude_cor = u.deg.to(u.rad, 110.0)
toroidal_height_cor = u.R_sun.to(u.m, 12.5)
poloidal_height_cor = u.R_sun.to(u.m, 3.5)
half_width_cor = u.deg.to(u.rad, 40.0)
tilt_cor = u.deg.to(u.rad, 37.0)
flattening_cor = 0.4
pancaking_cor = u.deg.to(u.rad, 25.0)
skew = 0.0
polarity = -1.0
chirality = 1.0
spline_s_phi_kind = 'cubic',
spline_s_phi_n = 500

# MESSENGER
d0_mes = datetime(2011, 6, 4, 17, 9)
d1_mes = datetime(2011, 6, 4, 17, 10)
t0_mes = calendar.timegm(d0_mes.timetuple())
t1_mes = calendar.timegm(d1_mes.timetuple())
d_mes, b_mes, _, p_mes = getMES(d0_mes, d1_mes)
t_mes = np.array([calendar.timegm(x.timetuple()) for x in d_mes])
bt_mes = np.mean(np.sqrt(b_mes[:,0]**2+b_mes[:,1]**2+b_mes[:,2]**2))
delta_mes = 10*3600

# VEX
d0_vex = datetime(2011, 6, 5, 8, 45)
d1_vex = datetime(2011, 6, 5, 11, 50)
t0_vex = calendar.timegm(d0_vex.timetuple())
t1_vex = calendar.timegm(d1_vex.timetuple())
d_vex, b_vex, _, p_vex = getVEX(d0_vex, d1_vex)
t_vex = np.array([calendar.timegm(x.timetuple()) for x in d_vex])
bt_vex = np.sqrt(b_vex[:,0]**2+b_vex[:,1]**2+b_vex[:,2]**2)