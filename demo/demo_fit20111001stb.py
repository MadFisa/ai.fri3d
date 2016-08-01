
from ai.fri3d.optimize import fit2remote
from astropy import units as u

fit2remote(
    cor2a=True,
    cor2a_img='data/cor2a_20111001_233900.png',
    cor2a_aov=u.deg.to(u.rad, 4.0),
    cor2a_xc=3.43595e7-7.15174e7,
    cor2a_yc=2.31816e7-3.21944e7,
    sta_r=u.au.to(u.m, 0.967106),
    sta_lon=u.deg.to(u.rad, 103.933),
    sta_lat=u.deg.to(u.rad, -4.464),

    cor2b=True,
    cor2b_img='data/cor2b_20111001_233900.png',
    cor2b_aov=u.deg.to(u.rad, 4.0),
    cor2b_xc=9.89179e7-4.22797e7,
    cor2b_yc=-7.36381e7+8.75883e7,
    stb_r=u.au.to(u.m, 1.078776),
    stb_lon=u.deg.to(u.rad, -97.833),
    stb_lat=u.deg.to(u.rad, 1.598),
    
    c3=True,
    c3_img='data/c3_20111001_233916.png',
    c3_fov=u.R_sun.to(u.m, 30.0),
    c3_xc=4.62916e8-1.94402e8,
    c3_yc=1.64648e9-8.5263e8,
    soho_r=u.au.to(u.m, 1.0),
    soho_lat=u.deg.to(u.rad, 0.0),
    soho_lon=u.deg.to(u.rad, 0.0),

    latitude=u.deg.to(u.rad, 5.5),
    longitude=u.deg.to(u.rad, -95.0),
    toroidal_height=u.R_sun.to(u.m, 14.9),
    poloidal_height=u.R_sun.to(u.m, 4.5),
    half_width=u.deg.to(u.rad, 75.0),
    tilt=u.deg.to(u.rad, 21.0),
    flattening=0.55,
    pancaking=u.deg.to(u.rad, 27.0),
    skew=u.deg.to(u.rad, 0.0),
    
    spline_s_phi_kind='cubic',
    spline_s_phi_n=500)
