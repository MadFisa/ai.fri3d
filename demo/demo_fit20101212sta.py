
from ai.fri3d.optimize import fit2remote
from astropy import units as u

fit2remote(
    cor2a=True,
    cor2a_img='data/cor2a_20101212_083900.png',
    cor2a_aov=u.deg.to(u.rad, 4.0),
    cor2a_xc=-1.16802e7,
    cor2a_yc=-2.43284e7,
    sta_r=u.au.to(u.m, 0.966087),
    sta_lon=u.deg.to(u.rad, 85.198),
    sta_lat=u.deg.to(u.rad, -7.346),

    cor2b=True,
    cor2b_img='data/cor2b_20101212_083900.png',
    cor2b_aov=u.deg.to(u.rad, 4.0),
    cor2b_xc=-3.85197e7,
    cor2b_yc=9.39991e7,
    stb_r=u.au.to(u.m, 1.070067),
    stb_lon=u.deg.to(u.rad, -87.282),
    stb_lat=u.deg.to(u.rad, 7.281),
    
    c3=True,
    c3_img='data/c3_20101212_083934.png',
    c3_fov=u.R_sun.to(u.m, 30.0),
    c3_xc=-2.07815e8,
    c3_yc=-8.50331e8,
    soho_r=u.au.to(u.m, 1.0),
    soho_lat=u.deg.to(u.rad, 0.0),
    soho_lon=u.deg.to(u.rad, 0.0),

    latitude=u.deg.to(u.rad, -14.5),
    longitude=u.deg.to(u.rad, 55.0),
    toroidal_height=u.R_sun.to(u.m, 12.5),
    poloidal_height=u.R_sun.to(u.m, 3.5),
    half_width=u.deg.to(u.rad, 55.0),
    tilt=u.deg.to(u.rad, 16.0),
    flattening=0.6,
    pancaking=u.deg.to(u.rad, 23.0),
    skew=u.deg.to(u.rad, 0.0),
    
    spline_s_phi_kind='cubic',
    spline_s_phi_n=500)
