
from ai.fri3d.optimize import fit2remote
from astropy import units as u

fit2remote(
    cor2a=True,
    cor2a_img='data/COR2A_20101212083900.png',
    cor2a_aov=u.deg.to(u.rad, 4.0),
    cor2a_xc=0.0,
    cor2a_yc=0.0,
    sta_r=u.au.to(u.m, 0.966087),
    sta_lon=u.deg.to(u.rad, 85.198),
    sta_lat=u.deg.to(u.rad, -7.346),

    cor2b=True,
    cor2b_img='data/COR2B_20101212083900.png',
    cor2b_aov=u.deg.to(u.rad, 4.0),
    cor2b_xc=0.0,
    cor2b_yc=0.0,
    stb_r=u.au.to(u.m, 1.070067),
    stb_lon=u.deg.to(u.rad, -87.282),
    stb_lat=u.deg.to(u.rad, 7.281),
    
    c3=True,
    c3_img='data/C3_20101212083900.png',
    c3_fov=u.R_sun.to(u.m, 30.0),
    c3_xc=0.0,
    c3_yc=0.0,
    soho_r=u.au.to(u.m, 1.0),
    soho_lat=u.deg.to(u.rad, 0.0),
    soho_lon=u.deg.to(u.rad, 0.0),

    latitude=u.deg.to(u.rad, 0.0),
    longitude=u.deg.to(u.rad, 0.0),
    toroidal_height=u.R_sun.to(u.m, 12.5),
    poloidal_height=u.R_sun.to(u.m, 3.5),
    half_width=u.deg.to(u.rad, 40.0),
    tilt=u.deg.to(u.rad, 0.0),
    flattening=0.5,
    pancaking=u.deg.to(u.rad, 20.0),
    skew=u.deg.to(u.rad, 0.0),
    
    spline_s_phi_kind='linear',
    spline_s_phi_n=100)
