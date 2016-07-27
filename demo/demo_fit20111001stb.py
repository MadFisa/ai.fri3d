
def fit2remote(
    latitude=-np.pi/180.0*5.0, 
    longitude=np.pi/180.0*123.0, 
    # longitude=np.pi/180.0*90.0, 
    toroidal_height=12.5/AU_RS,
    poloidal_height=3.5/AU_RS,
    half_width=np.pi/180.0*43.0, 
    tilt=np.pi/180.0*44.0, 
    flattening=0.37, 
    pancaking=np.pi/180.0*18.0, 
    skew=np.pi/180.0*5.0, 
    tapering=1.0,
    twist=2.79284826, 
    flux=1e14,
    sigma=2.29962923,
    polarity=-1.0,
    chirality=1.0,
    x=1.0,
    y=0.0,
    z=0.0):
    
    fr = FRi3D(
        latitude=latitude, 
        longitude=longitude, 
        toroidal_height=toroidal_height, 
        poloidal_height=poloidal_height, 
        half_width=half_width, 
        tilt=tilt, 
        flattening=flattening, 
        pancaking=pancaking, 
        skew=skew, 
        tapering=tapering,
        twist=twist, 
        flux=flux,
        sigma=sigma,
        polarity=polarity,
        chirality=chirality
    )
    fr.init()


    sta_lon = 103.932*np.pi/180.0
    sta_lat = -4.460*np.pi/180.0
    sta_r = 0.967105
    sta_fov = sta_r*AU_RS*np.tan(4.0*np.pi/180.0)
    stb_lon = -97.828*np.pi/180.0
    stb_lat = 1.594*np.pi/180.0
    stb_r = 1.078763
    stb_fov = stb_r*AU_RS*np.tan(4.0*np.pi/180.0)
    soho_lon = 0.0*np.pi/180.0
    soho_lat = 0.0*np.pi/180.0
    soho_r = 1.0
    soho_fov = 32.0

    x0, y0, z0 = fr.shell()
    fig = plt.figure()
    ax = fig.add_subplot(111, 
        projection='3d', 
        adjustable='box', 
        aspect=1.0
    )
    ax.plot_wireframe(x0, y0, z0, color=BLIND_PALETTE['blue'], alpha=0.4)
    plt.show()


    fig = plt.figure()

    gs = gridspec.GridSpec(2, 3)
    gs.update(wspace=0.0, hspace=0.0)

    ax = plt.subplot(gs[0])
    ax.imshow(
        plt.imread('/media/data/Documents/Articles/2016_Isavnin_FRi3D/20130106_103900_dbc2B_opt.png'),
        zorder=0,
        extent=[-stb_fov+0.05, stb_fov+0.05, -stb_fov-0.1, stb_fov-0.1]
    )
    ax.set_xlim([-stb_fov+0.05, stb_fov+0.05])
    ax.set_ylim([-stb_fov-0.1, stb_fov-0.1])
    ax.set_axis_bgcolor('black')
    plt.axis('off')

    ax = plt.subplot(gs[1])
    ax.imshow(
        plt.imread('/media/data/Documents/Articles/2016_Isavnin_FRi3D/20130106_1042_c3_1024_opt.png'),
        zorder=0,
        extent=[-soho_fov+0.3, soho_fov+0.3, -soho_fov+1.33, soho_fov+1.33]
    )
    ax.set_xlim([-soho_fov+0.3, soho_fov+0.3])
    ax.set_ylim([-soho_fov+1.33, soho_fov+1.33])
    ax.set_axis_bgcolor('black')
    plt.axis('off')

    ax = plt.subplot(gs[2])
    ax.imshow(
        plt.imread('/media/data/Documents/Articles/2016_Isavnin_FRi3D/20130106_103900_dbc2A_opt.png'),
        zorder=0,
        extent=[-sta_fov, sta_fov, -sta_fov+0.04, sta_fov+0.04]
    )
    ax.set_xlim([-sta_fov, sta_fov])
    ax.set_ylim([-sta_fov+0.04, sta_fov+0.04])
    ax.set_axis_bgcolor('black')
    plt.axis('off')

    
    ax = plt.subplot(gs[3])
    ax.imshow(
        plt.imread('/media/data/Documents/Articles/2016_Isavnin_FRi3D/20130106_103900_dbc2B_opt.png'),
        zorder=0,
        extent=[-stb_fov-0.03, stb_fov-0.03, -stb_fov-0.0, stb_fov-0.0]
    )
    # ax.plot([0.0], [0.0], '.y', markersize=5.0)
    T = cs.mx_rot_z(-stb_lon)*cs.mx_rot_y(stb_lat)
    x = T[0,0]*x0+T[0,1]*y0+T[0,2]*z0
    y = T[1,0]*x0+T[1,1]*y0+T[1,2]*z0
    z = T[2,0]*x0+T[2,1]*y0+T[2,2]*z0
    y = stb_r/(stb_r-x)*y
    z = stb_r/(stb_r-x)*z
    ax.scatter(y*AU_RS, z*AU_RS, 3, color=BLIND_PALETTE['yellow'], marker='.')
    ax.set_xlim([-stb_fov-0.03, stb_fov-0.03])
    ax.set_ylim([-stb_fov-0.0, stb_fov-0.0])
    ax.set_axis_bgcolor('black')
    plt.axis('off')

    ax = plt.subplot(gs[4])
    ax.imshow(
        plt.imread('/media/data/Documents/Articles/2016_Isavnin_FRi3D/20130106_1042_c3_1024_opt.png'),
        zorder=0,
        extent=[-soho_fov+0.37, soho_fov+0.37, -soho_fov+1.19, soho_fov+1.19]
    )
    # ax.plot([0.0], [0.0], '.y', markersize=5.0)
    T = cs.mx_rot_z(-soho_lon)*cs.mx_rot_y(soho_lat)
    x = T[0,0]*x0+T[0,1]*y0+T[0,2]*z0
    y = T[1,0]*x0+T[1,1]*y0+T[1,2]*z0
    z = T[2,0]*x0+T[2,1]*y0+T[2,2]*z0
    y = soho_r/(soho_r-x)*y
    z = soho_r/(soho_r-x)*z
    ax.scatter(y*AU_RS, z*AU_RS, 3, color=BLIND_PALETTE['yellow'], marker='.')
    ax.set_xlim([-soho_fov+0.37, soho_fov+0.37])
    ax.set_ylim([-soho_fov+1.19, soho_fov+1.19])
    ax.set_axis_bgcolor('black')
    plt.axis('off')

    
    ax = plt.subplot(gs[5])
    ax.imshow(
        plt.imread('/media/data/Documents/Articles/2016_Isavnin_FRi3D/20130106_103900_dbc2A_opt.png'),
        zorder=0,
        extent=[-sta_fov+0.1, sta_fov+0.1, -sta_fov+0.04, sta_fov+0.04]
    )
    # ax.plot([0.0], [0.0], '.y', markersize=5.0)
    T = cs.mx_rot_z(-sta_lon)*cs.mx_rot_y(sta_lat)
    x = T[0,0]*x0+T[0,1]*y0+T[0,2]*z0
    y = T[1,0]*x0+T[1,1]*y0+T[1,2]*z0
    z = T[2,0]*x0+T[2,1]*y0+T[2,2]*z0
    y = sta_r/(sta_r-x)*y
    z = sta_r/(sta_r-x)*z
    ax.scatter(y*AU_RS, z*AU_RS, 3, color=BLIND_PALETTE['yellow'], marker='.')
    ax.set_xlim([-sta_fov+0.1, sta_fov+0.1])
    ax.set_ylim([-sta_fov+0.04, sta_fov+0.04])
    ax.set_axis_bgcolor('black')
    ax.patch.set_facecolor('black')
    plt.axis('off')

    plt.show()