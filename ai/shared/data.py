
import os
from scipy.io import readsav
from datetime import datetime
import time
import numpy as np
from astropy import units as u
import calendar


u.nT = u.def_unit('nT', 1e-9*u.T)

DATADIR = 'data'
MES_SAV = os.path.join(DATADIR, 'MES_2007to2014_HEEQ.sav')
VEX_SAV = os.path.join(DATADIR, 'VEX_2007to2014_HEEQ_removed.sav')
STA_SAV = os.path.join(DATADIR, 'STA_2007to2015_HEEQ.sav')
STB_SAV = os.path.join(DATADIR, 'STB_2007to2014_HEEQ.sav')
WIND_SAV = os.path.join(DATADIR, 'WIND_2007to2015_HEEQ.sav')

def _getData(datetime0, datetime1, sc):

    if sc == 'vex':
        DATA_SAV = VEX_SAV
    elif sc == 'sta':
        DATA_SAV = STA_SAV
    elif sc == 'stb':
        DATA_SAV = STB_SAV
    elif sc == 'mes':
        DATA_SAV = MES_SAV
    elif sc == 'wind':
        DATA_SAV == WIND_SAV
    else:
        print('wrong spacecraft')

    fp = os.path.join(
        DATADIR,
        '_'.join([
                sc,
                datetime0.strftime('%Y%m%d%H%M%S'),
                datetime1.strftime('%Y%m%d%H%M%S')
            ])+'.npz'
    )

    if not os.path.isfile(fp):
        data = readsav(DATA_SAV, python_dict=True)
        t = data[sc]['time']+calendar.timegm(datetime(1979, 1, 1).timetuple())
        mask = np.logical_and(
            t >= calendar.timegm(datetime0.timetuple()), 
            t <= calendar.timegm(datetime1.timetuple())
        )
        t = t[mask]
        t = np.array([datetime.utcfromtimestamp(x) for x in t])
        b = np.stack([
            data[sc]['bx'][mask],
            data[sc]['by'][mask],
            data[sc]['bz'][mask]
        ], axis=1)*u.nT.to(u.T)
        if 'vtot' in data[sc]:
            v = data[sc]['vtot'][mask]*u.Unit("km/s").to(u.Unit("m/s"))
        else:
            v = np.nan*np.ones(t.shape)
        r = data[sc][sc+'_radius_in_km_heeq'][mask]*1e3
        theta = data[sc][sc+'_latitude_in_radians_heeq'][mask]
        phi = data[sc][sc+'_longitude_in_radians_heeq'][mask]
        x = r*np.cos(theta)*np.cos(phi)
        y = r*np.cos(theta)*np.sin(phi)
        z = r*np.sin(theta)
        p = np.stack([x, y, z], axis=1)

        mask = np.logical_not(np.logical_and.reduce(
            [np.isnan(b[:,0]), np.isnan(b[:,1]), np.isnan(b[:,2])]
        ))
        t = t[mask]
        b = b[mask,:]
        v = v[mask]
        p = p[mask,:]

        np.savez(fp, t=t, b=b, v=v, p=p)

    data = np.load(fp)
    return (data['t'], data['b'], data['v'], data['p'])

def getVEX(datetime0, datetime1):
    return _getData(datetime0=datetime0, datetime1=datetime1, sc='vex')

def getMES(datetime0, datetime1):
    return _getData(datetime0=datetime0, datetime1=datetime1, sc='mes')

def getSTA(datetime0, datetime1):
    return _getData(datetime0=datetime0, datetime1=datetime1, sc='sta')

def getSTB(datetime0, datetime1):
    return _getData(datetime0=datetime0, datetime1=datetime1, sc='stb')

def getWIND(datetime0, datetime1):
    return _getData(datetime0=datetime0, datetime1=datetime1, sc='wind')
