# seaice extent spiralling raph
# inspired by global temperature change of Ed Hawkin: https://twitter.com/ed_hawkins/status/729753441459945474
#
#
#

import os
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

# variable
area = ['Arctic', 'Antarctic']
data_out = os.path.join(os.path.expanduser('~'), 'Sea Ice Death Spiral')
flag_save = False
fig_num = 0

# parameter
data_url = {'Arctic': 'ftp://sidads.colorado.edu/DATASETS/NOAA/G02135/north/daily/data/N_seaice_extent_daily_v3.0.csv',
            'Antarctic': 'ftp://sidads.colorado.edu/DATASETS/NOAA/G02135/south/daily/data/S_seaice_extent_daily_v3.0.csv'}

if not isinstance(area, list):
    AREA = list(area)
else:
    AREA = area

os.makedirs(data_out, exist_ok=True)


# function
def year_frac2(row):
    """

    :param row:
    :return:
    """
    y = int(row['Year'])
    m = int(row['Month'])
    d = int(row['Day'])
    if np.isnan([y, m, d]).any():
        return np.nan
    else:
        return datetime.datetime(y, m, d).timetuple().tm_yday/datetime.datetime(y, 12, 31).timetuple().tm_yday*2*np.pi


# download data from NSIDC https://nsidc.org/data/seaice_index/archives.html
data = {}
NPOINTS = {}
color = {}
cm = plt.get_cmap('jet')

for area in AREA:
    data[area] = pd.read_csv(data_url[area], header=[0, 1], delimiter=',', skipinitialspace=True)
    unit = dict(data[area].columns.values)
    data[area].columns = data[area].columns.droplevel(-1)
    data[area]['Date'] = data[area][['Year', 'Month', 'Day']].apply(lambda s : datetime.datetime(*s),axis = 1)
    data[area].set_index('Date', drop=True, inplace=True)
    data[area].resample('D').mean().interpolate(inplace=True)

    data[area]['degree'] = data[area].apply(year_frac2, axis=1)

NPOINTS = data[area].degree.__len__()
color = [cm(ii/(NPOINTS-1)) for ii in range(NPOINTS-1)]

fig = plt.figure(fig_num)
fig_num +=1
ax = {}
ii_area = 1
for area in AREA:
    ax[area] = fig.add_subplot(1, AREA.__len__(), ii_area, projection='polar')
    ii_area += 1
import time
freq = 100
for ii in range(int(NPOINTS/freq-1)):
    plt.suptitle(datetime.datetime(data[area].Year[ii*freq], data[area].Month[ii*freq], data[area].Day[ii*freq]).strftime("%Y %b %d"))
    for area in AREA:
        if ii == 0:
            ax[area].set_title(area)
        ax[area].plot(data[area].degree[ii*freq:((ii+1)*freq+1)], data[area].Extent[ii*freq:((ii+1)*freq+1)], color = color[ii*freq])
        ax[area].set_thetagrids(np.linspace(360/24, 360*23/24, 12))
        ax[area].set_theta_direction('clockwise')
        ax[area].set_theta_offset(np.pi/2)
        ax[area].xaxis.set_ticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax[area].set_rmin(1)
        ax[area].set_rmax(max(data[area].Extent)*1.1)
        #ax[area].text(4.5, 14, datetime.datetime(2022, 12, 29).strftime("%Y %b %d"), fontsize=15, color='w', bbox={'facecolor':'w', 'alpha':1, 'pad':5, 'edgecolor':'w'})
        #ax[area].text(4.5, 14, datetime.datetime(data[area].Year[ii*freq], data[area].Month[ii*freq], data[area].Day[ii*freq]).strftime("%Y %b %d"), fontsize = 15 )
    plt.draw()
    fig.canvas.draw()
    time.sleep(freq/1000)
    if flag_save:
        plt.savefig(os.path.join(data_out, datetime.datetime(data[area].Year[ii*freq], data[area].Month[ii*freq], data[area].Day[ii*freq]).strftime("%Y%m%d")+'.png'), dpi=75)
