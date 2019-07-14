# seaice extent spiralling raph
# inspired by global temperature change of Ed Hawkin: https://twitter.com/ed_hawkins/status/729753441459945474
# V2. add background
#
#

import os
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

# variable
AREA = ['Arctic']
FREQ = 50  # days
FIG_TYPE = 'conc_hires'

flag_save = True
flag_show = False
dates = None
fig_size = [4, 4]
dpi = 300
#dates = [pd.datetime(1995, 6, 7), data[area].index.max()]

# data folder
data_dir = '/mnt/data/UAF-data/NSIDC/seaice_index/'
data_out = '/mnt/data/UAF-data/seaice/deathspiral'
data_subdir = {'Arctic': 'north/daily', 'Antarctic': 'south/daily'}
data_dict = {'Arctic': ['N'], 'Antarctic': ['S']}
fig_dict = {'extn_hires': ['_extn_hires_v3.0.png', [592, 915], [102, 1660], [57, 1111]],
            'conc_hires': ['_conc_hires_v3.0.png', [592, 915], [102, 1660], [57, 1111]]}

if isinstance(flag_show, int):
    update_time = flag_show
else:
    update_time = 500

# parameter
if not isinstance(AREA, list):
    AREA = list(AREA)
else:
    AREA = AREA

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

def create_circular_mask(h, w, center=None, radius=None, layer=1):
    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    mask = np.dstack([mask]*layer)
    return mask

# download data from NSIDC https://nsidc.org/data/seaice_index/archives.html
data = {}

for area in AREA:
    data_path = os.path.join(data_dir, data_subdir[area], 'data/N_seaice_extent_daily_v3.0.csv')
    data[area] = pd.read_csv(data_path, header=[0, 1], delimiter=',', skipinitialspace=True)
    unit = dict(data[area].columns.values)
    data[area].columns = data[area].columns.droplevel(-1)
    data[area]['Date'] = data[area][['Year', 'Month', 'Day']].apply(lambda s : datetime.datetime(*s),axis = 1)
    data[area].set_index('Date', drop=True, inplace=True)
    gap = data[area].resample('1D').mean()['Extent'].isna()
    data[area].resample('1D').mean().interpolate(inplace=True)
    data[area]['gap'] = gap
    data[area]['degree'] = data[area].apply(year_frac2, axis=1)
    data[area]['t_elapsed'] = data[area].index - data[area].index.min()
    data[area]['t_elapsed'] = data[area]['t_elapsed'].apply(lambda x : x.days)

# Figures:
cm = plt.get_cmap('jet')
rect = [0.05, 0.05, 0.9, 0.825]  # setting the axis limits in [left, bottom, width, height]

if flag_show:
    plt.ion()

for area in AREA:
    if dates is None:
        dates = pd.date_range(data[area].index.min(), data[area].index.max(),
                              freq=str('%.0f' % FREQ) + 'D').to_pydatetime()

    t_N = np.ceil(data[area].t_elapsed.max()/FREQ+1).astype(int)
    color_freq = [cm(ii/t_N) for ii in range(t_N)]
    date_previous = dates.min()
    for date in dates:
        if flag_save:
            fig_dir = os.path.join(data_out, area, FIG_TYPE)
            fig_output = os.path.join(fig_dir, area+date.strftime("-%Y%m%d-")+FIG_TYPE+'.png')
            if not os.path.exists(fig_dir):
                os.makedirs(fig_dir)
        if os.path.exists(fig_output) or date == dates.min():
            continue
        print(date)

        if FIG_TYPE is not None:
            xc = int(fig_dict[FIG_TYPE][1][0])
            yc = int(fig_dict[FIG_TYPE][1][1])
            dx = int(np.abs(np.diff(fig_dict[FIG_TYPE][3]))[0])
            dy = int(np.abs(np.diff(fig_dict[FIG_TYPE][2]))[0])
            ds = int(np.floor(min(dx, dy) / 2))

            # circular mask
            mask = create_circular_mask(h=2*ds, w=2*ds, center=[ds, ds], radius= ds-1, layer=4)

            fig_name = data_dict[area][0]+'_'+pd.datetime.strftime(date, '%Y%m%d')+fig_dict[FIG_TYPE][0]
            fig_path = os.path.join(data_dir, data_subdir[area], 'images', str('%02.0f' % date.year), str('%02.0f' % date.month)
                                   + '_'+ pd.datetime.strftime(date, '%b'), fig_name)

            if data[area].loc[data[area].index == date, 'gap'].all():
                if 7 < (date - date_previous).days:
                    img = np.ones([2 * ds, 2 * ds, 4])
            else:
                try:
                    img = plt.imread(fig_path)
                except FileNotFoundError:
                    img = np.ones([2*ds, 2*ds, 4])
                    mask_img = img
                else:
                    img_bkp = img.copy()
                    img = img[:, xc-ds:xc+ds, :]
                    img = img[yc-ds:yc+ds, :, :]

                    if img.shape == mask.shape:
                        mask_img = np.where(mask == 1, img, 0)
                    else:
                        img = np.ones([2 * ds, 2 * ds, 4])
                        mask_img = img
                date_previous = date

            # compute new fig size
            fig_size = [2.4*ds/dpi, 2.4*ds/dpi]

        # generate figures
        fig = plt.figure(figsize=fig_size)  # initializing the figure
        ax_c = fig.add_axes(rect)  # the carthesian axis
        ax_p = fig.add_axes(rect, polar=True, frameon=False)  # the polar axis
        ax_c.imshow(mask_img)

        try:
            t_elapsed = data[area].loc[data[area].index == date].t_elapsed.values[0]
        except IndexError:
            t_elapsed = 0
        t_step = np.arange(0, t_elapsed, FREQ)
        t_step = np.concatenate([t_step, [t_elapsed]])

        for ii, t in enumerate(t_step[:-1]):
            ax_p.plot(data[area].degree[t:t_step[ii+1]], data[area].Extent[t:t_step[ii+1]], color=color_freq[ii], alpha=1)

        # cosmetic
        ax_c.xaxis.set_visible(False)
        ax_c.yaxis.set_visible(False)
        ax_c.spines['right'].set_visible(False)
        ax_c.spines['left'].set_visible(False)
        ax_c.spines['top'].set_visible(False)
        ax_c.spines['bottom'].set_visible(False)


        ax_p.set_thetagrids(np.linspace(360/24, 360*23/24, 12))
        ax_p.set_theta_direction('clockwise')
        ax_p.set_theta_offset(np.pi/2)
        ax_p.xaxis.set_ticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax_p.set_rmin(1)
        ax_p.set_rmax(max(data[area].Extent)*1.1)
        ax_c.text(0.05, 0.95, area.capitalize() + ' sea-ice extent', ha='left', transform=plt.gcf().transFigure)
        ax_c.text(0.95, 0.95, date.strftime('%-d'), ha='right', transform=plt.gcf().transFigure)
        ax_c.text(0.90, 0.95, date.strftime('%B %Y,'), ha='right', transform=plt.gcf().transFigure)
        ax_c.text(0.95, 0.02, 'data: NSIDC', ha='right', transform=plt.gcf().transFigure)

        if flag_save:
            plt.savefig(fig_output, dpi=125)
            plt.close()
        if flag_show:
            plt.draw()
            fig.canvas.draw()
            import time
            time.sleep(update_time)
if flag_show:
    plt.show()
