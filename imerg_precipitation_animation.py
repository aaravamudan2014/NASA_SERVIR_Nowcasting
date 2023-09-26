import os
import glob
from osgeo import gdal

import pandas as pd
import matplotlib.pyplot as plt
import imageio
from load_data import *
from forecasts import *


from pysteps.visualization.animations import animate

event_data_df = pd.read_csv('data/EF5events.csv')

event_names = []
for index, row in event_data_df.iterrows():
    event_name = row['Country'] + '_' + row['Date'].replace('/', '_')
    event_names.append(event_name)
  
# event_name = "Côte d'Ivoire_18_06_2018"

metadata = {'accutime': 30.0,
    'cartesian_unit': 'degrees',
    'institution': 'NOAA National Severe Storms Laboratory',
    'projection': '+proj=longlat  +ellps=IAU76',
    'threshold': 0.0125,
    'timestamps': None,
    'transform': None,
    'unit': 'mm/h',
    'x1': -21.4,
    'x2': 30.4,
    'xpixelsize': 0.04,
    'y1': -2.9,
    'y2': 33.1,
    'yorigin': 'upper',
    'ypixelsize': 0.04,
    'zerovalue': 0}

# load data
init_IMERG_config_pysteps()

for event_name in event_names:
    precipitation, times = load_IMERG_data_tiff(data_location='data/'+event_name)

    sorted_precipitation, sorted_timestamps = sort_IMERG_data(precipitation, times)

    metadata['timestamps'] = sorted_timestamps
    # precipitations = np.stack(sorted_precipitation, axis=0)
    precipitations = sorted_precipitation
    path_outputs = 'precipitation_images/'+event_name

    if not os.path.isdir(path_outputs):
        os.mkdir(path_outputs)

    animate(precipitations, timestamps_obs  = sorted_timestamps, timestep_min = 30, geodata=metadata,
            savefig=True, fig_dpi=100, fig_format='png', path_outputs='precipitation_images/'+event_name)
    


    images = []
    precipitation_imgs = sorted(glob.glob(path_outputs+'/*.png'))


    for filename in precipitation_imgs:
        images.append(imageio.imread(filename))


    kargs = { 'duration': 10 }
    imageio.mimsave(path_outputs + '/all_day.gif', images, **kargs)