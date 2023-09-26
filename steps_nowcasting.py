import os
import glob
from osgeo import gdal

import pandas as pd
import matplotlib.pyplot as plt
import imageio
from load_data import *
from forecasts import *


from pysteps.visualization.animations import animate

# event_data_df = pd.read_csv('data/EF5events.csv')

# event_names = []
# for index, row in event_data_df.iterrows():
#     event_name = row['Country'] + '_' + row['Date'].replace('/', '_')
#     event_names.append(event_name)
  
event_name = "Côte d'Ivoire_18_06_2018"
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


setups = [{'train_st_ind':0, 'train_ed_ind':8, 'prediction_steps': 24},\
          {'train_st_ind':0, 'train_ed_ind':24, 'prediction_steps': 24}]


init_IMERG_config_pysteps()
precipitation, times = load_IMERG_data_tiff(data_location='data/'+event_name)

sorted_precipitation, sorted_timestamps = sort_IMERG_data(precipitation, times)

for setup in setups:

    train_precip = sorted_precipitation[setup['train_st_ind']: setup['train_ed_ind']]

    observed_precip = sorted_precipitation[setup['train_ed_ind']: setup['train_ed_ind'] + setup['prediction_steps']]

    R_train, metadata = transformation.dB_transform(train_precip, metadata, threshold=0.1, zerovalue=-15.0)

    # Set missing values with the fill value
    R_train[~np.isfinite(R_train)] = -15.0

    # Estimate the motion field
    V = dense_lucaskanade(R_train)

    # The STEPS nowcast
    n_ens_members = 20
    nowcast_method = nowcasts.get_method("steps")
    R_forcast = nowcast_method(R_train, V, setup['prediction_steps'],
                        n_ens_members,
                        n_cascade_levels=6,
                        precip_thr  = -10.0,
                        kmperpixel=2.0,
                        timestep=30,
                        seed=24,
                        )

    # Back-transform to rain rates
    R_forcast = transformation.dB_transform(R_forcast, threshold=-10.0, inverse=True)[0]

    # Get the ensemble mean
    R_f_mean = np.mean(R_forcast, axis=0)
