import os
import glob
import time

import pandas as pd
from osgeo import gdal
from load_data import init_IMERG_config_pysteps, load_IMERG_data_tiff, sort_IMERG_data
from forecasts import lp_nowcast, linda_nowcast, steps_nowcast
from visualization import  create_precipitation_gif
from pysteps import verification


gif_plot = False

timeLog = open("results/times.txt","a")


methods_dict = {
                'STEPS': {'func': steps_nowcast, 'kargs': {'n_ens_members': 20, 'n_cascade_levels': 6}}, \
                'LINDA': {'func': linda_nowcast, 'kargs': {'max_num_features': 15, 'add_perturbations': False}}, \
                'Lagrangian_Persistence': {'func': lp_nowcast, 'kargs': {}},\
                }


setups = [{'train_st_ind':16, 'train_ed_ind':24, 'prediction_steps': 24},\
          {'train_st_ind':0, 'train_ed_ind':24, 'prediction_steps': 24}]


# event_data_df = pd.read_csv('data/EF5events.csv')

# event_names = []
# for index, row in event_data_df.iterrows():
#     event_name = row['Country'] + '_' + row['Date'].replace('/', '_')
#     event_names.append(event_name)
  
event_names = ["Côte d'Ivoire_18_06_2018", "Cote d'Ivoire_25_06_2020", 'Ghana _10_10_2020', 'Nigeria_18_06_2020']

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

# FSS score
# calculate FSS
fss = verification.get_method("FSS")

thr=1.0
scale=2


fss_scores = []


for event_name in event_names:

    # load data
    init_IMERG_config_pysteps()
    precipitation, times = load_IMERG_data_tiff(data_location='data/'+event_name)
    sorted_precipitation, sorted_timestamps = sort_IMERG_data(precipitation, times)


    # observed precipitation .gif creation
    path_outputs = 'results/'+event_name
    
    if not os.path.isdir(path_outputs):
        os.mkdir(path_outputs)


    for setup in setups:

        train_timesteps = sorted_timestamps[setup['train_st_ind']: setup['train_ed_ind']]
        pred_timesteps  = sorted_timestamps[setup['train_ed_ind']: setup['train_ed_ind'] + setup['prediction_steps']]

        obj = dict.fromkeys(methods_dict.keys())

        obj['pred_timesteps'] = pred_timesteps
        obj['train_timesteps'] = train_timesteps
        obj['event'] = event_name


        train_precip = sorted_precipitation[setup['train_st_ind']: setup['train_ed_ind']]
        observed_precip = sorted_precipitation[setup['train_ed_ind']: setup['train_ed_ind'] + setup['prediction_steps']]

        if gif_plot == True:

            gif_title = "observed precipitation" #f"observed precipitation {pred_timesteps[0]} -- {pred_timesteps[-1]}"

            create_precipitation_gif(observed_precip, pred_timesteps, 30, metadata, path_outputs, gif_title, gif_dur = 1000)
        
        for method in methods_dict.keys():
            paras = methods_dict[method]
            pfunc = paras['func'] 
            kargs = paras['kargs']
            #==========Forcast===========
            steps_st = time.time()
            forcast_precip = pfunc(train_precip, setup['prediction_steps'], **kargs)
            steps_ed = time.time()

            # log running time
            timeLog.write(f"{event_name}: {method} nowcast with {setup['train_ed_ind'] - setup['train_st_ind']} training steps to predict {setup['prediction_steps']} steps takes {(steps_ed - steps_st)/60} mins \n")
            
            # Calculate the FSS for every lead time and all predefined scales.
            scores = []
            for i in range(setup['prediction_steps']):
                scores.append(fss(forcast_precip[i, :, :], observed_precip[i, :, :], thr=thr, scale=scale))

            # save fss scores
            obj[method] = scores

            if gif_plot == True:
                gif_title = f"{method} -- {int((setup['train_ed_ind'] - setup['train_st_ind'])/2)}-hour training"
                create_precipitation_gif(forcast_precip, pred_timesteps, 30, metadata, path_outputs, gif_title, gif_dur = 1000)

        fss_scores.append(obj)


timeLog.close() #to change file access modes

fss_scores_df = pd.DataFrame(fss_scores)

fss_scores_df.to_csv('results/fss_scores.csv')



