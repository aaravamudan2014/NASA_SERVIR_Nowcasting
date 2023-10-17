import os
import glob
import time
import json

import pandas as pd
from osgeo import gdal
from servir.utils import WriteGrid, processIMERG, init_IMERG_config_pysteps, load_IMERG_data_tiff, sort_IMERG_data
from servir.forecasts import lp_nowcast, linda_nowcast, steps_nowcast
from servir.visualization import  create_precipitation_gif
from matplotlib import pyplot as plt
from pysteps import verification


gif_plot = False

timeLog = open("results/times.txt","a")


methods_dict = {
                # 'STEPS': {'func': steps_nowcast, 'kargs': {'n_ens_members': 20, 'n_cascade_levels': 6}}, \
                'LINDA': {'func': linda_nowcast, 'kargs': {'max_num_features': 15, 'add_perturbations': False}}, \
                # 'Lagrangian_Persistence': {'func': lp_nowcast, 'kargs': {}},\
                }


setups = [{'train_st_ind':0, 'train_ed_ind':16, 'prediction_steps': 16},\
        #   {'train_st_ind':0, 'train_ed_ind':24, 'prediction_steps': 24},\
         ]


# event_data_df = pd.read_csv('data/EF5events.csv')

# event_names = []
# for index, row in event_data_df.iterrows():
#     event_name = row['Country'] + '_' + row['Date'].replace('/', '_')
#     event_names.append(event_name)
  
event_names = ["Côte d'Ivoire_18_06_2018", "Cote d'Ivoire_25_06_2020", 'Ghana _10_10_2020', 'Nigeria_18_06_2020']

xmin = -21.4
xmax = 30.4
ymin = -2.9
ymax = 33.1

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

        # get geotiff metadata
        geo_tiff_metadata =[]
        for dt in pred_timesteps:
            # '3B-HHR-E.MS.MRG.3IMERG.20180618-S123000-E125959.0750.V06B.30min'
            f_str = glob.glob(f'data/{event_name}/raw_imerg/3B-HHR-E.MS.MRG.3IMERG.{dt.strftime("%Y%m%d")}-S{dt.strftime("%H%M%S")}*.tif')[0]
            # f_str = f'data/{event_name}/processed_imerg/imerg.{dt.strftime("%Y%m%d%H%M")}.30minAccum.tif'
            _, nx, ny, gt, proj = processIMERG(f_str,xmin,ymin,xmax,ymax)

            # WriteGrid(gridOutName, NewGrid, nx, ny, gt, proj)
            geo_tiff_metadata.append([nx, ny, gt, proj])

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
            
            # save to geotiff file
            if method=="LINDA":
                for i, dt in enumerate(pred_timesteps):
                    nx, ny, gt, proj = geo_tiff_metadata[i]
                    f_str = f'results/tifs/linda_forcast_{event_name}_{dt.strftime("%Y%m%d%H%M")}.tif'
                    WriteGrid(f_str, forcast_precip[i,:,:], nx, ny, gt, proj)
    
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

# with open("results/fss_scores.json", "w") as f:
#     json.dump(fss_scores, f)


fss_df = pd.DataFrame(fss_scores)


methods = ['Lagrangian_Persistence', 'LINDA', 'STEPS']

# plot per setup
for _, row in fss_df.iterrows():
    train_steps = len(row['train_timesteps'])
    # re-do by train_steps[-1] - train_steps[0]

    plt.figure()
    # shorten to time only
    x = [t.strftime("%H:%M") for t in row['pred_timesteps']]

    for method in methods: 
        plt.plot(x, row[method], label=method, lw=2.0)
    
    plt.xlabel("Lead time [min]")
    plt.xticks(rotation=90)
    plt.ylabel("FSS")
    plt.title(f"{row['event']} - {int(train_steps/2)}-hour train")
    plt.legend(
        title="method",
        # loc="center left",
        # bbox_to_anchor=(1.01, 0.5),
        bbox_transform=plt.gca().transAxes,
    )
    # plt.autoscale(axis="x", tight=True)
    plt.savefig(f"results/fss/{row['event']} - {int(train_steps/2)}-hour train", transparent=False,  bbox_inches="tight")

# compare 4-hour training and 12-hour training
events = fss_df['event'].unique()
for event in events:
    event_df = fss_df.loc[fss_df['event'] == event] 

    for method in methods:

        plt.figure()
        for _, row in event_df.iterrows():
            train_steps = len(row['train_timesteps'])
            # re-do by train_steps[-1] - train_steps[0]
            # shorten to time only
            x = [t.strftime("%H:%M") for t in row['pred_timesteps']]

            plt.plot(x, row[method], label=f"{int(train_steps/2)}-hour", lw=2.0)
        
            plt.xlabel("Lead time [min]")
            plt.xticks(rotation=90)
            plt.ylabel("FSS")
            plt.title(f"{row['event']} - {method}")
            plt.legend(
                title="train hours",
                # loc="center left",
                # bbox_to_anchor=(1.01, 0.5),
                bbox_transform=plt.gca().transAxes,
            )
            # plt.autoscale(axis="x", tight=True)
            plt.savefig(f"results/fss/{row['event']} - {method}", transparent=False,  bbox_inches="tight")




fss_df.to_csv('results/fss_scores.csv', index=False)



