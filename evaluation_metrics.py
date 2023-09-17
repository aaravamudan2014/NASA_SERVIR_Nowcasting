from pysteps import verification
import matplotlib.pyplot as plt
import numpy as np
from pysteps.visualization import plot_precip_field
import matplotlib.animation as animation
from matplotlib import rc
import datetime
import os


def FSS_plot(forecasts, observed_precip, methods, start_index, event_identifier):
    """Function for generating Fraction Skill SCore (FSS) scores for passed forecasts

    Args:
        forecasts (tuple): tuple of forecasts, this show follow the same order as the variable methods passed in
        observed_precip (np.array): ground truth  precipitation data
        methods (list of str): list of methods
        start_index (int): Start index in the training data (used for filenames while saving)
        event_identifier (str): Name of the event

    Returns:
        _type_: _description_
    """

    for index, method in enumerate(methods):
        if method == 'Naive Persistence':
            persistence_forecast = forecasts[index]
        if method == 'Lagrangian Persistence':
            precip_forecast = forecasts[index]
        if method == 'LINDA':
            linda_forecast = forecasts[index]
        if method == 'STEPS':
            steps_forecast = forecasts[index]
        
    fss = verification.get_method("FSS")

    # Compute fractions skill score (FSS) for all lead times for different scales using a 1 mm/h detection threshold.
    scales = [
        2,
        # 4,
        # 8,
        # 16,
        # 32,
        # 64,
    ]  # In grid points.

    scales_in_km = np.array(scales) * 10

    # Set the threshold
    thr = 1.0  # in mm/h

    os.makedirs('figures/'+event_identifier, exist_ok=True)
    # Now plot it
    plt.figure(figsize=(10,10))
    # Calculate the FSS for every lead time and all predefined scales.
    for method in methods:
        score = []
        if method == 'Naive Persistence':
            prediction_forecast = persistence_forecast
        elif method == 'Lagrangian Persistence':
            prediction_forecast = precip_forecast
        elif method == 'LINDA':
            prediction_forecast = linda_forecast
        elif method == 'STEPS':
            prediction_forecast = steps_forecast

        for i in range(len(observed_precip)):
            score_ = []
            for scale in scales:
                score_.append(
                    fss(prediction_forecast[i, :, :], observed_precip[i, :, :], thr, scale)
                )
            score.append(score_)
        x = np.arange(1, len(observed_precip) + 1)
        plt.plot(x*30, score, lw=2.0, label=method)

    plt.xlabel("Lead time [min]")
    plt.ylabel("FSS ( > 1.0 mm/h ) ")
    plt.title("Fractions Skill Score")
    plt.legend(
        # scales_in_km,
        # title="Scale [km]",
        # loc="center left",
        # bbox_to_anchor=(1.01, 0.5),
        # bbox_transform=plt.gca().transAxes,
    )
    plt.autoscale(axis="x", tight=True)
    plt.tight_layout()
    plt.savefig('figures/'+event_identifier+'/FSS_comparison_'+str(start_index)+'.png')
    return score
import osgeo.gdal as gdal
import osgeo.osr as osr
from osgeo.gdal import gdalconst
from osgeo.gdalconst import GA_ReadOnly

from data.early_run_IMERG_download import *

def save_tiff_files( forecasts, start_time, event_identifier, methods, best_method):
    """ This function generates tiff files and saves them in the appropriate directory. 
        The directory will be results/**event_name**/**start_time**

    Args:
        forecasts (list): list containing all the forecasts, it has to follow the same order as methods
        start_time (DateTime): Start time of the prediction 
        event_identifier (str): Name of the event
        methods (list of str):  list ofnowcasting methods
        best_method (str) Method chosen for saving (typically the best method)
    """
    for index, method in enumerate(methods):
        if method == 'Naive Persistence':
            persistence_forecast = forecasts[index]
        if method == 'Lagrangian Persistence':
            precip_forecast = forecasts[index]
        if method == 'LINDA':
            linda_forecast = forecasts[index]
        if method == 'STEPS':
            steps_forecast = forecasts[index]
        
    for model_index in range(len(forecasts)):
        if model_index == 0:
            predicted_forecast = precip_forecast
            model_name = 'lagrangian'
        if model_index == 1:
            predicted_forecast = persistence_forecast
            model_name = 'persistence'
        if model_index == 2:
            predicted_forecast = linda_forecast
            model_name = 'LINDA'
        if model_index == 3:
            predicted_forecast = steps_forecast
            model_name = 'STEPS'

        os.makedirs('results/'+event_identifier, exist_ok=True)

        if best_method == 'STEPS':
            best_model_forecast  = steps_forecast   
        elif best_method == 'LINDA':
            best_model_forecast  = linda_forecast   
        elif best_method == 'lagrangian':
            best_model_forecast  = precip_forecast 
        elif best_method == 'persistence':
            best_model_forecast  = persistence_forecast 
          
        # Domain coordinates
        xmin = -21.4
        xmax = 30.4
        ymin = -2.9
        ymax = 33.1
  
        for index, predicted_observation in enumerate(best_model_forecast):
            os.makedirs('results/'+event_identifier+'/'+str(start_time), exist_ok=True)
            filename = 'results/' +event_identifier+'/'+ str(start_time) +'/'+ str(start_time + datetime.timedelta(minutes= int(index) *30 )) +'.tif'
            drv = gdal.GetDriverByName("GTiff")
            height = predicted_observation.shape[0]
            width = predicted_observation.shape[1]
            ds = drv.Create(filename, width, height, 6, gdal.GDT_Float32)
            ds.SetGeoTransform([xmin, height,0, ymin,0, -width])
            ds.SetProjection('+proj=longlat  +ellps=IAU76')
            ds.GetRasterBand(1).WriteArray(predicted_observation)