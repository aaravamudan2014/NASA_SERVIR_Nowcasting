from load_data import *
from forecasts import *
from evaluation_metrics import FSS_plot, save_tiff_files
from visualization import create_animation, create_precipitation_prediction_plots

import pandas as pd

def main(event_name):
    # load data
    init_IMERG_config_pysteps()
    precipitation, times = load_IMERG_data_tiff(data_location='data/'+event_name)

    sorted_precipitation, sorted_timestamps = sort_IMERG_data(precipitation, times)

    if len(sorted_precipitation) == 0:
        print("Imerg data is empty, ignoring this event")
        return None
    # Iterate through start indices (determines "how much" data is used for cosntructing motion vectors (training)
    for start_index in range(10,12, 1):
        
        prediction_timesteps = 10 #5 hours = 10 timesteps * 30 minutes per time step

        # precipitation[0:5] -> Used to find motion (past data). Let's call it training precip.
        train_precip = sorted_precipitation[0:start_index]

        # Used to evaluate forecasts (future data, not available in "real" forecast situation)
        # Let's call it observed precipitation because we will use it to compare our forecast with the actual observations.
        observed_precip = sorted_precipitation[start_index: start_index + prediction_timesteps]

        train_precip_dbr = transform_data_dBR(train_precip)

        # Generate motion field via Lukas-Kanade
        motion_field = getMotionField(train_precip_dbr)
        
        # Generate forecasts
        linda_forecast = LINDAForecast(train_precip, observed_precip,motion_field)
        precip_forecast = LagrangianPersistenceForecast(train_precip, observed_precip,motion_field)
        persistence_forecast = NaivePersistence(train_precip, observed_precip)
        steps_forecast = StepsForecast(train_precip, observed_precip,motion_field)
        
        forecasts = [precip_forecast, persistence_forecast, linda_forecast, steps_forecast]

        methods = ['Naive Persistence', 'Lagrangian Persistence','LINDA' , 'STEPS']

        # Evaluation function
        scores = FSS_plot(forecasts, observed_precip, methods, start_index, event_identifier=event_name)
        
        # create a png with a list of predictions (stored in figures/**event_name**/)
        create_precipitation_prediction_plots(observed_precip,methods, forecasts, sorted_timestamps[start_index], event_identifier=event_name)
        
        # create an animated gif for the predictions (stored in figures/**event_name**/)
        create_animation(observed_precip, forecasts, sorted_timestamps[start_index], event_identifier=event_name, methods=methods)

        # saves the tiff files of the predictions (stored in figures/**event_name**/**start_time**/)
        save_tiff_files(forecasts, sorted_timestamps[start_index],event_identifier=event_name, methods=methods, best_method= 'LINDA')
        print("Results calculated for one training setup instance")
        

if __name__ == "__main__":
    # read all the events from the csv file
    event_data_df = pd.read_csv('data/EF5events.csv')
    for index, row in event_data_df.iterrows():
        event_name = row['Country'] + '_' + row['Date'].replace('/', '_')
        main(event_name)


