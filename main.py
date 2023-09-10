from load_data import *
from forecasts import *
from evaluation_metrics import FSS_plot, create_animation, create_precipitation_prediction_plots,save_tiff_files
import pandas as pd

def main(data_location):
    # load data
    init_IMERG_config_pysteps()
    precipitation, times = load_IMERG_data_tiff(data_location=data_location)

    sorted_precipitation, sorted_timestamps = sort_IMERG_data(precipitation, times)
    
    delayed_results = []

    for start_index in range(10,12, 1):
        prediction_timesteps = 10

        # precipitation[0:5] -> Used to find motion (past data). Let's call it training precip.
        train_precip = sorted_precipitation[0:start_index]

        # precipitation[5:] -> Used to evaluate forecasts (future data, not available in "real" forecast situation)
        # Let's call it observed precipitation because we will use it to compare our forecast with the actual observations.
        observed_precip = sorted_precipitation[start_index: start_index + prediction_timesteps]

        # train_precip_dbr = dask.delayed(transform_data_dBR)(dask.delayed(train_precip))
        train_precip_dbr = transform_data_dBR(train_precip)


        # motion_field = dask.delayed(getMotionField)(train_precip_dbr)
        motion_field = getMotionField(train_precip_dbr)
        

        # precip_forecast = dask.delayed(LagrangianPersistenceForecast)(train_precip, observed_precip,motion_field)
        # persistence_forecast = dask.delayed(NaivePersistence)(train_precip, observed_precip)
        # steps_forecast = dask.delayed(StepsForecast)(train_precip, observed_precip,motion_field)
        # linda_forecast = dask.delayed(LINDAForecast)(train_precip, observed_precip,motion_field)

        linda_forecast = LINDAForecast(train_precip, observed_precip,motion_field)
        
        precip_forecast = LagrangianPersistenceForecast(train_precip, observed_precip,motion_field)
        persistence_forecast = NaivePersistence(train_precip, observed_precip)
        steps_forecast = StepsForecast(train_precip, observed_precip,motion_field)
        
        forecasts = precip_forecast, persistence_forecast, linda_forecast, steps_forecast

        # delayed_results.append(forecasts)
        methods = ['Naive Persistence', 'Lagrangian Persistence','LINDA' , 'STEPS']

        scores = FSS_plot(forecasts, observed_precip, methods, start_index)
        create_precipitation_prediction_plots(prediction_timesteps,observed_precip,methods, metadata, forecasts, sorted_timestamps[start_index])
        create_animation(prediction_timesteps,observed_precip, metadata, forecasts, sorted_timestamps[start_index])
        save_tiff_files(prediction_timesteps,observed_precip, forecasts, sorted_timestamps[start_index],event_name=data_folder_name.split('/')[-1])
        print("Results calculated for one training setup instance")
        input()


if __name__ == "__main__":

    event_data_df = pd.read_csv('data/EF5events.csv')

    for index, row in event_data_df.iterrows():
        data_folder_name = 'data/' +row['Country'] + '_' + row['Date'].replace('/', '_')
        main(data_folder_name)


