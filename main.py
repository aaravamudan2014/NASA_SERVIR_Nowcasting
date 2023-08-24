from load_data import *
from forecasts import *
from evaluation_metrics import FSS_plot


def main():
    # load data
    init_IMERG_config_pysteps()
    precipitation, locations, times, metadata = load_IMERG_data(data_location='data/Flood_Ghana_032023/')
    sorted_precipitation, sorted_timestamps, sorted_location = sort_IMERG_data(precipitation, times, locations, metadata,generate_animated_gif=False)
    
    import dask

    from dask.distributed import Client
    client = Client(n_workers=1)
    delayed_results = []

    for start_index in range(5,12, 1):
        prediction_timesteps = 10

        # precipitation[0:5] -> Used to find motion (past data). Let's call it training precip.
        train_precip = sorted_precipitation[0:start_index]

        # precipitation[5:] -> Used to evaluate forecasts (future data, not available in "real" forecast situation)
        # Let's call it observed precipitation because we will use it to compare our forecast with the actual observations.
        observed_precip = sorted_precipitation[start_index: start_index + prediction_timesteps]

        train_precip_dbr = dask.delayed(transform_data_dBR)(dask.delayed(train_precip))


        motion_field = dask.delayed(getMotionField)(train_precip_dbr)

        precip_forecast = dask.delayed(LagrangianPersistenceForecast)(train_precip, observed_precip,motion_field)
        persistence_forecast = dask.delayed(NaivePersistence)(train_precip, observed_precip)
        steps_forecast = dask.delayed(StepsForecast)(train_precip, observed_precip,motion_field)
        linda_forecast = dask.delayed(LINDAForecast)(train_precip, observed_precip,motion_field)
        
        forecasts = precip_forecast, persistence_forecast , linda_forecast, steps_forecast

        delayed_results.append(forecasts)
        methods = ['Naive Persistence', 'Lagrangian Persistence', 'LINDA', 'STEPS']
        
        #delayed_results.append(FSS_plot(forecasts, observed_precip, methods, start_index))

    results = dask.compute(*delayed_results)




if __name__ == "__main__":
    main()