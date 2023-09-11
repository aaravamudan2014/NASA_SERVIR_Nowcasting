from pysteps.utils import transformation
from pysteps import motion
from pysteps.visualization import plot_precip_field, quiver
import matplotlib.pyplot as plt
from pysteps import nowcasts
import time
import numpy as np

from pysteps.motion.lucaskanade import dense_lucaskanade


def transform_data_dBR(train_precip):
    # Log-transform the data to dBR.
    # The threshold of 0.1 mm/h sets the fill value to -15 dBR.
    train_precip_dbr, metadata_dbr = transformation.dB_transform(
        train_precip, threshold=0.1, zerovalue=-15.0
    )
    return train_precip_dbr


# Estimate the motion field with chosen method
def getMotionField(train_precip_dbr, metadata=None, method = "LK", plot_motion_field=False):
    # Import the Lucas-Kanade optical flow algorithm
    oflow_method = motion.get_method(method)

    # Estimate the motion field from the training data (in dBR)
    motion_field = oflow_method(train_precip_dbr)

    if plot_motion_field:
        # Plot the motion field.
        plt.figure(figsize=(9, 5), dpi=100)
        plt.title("Estimated motion field with the Lukas-Kanade algorithm")
        plt.show()

        # Plot the motion field vectors
        quiver(motion_field, geodata=metadata, step=15)
        plt.savefig('motion_field.png')
        plt.tight_layout()
        plt.show()

    return motion_field

def LagrangianPersistenceForecast(train_precip, observed_precip,motion_field):
  start = time.time()

  # Extrapolate the last radar observation
  extrapolate = nowcasts.get_method("extrapolation")

  # You can use the precipitation observations directly in mm/h for this step.
  last_observation = train_precip[-1]

  # last_observation[~np.isfinite(last_observation)] = metadata["zerovalue"]

  # We set the number of leadtimes (the length of the forecast horizon) to the
  # length of the observed/verification preipitation data. In this way, we'll get
  # a forecast that covers these time intervals.
  n_leadtimes = observed_precip.shape[0]

  # Advect the most recent radar rainfall field and make the nowcast.
  precip_forecast = extrapolate(last_observation, motion_field, n_leadtimes)

  # This shows the shape of the resulting array with [time intervals, rows, cols]
  print("The shape of the resulting array is: ", precip_forecast.shape)

  end = time.time()
  print("Advecting the radar rainfall fields took ", (end - start), " seconds")
  return precip_forecast


def StepsForecast(train_precip, observed_precip,motion_field):
  start = time.time()

  # Extrapolate the last radar observation
  extrapolate = nowcasts.get_method("steps")

  # last_observation[~np.isfinite(last_observation)] = metadata["zerovalue"]

  # We set the number of leadtimes (the length of the forecast horizon) to the
  # length of the observed/verification preipitation data. In this way, we'll get
  # a forecast that covers these time intervals.
  n_leadtimes = observed_precip.shape[0]
  n_ens_members = 20
  # Advect the most recent radar rainfall field and make the nowcast.
  precip_forecast_ensemble = extrapolate(train_precip[-3:, :, :], 
                                motion_field, 
                                n_leadtimes,
                                n_ens_members,
                                n_cascade_levels=3,
                                R_thr=-10.0,
                                kmperpixel=2.0,
                                timestep=30,
                                seed=20)
  
  # take the ensemble mean to prodice a deterministic forecast
  precip_forecast = np.mean(precip_forecast_ensemble, axis=0)
  
  # This shows the shape of the resulting array with [time intervals, rows, cols]
  print("The shape of the resulting array is: ", precip_forecast.shape)

  end = time.time()
  print("Advecting the radar rainfall fields took ", (end - start), " seconds")
  return precip_forecast



def LINDAForecast(train_precip, observed_precip,motion_field):
  start = time.time()

  # Extrapolate the last radar observation
  # extrapolate = nowcasts.get_method("linda")
  from pysteps.nowcasts import linda
  # You can use the precipitation observations directly in mm/h for this step.
  last_observation = train_precip[-1]

  # last_observation[~np.isfinite(last_observation)] = metadata["zerovalue"]

  # We set the number of leadtimes (the length of the forecast horizon) to the
  # length of the observed/verification preipitation data. In this way, we'll get
  # a forecast that covers these time intervals.
  n_leadtimes = observed_precip.shape[0]
  # Advect the most recent radar rainfall field and make the nowcast.
  linda_forecast = linda.forecast(train_precip, motion_field, n_leadtimes,max_num_features=15, kmperpixel = 10, timestep=30
                                ,add_perturbations =False)

  # This shows the shape of the resulting array with [time intervals, rows, cols]
  print("The shape of the resulting array is: ", linda_forecast.shape)

  end = time.time()
  print("Advecting the radar rainfall fields took ", (end - start), " seconds")
  return linda_forecast

def NaivePersistence(train_precip, observed_precip):
  start = time.time()
  persistence_forecast = np.empty_like(observed_precip)
  for precipitation_index in range(len(observed_precip)):


    # You can use the precipitation observations directly in mm/h for this step.
    if precipitation_index < 1:
      last_observation = train_precip[-1]
    else:
      last_observation = train_precip[-1]#observed_precip[precipitation_index-1]

    # last_observation[~np.isfinite(last_observation)] = metadata["zerovalue"]

    # We set the number of leadtimes (the length of the forecast horizon) to the
    # length of the observed/verification preipitation data. In this way, we'll get
    # a forecast that covers these time intervals.
    n_leadtimes = observed_precip.shape[0]

    # Advect the most recent radar rainfall field and make the nowcast.
    persistence_forecast[precipitation_index] = last_observation

  # This shows the shape of the resulting array with [time intervals, rows, cols]
  print("The shape of the resulting array is: ", persistence_forecast.shape)

  end = time.time()
  print("Advecting the radar rainfall fields took ", (end - start), " seconds")
  return persistence_forecast