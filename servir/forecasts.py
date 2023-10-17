import time

import numpy as np
from pysteps.utils import transformation
from pysteps.nowcasts import linda
from pysteps import nowcasts
from pysteps import motion
from pysteps.motion.lucaskanade import dense_lucaskanade

def linda_nowcast(train_precip,timesteps, max_num_features = 15, add_perturbations=False):

    # Estimate the motion field
    V = dense_lucaskanade(train_precip)

    # The linda nowcast
    forcast = linda.forecast(train_precip, V, timesteps, max_num_features=max_num_features, add_perturbations=add_perturbations)

    return forcast


def steps_nowcast(train_precip, timesteps, n_ens_members = 20, n_cascade_levels=6):

    R_train, _ = transformation.dB_transform(train_precip, threshold=0.1, zerovalue=-15.0)

    # Set missing values with the fill value
    R_train[~np.isfinite(R_train)] = -15.0

    # Estimate the motion field
    V = dense_lucaskanade(R_train)

    # The STEPS nowcast
    nowcast_method = nowcasts.get_method("steps")
    R_forcast = nowcast_method(R_train, V, timesteps, n_ens_members=n_ens_members, n_cascade_levels=n_cascade_levels,\
                               precip_thr = -10.0, kmperpixel=10, timestep=30)

    # Back-transform to rain rates
    R_forcast = transformation.dB_transform(R_forcast, threshold=-10.0, inverse=True)[0]

    # the ensemble mean
    R_f_mean = np.nanmean(R_forcast, axis=0)

    return R_f_mean


def lp_nowcast(train_precip, timesteps):
    
    R_train, _ = transformation.dB_transform(train_precip, threshold=0.1, zerovalue=-15.0)

    # Estimate the motion field with Lucas-Kanade
    oflow_method = motion.get_method("LK")
    V = oflow_method(R_train)

    # Extrapolate the last radar observation
    extrapolate = nowcasts.get_method("extrapolation")
    R_train[~np.isfinite(R_train)] =-15.0
    R_f = extrapolate(R_train[-1, :, :], V, timesteps)

    # Back-transform to rain rate
    R_f = transformation.dB_transform(R_f, threshold=-10.0, inverse=True)[0]

    return R_f





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