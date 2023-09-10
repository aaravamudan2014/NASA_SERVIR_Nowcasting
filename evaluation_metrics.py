from pysteps import verification
import matplotlib.pyplot as plt
import numpy as np
from pysteps.visualization import plot_precip_field
import matplotlib.animation as animation
from matplotlib import rc
import datetime
import os


def FSS_plot(forecasts, observed_precip, methods, start_index):

    precip_forecast, persistence_forecast, linda_forecast, steps_forecast = forecasts

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
        elif method == 'steps':
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
    plt.savefig('figures/FSS_comparison_'+str(start_index)+'.png')
    return score


def create_precipitation_prediction_plots(prediction_timesteps,observed_precip,methods, metadata, forecasts, start_time):
    min = -21.4
    xmax = 30.4
    ymin = -2.9
    ymax = 33.1
    geodata = {'x1':-21.4,
            'x2':30.4,
            'y1':-2.9,
            'y2': 33.1,
            'yorigin':'lower',
            'projection':'EPSG:2269'
            }
    precip_forecast, persistence_forecast, linda_forecast, steps_forecast = forecasts

    fig, axs = plt.subplots(nrows=prediction_timesteps, ncols=len(methods), figsize=(50,50))

    for j in range(prediction_timesteps):
        for i, ax in enumerate(axs[j].flatten()):
            plt.sca(ax)
            if i == 0 and j==0 :
                plot_precip_field(np.flip(observed_precip[j].T,0), axis="off",colorbar=False, ax=ax,geodata=metadata)
                plt.title('Observed precipitation')
            elif i == 1 and j==0:
                plot_precip_field(np.flip(precip_forecast[j].T,0), axis="off",colorbar=False, ax=ax,geodata=metadata)
                plt.title('LP forecast')
            elif i == 2 and j==0:
                plot_precip_field(np.flip(persistence_forecast[j].T,0), axis="off",colorbar=False, ax=ax,geodata=metadata)
                plt.title('NP forecast')
            elif i == 3 and j==0:
                plot_precip_field(np.flip(linda_forecast[j].T,0), axis="off",colorbar=False, ax=ax,geodata=metadata)
                plt.title('LINDA forecast')
            elif i == 4 and j==0:
                plot_precip_field(np.flip(steps_forecast[j].T,0), axis="off",colorbar=False, ax=ax,geodata=metadata)
                plt.title('STEPS forecast')
            
    plt.suptitle('Initialization time: ' + str(start_time))
    plt.tight_layout()
    plt.savefig('figures/predicted_precipitation_start_index' + str(start_time) + '.png', dpi=100)



def create_animation(prediction_timesteps,observed_precip, metadata, forecasts, start_time):
    min = -21.4
    xmax = 30.4
    ymin = -2.9
    ymax = 33.1
    geodata = {'x1':-21.4,
            'x2':30.4,
            'y1':-2.9,
            'y2': 33.1,
            'yorigin':'lower',
            'projection':'EPSG:2269'
            }
    precip_forecast, persistence_forecast, linda_forecast, steps_forecast = forecasts
    rc('animation', html='jshtml')
    ims = []
    fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(12,4))

    for i in range(1,len(observed_precip)):

        for j, ax in enumerate(axs.flatten()):
            plt.sca(ax)
            plt.xticks([])
            plt.yticks([])
            if j == 0:
            # plot_precip_field(observed_precip[j], axis="off",colorbar=False, ax=ax)
                plt.title('Observed precipitation')
            elif j == 1:
            # plot_precip_field(precip_forecast[j], axis="off",colorbar=False, ax=ax)
                plt.title('LP forecast')
            elif j == 2:
            # plot_precip_field(persistence_forecast[j], axis="off",colorbar=False, ax=ax)
                plt.title('NP forecast')
            elif j == 3:
            # plot_precip_field(linda_forecast[j], axis="off",colorbar=False, ax=ax)
                plt.title('LINDA forecast')
            elif j == 4:
            # plot_precip_field(linda_forecast[j], axis="off",colorbar=False, ax=ax)
                plt.title('STEPS forecast')
        plt.suptitle('Initialization time: ' + str(start_time))
        plt.tight_layout()

        im_1 = axs.flatten()[3].imshow(np.flip(linda_forecast[i].T,0), animated=True)
        if i == 0:
            axs.flatten()[1].imshow(np.zeros_like(np.flip(persistence_forecast[0].T,0)))  # show an initial one first
        im_2 = axs.flatten()[2].imshow(np.flip(persistence_forecast[i].T,0), animated=True)
        if i == 0:
            axs.flatten()[1].imshow(np.zeros_like(np.flip(observed_precip[0].T,0)))  # show an initial one first
        im_3 = axs.flatten()[1].imshow(np.flip(precip_forecast[i].T,0), animated=True)
        if i == 0:
            axs.flatten()[1].imshow(np.zeros_like(np.flip(observed_precip[0].T,0)))  # show an initial one first

        im_4 = axs.flatten()[4].imshow(np.flip(steps_forecast[i].T,0), animated=True)
        if i == 0:
            axs.flatten()[1].imshow(np.zeros_like(np.flip(observed_precip[0].T,0)))  # show an initial one first

        im_5 = axs.flatten()[0].imshow(np.flip(observed_precip[i].T,0), animated=True)
        if i == 0:
            axs.flatten()[1].imshow(np.zeros_like(np.flip(observed_precip[0].T,0)))  # show an initial one first
        
        ims.append([im_1, im_2, im_3, im_4, im_5])

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=False,
                                    repeat_delay=1000)
    ani.save('figures/precipitation_forecasts_'+str(start_time)+'.gif', writer='imagemagick', fps=30)


def save_tiff_files(prediction_timesteps,observed_precip, forecasts, start_time, event_name):

    precip_forecast, persistence_forecast, linda_forecast, steps_forecast = forecasts
    for model_index in range(4):
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

        os.makedirs('results/'+event_name, exist_ok=True)
        for index, predicted_observation in enumerate(steps_forecast):
            
            filename = 'results/' +event_name+'/'+ str(start_time + datetime.timedelta(minutes= int(index) *30 )) +'.tif'
            plt.imshow(predicted_observation)
            plt.savefig(filename)
