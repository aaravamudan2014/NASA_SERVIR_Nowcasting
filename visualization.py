
from pysteps import verification
import matplotlib.pyplot as plt
import numpy as np
from pysteps.visualization import plot_precip_field
import matplotlib.animation as animation
from matplotlib import rc
import datetime
import os
from load_data import metadata


def create_precipitation_prediction_plots(observed_precip,methods, forecasts, start_time, event_identifier):
    """Function for creating a png of predictions for the given start time

    Args:
        observed_precip (np.array): array of ground truth precipitation
        methods (list of str): list of nowcasting method names
        forecasts (tuple): tuple of forecasts, this should be of the same order as methods
        start_time (DateTime): Start time of the prediction sequence
        event_identifier (str): Name of the event
    """
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
    
    for index, method in enumerate(methods):
        if method == 'Naive Persistence':
            persistence_forecast = forecasts[index]
        if method == 'Lagrangian Persistence':
            precip_forecast = forecasts[index]
        if method == 'LINDA':
            linda_forecast = forecasts[index]
        if method == 'STEPS':
            steps_forecast = forecasts[index]
        

    os.makedirs('figures/'+event_identifier, exist_ok=True)
    fig, axs = plt.subplots(nrows=len(observed_precip), ncols=len(methods) + 1, figsize=(50,50))

    for j in range(len(observed_precip)):
        for i, ax in enumerate(axs[j].flatten()):
            plt.sca(ax)
            if i == 0  :
                plot_precip_field(np.flip(observed_precip[j].T,0), axis="off",colorbar=False, ax=ax,geodata=metadata)
                if j == 0:
                    plt.title('Observed precipitation')
            elif i == 1:
                plot_precip_field(np.flip(precip_forecast[j].T,0), axis="off",colorbar=False, ax=ax,geodata=metadata)
                if j == 0:
                    plt.title('LP forecast')
            elif i == 2:
                plot_precip_field(np.flip(persistence_forecast[j].T,0), axis="off",colorbar=False, ax=ax,geodata=metadata)
                if j == 0:
                    plt.title('NP forecast')
            elif i == 3:
                plot_precip_field(np.flip(linda_forecast[j].T,0), axis="off",colorbar=False, ax=ax,geodata=metadata)
                if j == 0:
                    plt.title('LINDA forecast')
            elif i == 4:
                plot_precip_field(np.flip(steps_forecast[j].T,0), axis="off",colorbar=False, ax=ax,geodata=metadata)
                if j == 0:
                    plt.title('STEPS forecast')
            
    plt.suptitle('Initialization time: ' + str(start_time))
    plt.tight_layout()
    plt.savefig('figures/' +event_identifier+ '/predicted_precipitation_start_index' + str(start_time) + '.png', dpi=100)



def create_animation(observed_precip, forecasts, start_time, event_identifier, methods):
    """Function to create an animated gif of the predictions

    Args:
        observed_precip (np.array): array of ground truth precipitation
        forecasts (tuple): tuple of forecasts from each method. This should be of the same order as methods
        start_time (DateTime): Start time of the predictions sequence
        event_identifier (str): name of the event
        methods (list of str): list of nowcasting method names
    """
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
    
    for index, method in enumerate(methods):
        if method == 'Naive Persistence':
            persistence_forecast = forecasts[index]
        if method == 'Lagrangian Persistence':
            precip_forecast = forecasts[index]
        if method == 'LINDA':
            linda_forecast = forecasts[index]
        if method == 'STEPS':
            steps_forecast = forecasts[index]
        

    os.makedirs('results/'+event_identifier, exist_ok=True)
    rc('animation', html='jshtml')
    ims = []
    fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(12,4))

    for i in range(1,len(observed_precip)):

        for j, ax in enumerate(axs.flatten()):
            plt.sca(ax)
            plt.xticks([])
            plt.yticks([])
            if j == 0:
                plt.title('Observed precipitation')
            else:
                plt.title(methods[j-1])
            plt.suptitle('Initialization time: ' + str(start_time))
        plt.tight_layout()

        im_1 = axs.flatten()[3].imshow(np.flip(linda_forecast[i].T,0), animated=True)
        if i == 0:
            axs.flatten()[1].imshow(np.zeros_like(np.flip(persistence_forecast[0].T,0)))  # show an initial one first
        im_2 = axs.flatten()[2].imshow(np.flip(persistence_forecast[i].T,0), animated=True)
        # if i == 0:
        #     axs.flatten()[1].imshow(np.zeros_like(np.flip(observed_precip[0].T,0)))  # show an initial one first
        im_3 = axs.flatten()[1].imshow(np.flip(precip_forecast[i].T,0), animated=True)
        # if i == 0:
        #     axs.flatten()[1].imshow(np.zeros_like(np.flip(observed_precip[0].T,0)))  # show an initial one first

        im_4 = axs.flatten()[4].imshow(np.flip(steps_forecast[i].T,0), animated=True)
        # if i == 0:
        #     axs.flatten()[1].imshow(np.zeros_like(np.flip(observed_precip[0].T,0)))  # show an initial one first

        im_5 = axs.flatten()[0].imshow(np.flip(observed_precip[i].T,0), animated=True)
        # if i == 0:
        #     axs.flatten()[1].imshow(np.zeros_like(np.flip(observed_precip[0].T,0)))  # show an initial one first
        
        ims.append([im_1, im_2, im_3, im_4, im_5])

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=False,
                                    repeat_delay=1000)
    ani.save('figures/'+event_identifier+'/precipitation_forecasts_'+str(start_time)+'.gif', writer='imagemagick', fps=30)

