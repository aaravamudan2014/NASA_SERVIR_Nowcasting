# code obtained from https://gis.stackexchange.com/questions/420183/how-to-read-and-plot-multiple-tiff-files-in-colaboratory

import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
from pysteps.io.nowcast_importers import import_netcdf_pysteps
from cftime import num2pydate
from datetime import datetime
import imageio
import numpy
from pysteps.datasets import  create_default_pystepsrc
import netCDF4 as nc
import time
import glob
import numpy as np
import netCDF4 as nc

import datetime as DT
import osgeo.gdal as gdal
from osgeo.gdal import gdalconst
from osgeo.gdalconst import GA_ReadOnly

# metadata variables, useful for plotting when you want to visualize map in background
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


def load_IMERG_data_tiff(data_location):
    """Function to load IMERG tiff data from the associate event folder

    Args:
        data_location (str): string path to the location of the event data

    Returns:
        precipitation (np.array): np.array of precipitations (not sorted by time)
        times (np.array): np.array of date times that match 1:q with precipitation
    """
    precipitation = []
    times = []
    files = glob.glob(data_location+'/processed_imerg/*.tif')
    for file in files:
        tiff_data = gdal.Open(file, GA_ReadOnly)
        imageArray = np.array(tiff_data.GetRasterBand(1).ReadAsArray())
        date_str = file.split('/')[-1].split('.')[1]
        year = date_str[0:4]
        month = date_str[4:6]
        day = date_str[6:8]
        hour = date_str[8:10]
        minute = date_str[10:12]
        dt = DT.datetime.strptime(year + '-'+ month + '-' + day + ' '+ hour + ':' + minute, '%Y-%m-%d %H:%M')
        times.append(dt)
        precipitation.append(imageArray)
    precipitation = np.array(precipitation)

    return precipitation, times, 
    
def load_IMERG_data_nc4(data_location):
    """ [DEPRECATED] Function to load IMERG nc4 data from the associate  folder

    Args:
        data_location (str): string path to the location of the event data

    Returns:
        precipitation (np.array): np.array of precipitations (not sorted by time)
        times (np.array): np.array of date times that match 1:q with precipitation
    """

    start_time = time.time()
    precipitation = []
    times = []
    
    files = glob.glob(data_location+'*.nc4')

    
    for index, file in enumerate(files):
        if index % 5 == 0:
            print('\r', index , " of ", len(files), " have been processed", end='')
        ds = nc.Dataset(file)
        precipitationCal = np.array(ds.variables['precipitationCal'])
        timestamp = ds.variables['time']
        longitude = ds.variables['lon']
        latitude = ds.variables['lat']

        times.append(timestamp)
        precipitation.append(precipitationCal[0])

    precipitation = np.array(precipitation)
    end_time = time.time()

    print("Precipitation data imported")
    print("Importing the data took ", (end_time - start_time), " seconds")

    return precipitation, times


def sort_IMERG_data(precipitation, times):
    """Function to sort the imerg data based on precitpitation and times array

    Args:
        precipitation (np.array): numpy array of precipitation images
        times (np.array): numpy array of datetime objects that match 1:1 with precipitation array

    Returns:
        sorted_precipitation (np.array): sorted numpy array of precipitation images
        sorted_timestamps (np.array): sorted numpy array of datetime objects that match 1:1 with sorted precipitation array

    """
    sorted_index_array = np.argsort(times)
    # print(sorted_index_array)
    sorted_timestamps = np.array(times)[sorted_index_array]
    sorted_precipitation = np.array(precipitation)[sorted_index_array]

    timestep = np.diff(sorted_timestamps)

    # Let's inspect the shape of the imported data array
    print("Shape of the sorted precipitation array", sorted_precipitation.shape)

    return sorted_precipitation, sorted_timestamps

def init_IMERG_config_pysteps():
    """Function to initialize Pysteps for IMERG data
        This has been adapted from Pysteps' tutorial colab notebook
    
    """
    # If the configuration file is placed in one of the default locations
    # (https://pysteps.readthedocs.io/en/latest/user_guide/set_pystepsrc.html#configuration-file-lookup)
    # it will be loaded automatically when pysteps is imported.
    config_file_path = create_default_pystepsrc("pysteps_data")



    # Import pysteps and load the new configuration file
    import pysteps
    from pprint import pprint

    _ = pysteps.load_config_file(config_file_path, verbose=True)
    # The default parameters are stored in pysteps.rcparams.

    # print(pysteps.rcparams.data_sources.keys())
    pysteps.rcparams.data_sources['imerg'] = {'fn_ext': 'nc4',
                                            'fn_pattern': 'PrecipRate_00.00_%Y%m%d-%H%M%S',
                                            'importer': 'netcdf_pysteps',
                                            'importer_kwargs': {},
                                            'path_fmt': '%Y/%m/%d',
                                            'root_path': '/content/IMERG/Flood_Ghana_032023/',
                                            'timestep': 30}
    print(pysteps.rcparams.data_sources['imerg'])
    

def generate_animation(sorted_precipitation):
    import matplotlib.animation as animation
    from matplotlib import rc

    print('Genearting animation')
    rc('animation', html='jshtml')
    ims = []

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(12,4))

    for i in range(len(sorted_precipitation)):
        ims.append([axs.imshow(sorted_precipitation[i],  animated=True)])

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=False,
                                    repeat_delay=1000)
    ani.save('observed_precipitation.gif', writer='imagemagick', fps=30)


def generate_animation_pysteps(sorted_precipitation):
    from pysteps.visualization.animations import animate
    animate(sorted_precipitation, geodata=metadata, time_wait=0.001)

def main():
    init_IMERG_config_pysteps()
    # precipitation, locations, times, metadata = load_IMERG_data_tiff(data_location='data/IMERG/Flood_Ghana_032023/')
    precipitation, times = load_IMERG_data_tiff(data_location="data/Côte d'Ivoire_18_06_2018/")
    
    sorted_precipitation, sorted_timestamps = sort_IMERG_data(precipitation, times)

    generate_animation(sorted_precipitation)
    generate_animation_pysteps(sorted_precipitation)
    
    
if __name__ == '__main__':
    main()