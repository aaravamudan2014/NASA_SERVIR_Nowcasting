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


def loadTiff(in_image, init=None, size_img=None):
    src = gdal.Open(in_image)

    print(src)

    nbands = src.RasterCount
    in_band = src.GetRasterBand(1)  # load one band for size reference
    if init is None:
        xinit,yinit = (0, 0)
    else:
        xinit,yinit = init
    if size_img is None:
        block_xsize, block_ysize = (in_band.XSize, in_band.YSize)
    else:
        block_xsize, block_ysize = size_img

    # read the (multiband) file into an array
    image = src.ReadAsArray(xinit, yinit, block_xsize, block_ysize)
    # reshape from bandsxheightxwidth to wxhxb
    image = np.moveaxis(image, 0, -1)
    return image, block_ysize, block_xsize, nbands



def load_IMERG_data(data_location):
    #i We'll import the time module to measure the time the importer needed
    import time
    import glob
    import numpy as np
    import netCDF4 as nc


    start_time = time.time()
    precipitation = []
    times = []
    locations = []

    files = glob.glob(data_location+'*.nc4')

    metadata = {'accutime': 2.0,
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

    for index, file in enumerate(files):
        if index % 5 == 0:
            print('\r', index , " of ", len(files), " have been processed", end='')
        ds = nc.Dataset(file)
        precipitationCal = np.array(ds.variables['precipitationCal'])
        timestamp = ds.variables['time']
        longitude = ds.variables['lon']
        latitude = ds.variables['lat']

        times.append(timestamp)
        locations.append((latitude[0], longitude[0]))
        precipitation.append(precipitationCal[0])

    precipitation = np.array(precipitation)
    end_time = time.time()

    print("Precipitation data imported")
    print("Importing the data took ", (end_time - start_time), " seconds")

    return precipitation, locations, times, metadata



def sort_IMERG_data(precipitation, times, locations, metadata,generate_animated_gif=False):
    timestamp_list = []


    for timestamp in times:
        timestamp = nc.num2date(timestamp[:],timestamp.units, only_use_cftime_datetimes=False)
        timestamp_list.extend(timestamp)

    metadata['timestamps'] = np.array(timestamp_list, dtype=object)

    sorted_index_array = np.argsort(timestamp_list)
    sorted_timestamps = np.array(timestamp_list)[sorted_index_array]
    sorted_precipitation = np.array(precipitation)[sorted_index_array]
    sorted_location = np.array(locations)[sorted_index_array]
    precipitation = sorted_precipitation

    if generate_animated_gif:
        images = []
        for i in range(len(sorted_precipitation)):
            images.append(imageio.core.util.Array(sorted_precipitation[i]))
        imageio.mimsave('movie.gif', images)




    timestep = np.diff(sorted_timestamps)
    # Let's inspect the shape of the imported data array
    print(sorted_precipitation.shape)

    return sorted_precipitation, sorted_timestamps, sorted_location

# image, block_ysize, block_xsize, nbands = loadTiff('/content/imerg.202201010100.30minAccum.tif')


def init_IMERG_config_pysteps():
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
    
def main():
    init_IMERG_config_pysteps()
    precipitation, locations, times, metadata = load_IMERG_data(data_location='data/IMERG/Flood_Ghana_032023/')
    sorted_precipitation, sorted_timestamps, sorted_location = sort_IMERG_data(precipitation, times, locations, metadata,generate_animated_gif=False)
    
if __name__ == '__main__':
    main()