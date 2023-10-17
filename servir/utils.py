# to download IMERG early run data from online
# Import all libraries
import sys
import subprocess
import os
import datetime as DT
import osgeo.gdal as gdal
from osgeo.gdalconst import GA_ReadOnly
#from gdalconst import GA_ReadOnly
#from gdalconst import *
import numpy as np


from pysteps.io.nowcast_importers import import_netcdf_pysteps
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

def ReadandWarp(gridFile,xmin,ymin,xmax,ymax):
    #Read grid and warp to domain grid
    #Assumes no reprojection is necessary, and EPSG:4326
    rawGridIn = gdal.Open(gridFile, GA_ReadOnly)
    # Adjust grid
    pre_ds = gdal.Translate('OutTemp.tif', rawGridIn, options="-co COMPRESS=Deflate -a_nodata 29999 -a_ullr -180.0 90.0 180.0 -90.0")

    gt = pre_ds.GetGeoTransform()
    proj = pre_ds.GetProjection()
    nx = pre_ds.GetRasterBand(1).XSize
    ny = pre_ds.GetRasterBand(1).YSize
    NoData = 29999
    pixel_size = gt[1]

    #Warp to model resolution and domain extents
    ds = gdal.Warp('', pre_ds, srcNodata=NoData, srcSRS='EPSG:4326', dstSRS='EPSG:4326', dstNodata='-9999', format='VRT', xRes=pixel_size, yRes=-pixel_size, outputBounds=(xmin,ymin,xmax,ymax))

    WarpedGrid = ds.ReadAsArray()
    new_gt = ds.GetGeoTransform()
    new_proj = ds.GetProjection()
    new_nx = ds.GetRasterBand(1).XSize
    new_ny = ds.GetRasterBand(1).YSize

    return WarpedGrid, new_nx, new_ny, new_gt, new_proj

def WriteGrid(gridOutName, dataOut, nx, ny, gt, proj):
    #Writes out a GeoTIFF based on georeference information in RefInfo
    driver = gdal.GetDriverByName('GTiff')
    dst_ds = driver.Create(gridOutName, nx, ny, 1, gdal.GDT_Float32, ['COMPRESS=DEFLATE'])
    dst_ds.SetGeoTransform(gt)
    dst_ds.SetProjection(proj)
    dataOut.shape = (-1, nx)
    dst_ds.GetRasterBand(1).WriteArray(dataOut, 0, 0)
    dst_ds.GetRasterBand(1).SetNoDataValue(-9999.0)
    dst_ds = None

def processIMERG(local_filename,llx,lly,urx,ury):
  # Process grid
  # Read and subset grid
  NewGrid, nx, ny, gt, proj = ReadandWarp(local_filename,llx,lly,urx,ury)

  # Scale value
  NewGrid = NewGrid*0.1

  return NewGrid, nx, ny, gt, proj


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