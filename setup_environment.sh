sudo apt-get install libeccodes-dev libproj-dev

pip install pyproj pygrib netCDF4
pip uninstall --yes shapely

sudo apt-get install -qq libgdal-dev libgeos-dev
pip install shapely --no-binary shapely

pip install cartopy pysteps imageio utm netcdf4 opencv-python dask distributed

sudo apt-get install build-essential python-all-dev

conda install -c conda-forge gdal