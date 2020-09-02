"""
Program Name: vertical_interp.py

A python script to vertically interpolate fields
between grids with pressure or height vertical coordinates.

Version  Date
0.1.0    2020/09/01  David Fillmore  Initial version
"""

__author__ = 'David Fillmore'
__version__ = '0.1.0'
__email__ = 'met_help@ucar.edu'

"""
Import standard modules
"""
import os
import sys
import argparse
import logging
import yaml
import xarray as xr # http://xarray.pydata.org/

"""
Import MetPy
    https://unidata.github.io/MetPy/
"""
from metpy import calc

def vertical_interp(
    vertical_coord, coordinate_surfaces, field):
    """
    Interpolate field onto coordinate surfaces.

    Arguments:
        vertical_coord (str): vertical coordinate in field
        coordinate_surfaces (DataArray): coordinate surfaces
        field (DataArray): field

    Returns:
        field_interp (DataArray): Interpolated field
    """
    pass

def height_from_pressure(
    surface_pressure, temperature, relative_humidity):
    """
    Compute height coordinate surfaces as a function of pressure.

    Arguments:
        surface_pressure (DataArray) : surface pressure
        temperature (DataArray) : temperature
        relative_humidity (DataArray) : relative humidity

    Returns:
        height (DataArray) : height
    """
    logging.info('pressure to height conversion')
    # use thickness_hydrostatic_from_relative_humidity

def read_required_fields(config, ds):
    """
    Read required fields.

    Arguments:
        config (dictionary) : configuration parameters
        ds (DataSet) : xarray dataset

    Returns:
        surface_pressure (DataArray) : surface pressure
        temperature (DataArray) : temperature
        relative_humidity (DataArray) : relative humidity
    """
    surface_pressure \
        = ds[config['surface_pressure_name']]
    temperature \
        = ds[config['temperature_name']]
    relative_humidity \
        = ds[config['relative_humidity_name']]
    return surface_pressure, temperature, relative_humidity

if __name__ == '__main__':
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str,
        default=os.getenv('DATA_DIR'),
        help='top-level data directory (default $DATA_DIR)')
    parser.add_argument('--input', type=str,
        required=True,
        help='input file name')
    parser.add_argument('--config', type=str,
        required=True,
        help='configuration file')
    parser.add_argument('--output', type=str,
        required=True,
        help='output file name')
    parser.add_argument('--logfile', type=str, 
        default=sys.stdout,
        help='log file (default stdout)')
    parser.add_argument('--debug', action='store_true',
        help='set logging level to debug')
    args = parser.parse_args()

    """
    Setup logging
    """
    logging_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(stream=args.logfile, level=logging_level)

    if os.path.isdir(args.datadir):
        logging.info(args.datadir)
    else:
        logging.error(args.datadir + ' not found')
        sys.exit(1)
    logging.info(args.input)
    logging.info(args.config)
    logging.info(args.output)

    filename_in = os.path.join(args.datadir, args.input)

    """
    Read YAML configuration file
    """
    config = yaml.load(
        open(args.config), Loader=yaml.FullLoader)
    logging.info(config)

    """
    Convert pressure levels to height levels
    """
    if (config['vertical_coord_type_in'] == 'pressure'
        and config['vertical_coord_type_out'] == 'height'):

        try:
            logging.info('Opening ' + filename_in)
            ds = xr.open_dataset(filename_in)
        except:
            logging.error('Unable to open ' + filename_in)

        surface_pressure, temperature, relative_humidity \
            = read_required_fields(config, ds)

        height_from_pressure(
            surface_pressure, temperature, relative_humidity)
