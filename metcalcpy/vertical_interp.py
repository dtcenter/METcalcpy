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
import xarray as xr

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
    surface_temperature, virtual_temperature, pressure):
    """
    Compute height coordinate surfaces as a function of pressure.

        Z_2 - Z_1 = (R_d / g) <T_v> log(p_1 / p_2)
        R_d / g = 29.3
        <T_v> = integral_p_2^p_1 T_v(p) (dp / p) / log(p_1 / p_2)

    Arguments:
        surface_temperature (DataArray) : surface temperature
        virtual_temperature (DataArray) : virtual temperature
        pressure (DataArray) : pressure

    Returns:
        height (DataArray) : height
    """
    logging.info('pressure to height conversion')

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

        height_from_pressure()
