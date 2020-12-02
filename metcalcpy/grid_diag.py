"""
Program Name: grid_diag.py

A python script to create histograms from gridded datasets.

Version  Date
0.1.0    2020/12/01  David Fillmore  Initial version
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
from glob import glob
from datetime import datetime
import numpy as np
import xarray as xr  # http://xarray.pydata.org/
import netCDF4 as nc


def process_data(args, config):
    """
    Arguments:
        args (argparse.Namespace): command line arguments
        config (dictionary): configuration parameters
    """
    regex = os.path.join(args.datadir, config['regex'])
    logging.debug(regex)
    datafiles = glob(regex)

    for datafile in sorted(datafiles):
        logging.info(datafile)


if __name__ == '__main__':
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str,
        required=True,
        help='input data directory')
    parser.add_argument('--outfile', type=str,
        required=True,
        help='output file')
    parser.add_argument('--config', type=str,
        required=True,
        help='configuration file')
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

    """
    Read YAML configuration file
    """
    logging.info(args.config)
    config = yaml.load(
        open(args.config), Loader=yaml.FullLoader)
    logging.info(config)

    process_data(args, config)
