import os
import sys
import argparse
import logging
import numpy as np
import xarray as xr


def read_tcrmw(filename):
    ds = xr.open_dataset(filename)
    # range, azimuth, pressure, track_point
    for var in ds.keys():
        logging.info((var, ds[var].dims))
    for coord in ds.coords:
        logging.debug((coord, ds[coord].values))
    return ds


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
    parser.add_argument('--output', type=str,
                        required=True,
                        help='output file name')
    parser.add_argument('--logfile', type=str,
                        default=sys.stdout,
                        help='log file (default stdout)')
    parser.add_argument('--debug', action='store_true',
                        help='set logging level to debug')
    parser.add_argument('--vars', type=str,
                        help='variables',
                        default='UGRD,VGRD,TMP')
    parser.add_argument('--levels', type=str,
                        help='vertical height levels',
                        default='100,200,500,1000,1500,2000,3000,4000,5000')
    args = parser.parse_args()

    """
    Setup logging
    """
    logging_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(stream=args.logfile, level=logging_level)

    """
    Construct input and output filenames
    """
    if os.path.isdir(args.datadir):
        logging.info(args.datadir)
    else:
        logging.error(args.datadir + ' datadir not found')
        sys.exit(1)
    filename_in = os.path.join(args.datadir, args.input)
    filename_out = os.path.join(args.datadir, args.output)

    """
    Height levels
    """
    levels = np.array([float(lev) for lev in args.levels.split(',')])
    logging.info(('levels', levels))

    """
    Height levels
    """
    var_list = args.vars.split(',')
    logging.info(('vars', var_list))

    """
    Open dataset
    """
    ds = read_tcrmw(filename_in)
