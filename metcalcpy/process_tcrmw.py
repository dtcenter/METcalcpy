import os
import sys
import argparse
import logging


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
    args = parser.parse_args()

    """
    Setup logging
    """
    logging_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(stream=args.logfile, level=logging_level)

    """
    Construct input and output filenames
    """
    filename_in = os.path.join(args.datadir, args.input)
    filename_out = os.path.join(args.datadir, args.output)