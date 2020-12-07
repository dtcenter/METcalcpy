import argparse
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--nx', type=int,
        default=100,
        help='Number of x grid points')
    parser.add_argument('--ny', type=int,
        default=100,
        help='Number of y grid points')
    args = parser.parse_args()
