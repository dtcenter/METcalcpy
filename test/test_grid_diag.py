import argparse
import numpy as np

if __name__ == '__main__':

    nx_default = 100
    ny_default = 100

    parser = argparse.ArgumentParser()
    parser.add_argument('--nx', type=int,
        default=nx_default,
        help='Number of x grid points')
    parser.add_argument('--ny', type=int,
        default=ny_default,
        help='Number of y grid points')
    parser.add_argument('--i_min', type=int,
                        default=0,
                        help='Domain left grid point')
    parser.add_argument('--i_max', type=int,
                        default=nx_default - 1,
                        help='Domain right grid point')
    parser.add_argument('--j_min', type=int,
                        default=0,
                        help='Domain bottom grid point')
    parser.add_argument('--j_max', type=int,
                        default=ny_default - 1,
                        help='Domain top grid point')
    args = parser.parse_args()
