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
    parser.add_argument('--x_min', type=float,
                        default=0,
                        help='Domain left x-coordinate')
    parser.add_argument('--x_max', type=float,
                        default=nx_default,
                        help='Domain right x-coordinate')
    parser.add_argument('--y_min', type=float,
                        default=0,
                        help='Domain bottom y-coordinate')
    parser.add_argument('--y_max', type=float,
                        default=ny_default,
                        help='Domain top y-coordinate')
    parser.add_argument('--n_bin', type=int,
                        default=10,
                        help='Number of bins')
    parser.add_argument('--min', type=int,
                        default=0,
                        help='Bin minimum')
    parser.add_argument('--max', type=int,
                        default=0,
                        help='Bin maximum')
    args = parser.parse_args()

    x_coords = np.linspace(args.x_min, args.x_max, args.nx + 1)
    y_coords = np.linspace(args.y_min, args.y_max, args.ny + 1)

    print(x_coords)
    print(y_coords)

