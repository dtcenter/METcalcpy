import argparse
import math
import numpy as np

if __name__ == '__main__':

    nx_default = 10
    ny_default = 10

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
    parser.add_argument('--sigma', type=float,
                        default=1,
                        help='Normal distribution width')
    parser.add_argument('--mu_x', type=float,
                        default=nx_default/2,
                        help='Normal distribution x-mean')
    parser.add_argument('--mu_y', type=float,
                        default=ny_default/2,
                        help='Normal distribution y-mean')
    parser.add_argument('--n_bin', type=int,
                        default=10,
                        help='Number of bins')
    parser.add_argument('--min', type=int,
                        default=0,
                        help='Bin minimum')
    parser.add_argument('--max', type=int,
                        default=0.001,
                        help='Bin maximum')
    args = parser.parse_args()

    x_coords = np.linspace(args.x_min, args.x_max, args.nx + 1)
    y_coords = np.linspace(args.y_min, args.y_max, args.ny + 1)
    x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)

    r2 = (x_mesh - args.mu_x) * (x_mesh - args.mu_x) \
        + (y_mesh - args.mu_y) * (y_mesh - args.mu_y)
    sigma2 = args.sigma * args.sigma
    values = np.exp(- r2 / (2 * sigma2)) / math.sqrt(2 * math.pi * sigma2)

    print(values)

    pdf = np.histogram(values, bins=args.n_bin, range=(args.min, args.max))
    print(pdf)

