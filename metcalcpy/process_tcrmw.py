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


def compute_interpolation_weights(args, ds, levels):
    dims = ds[args.T].dims
    logging.info(dims)
    nr, na, nl, nt = ds[args.T].shape
    logging.info((nr, na, nl, nt))


def compute_wind_components(args, ds):
    """
    e_r = cos(theta) e_x + sin(theta) e_y
    e_theta = - sin(theta) e_x + cos(theta) e_y
    """
    nr, na, nl, nt = ds[args.T].shape
    logging.info((nr, na, nl, nt))
    theta = np.empty((nr, na, nl, nt), dtype=np.float32)
    for i in range(nr):
        for k in range(nl):
            for t in range(nt):
                theta[i, :, k, t] \
                    = (np.pi / 180) * ds['azimuth'].values + np.pi / 2
    mask = np.greater(theta, 2 * np.pi)
    theta[mask] = theta[mask] - 2 * np.pi
    u_radial = np.cos(theta) * ds[args.u].values + np.sin(theta) * ds[args.v].values
    u_tangential = - np.sin(theta) * ds[args.u].values + np.cos(theta) * ds[args.v].values
    return u_radial, u_tangential


def test_plot_pressure_lev(args, ds, track_index=0):
    # Plot setup
    import matplotlib.pyplot as plt
    #import seaborn as sns
    # Set for dark PyCharm theme
    #plt.style.use('dark_background')
    textcolor = (175 / 255, 177 / 255, 179 / 255)
#    facecolor = (60 / 255, 63 / 255, 65 / 255)
    # textcolor = (0, 0, 0)
    # facecolor = (1, 1, 1)
    #sns.set_context('notebook')
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['text.color'] = textcolor
    plt.rcParams['axes.edgecolor'] = textcolor
    plt.rcParams['axes.labelcolor'] = textcolor
    plt.rcParams['xtick.color'] = textcolor
    plt.rcParams['ytick.color'] = textcolor
    fig = plt.figure(figsize=(10, 5))
    #fig.patch.set_facecolor(facecolor)
    ax = fig.add_subplot()
    #ax.set_facecolor(facecolor)
    ax.annotate('Tangential Wind (m s-1)', xy=(14, 350), color='darkgreen')
    ax.annotate('Temperature (K)', xy=(14, 370), color='darkblue')
    # nautical miles to kilometers conversion factor
    nm_to_km = 1.852
    ax.set_xlabel(
        'Range (RMW = %4.1f km)' % (nm_to_km * ds['RMW'][track_index]))
    ax.set_xticks(np.arange(1, 20))
    ax.set_ylabel('Pressure (mb)')
    ax.set_yscale('symlog')
    ax.set_ylim(1000, 250)
    ax.set_yticks(np.arange(1000, 250, -100))
    ax.set_yticklabels(np.arange(1000, 250, -100))

    # Azimuthal averages
    u_radial_azi_mean = np.mean(ds['u_radial'].values, axis=1)
    u_tangential_azi_mean = np.mean(ds['u_tangential'].values, axis=1)
    T_azi_mean = np.mean(ds[args.T].values, axis=1)

    # Contour plots
    u_contour = ax.contour(ds['range'].values, ds['pressure'].values,
        u_tangential_azi_mean[:, :, track_index].transpose(),
        levels=np.arange(5, 40, 5), colors='darkgreen', linewidths=1)
    ax.clabel(u_contour, colors='darkgreen', fmt='%1.0f')
    T_contour = ax.contour(ds['range'].values, ds['pressure'].values,
        T_azi_mean[:, :, track_index].transpose(),
        levels=np.arange(250, 300, 10), colors='darkblue', linewidths=1)
    ax.clabel(T_contour, colors='darkblue', fmt='%1.0f')

    # plt.savefig(os.path.join(args.datadir, 'test.pdf'))
    # plt.savefig(os.path.join(args.datadir, 'test.png'), dpi=300)
    plt.show()


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
    parser.add_argument('--u', type=str,
                        help='zonal wind field',
                        default='UGRD')
    parser.add_argument('--v', type=str,
                        help='meridional wind field',
                        default='VGRD')
    parser.add_argument('--T', type=str,
                        help='temperature field',
                        default='TMP')
    parser.add_argument('--RH', type=str,
                        help='relative humidity field',
                        default='RH')
    parser.add_argument('--vars', type=str,
                        help='additional variables to process',
                        default=',')
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
    Variable list
    """
    var_list = args.vars.split(',')
    logging.info(('vars', var_list))

    """
    Open dataset
    """
    ds = read_tcrmw(filename_in)

    """
    Compute interpolation weights
    """
    compute_interpolation_weights(args, ds, levels)

    """
    Compute tangential and radial wind components
    """
    u_radial, u_tangential = compute_wind_components(args, ds)

    """
    Write dataset
    """
    ds['u_radial'] = xr.DataArray(u_radial, coords=ds[args.T].coords)
    ds['u_tangential'] = xr.DataArray(u_tangential, coords=ds[args.T].coords)
    ds.to_netcdf(filename_out)

    """
    Test plots
    """
    test_plot_pressure_lev(args, ds)
