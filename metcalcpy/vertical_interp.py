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
import numpy as np
import pandas as pd
import xarray as xr # http://xarray.pydata.org/
import netCDF4 as nc

"""
Import Pint and MetPy modules
    https://unidata.github.io/MetPy/
"""
import pint
from metpy import calc, constants

def vertical_interp(config,
    coordinate_surfaces, field):
    """
    Interpolate field onto coordinate surfaces.
    Linear interpolation is the only method currently implemented.

    Arguments:
        config (dictionary) : configuration parameters
        coordinate_surfaces (DataArray): coordinate surfaces
        field (DataArray): field

    Returns:
        field_interp (DataArray): Interpolated field
    """

    """
    Vertical coordinates
    """
    lev_dim = config['vertical_dim_name']
    vertical_coord = field.coords[lev_dim]
    nlev = len(vertical_coord)
    vertical_indices = np.arange(nlev)

    """
    Vertical coordinates on which to interpolate
    """
    vertical_levels = np.array(config['vertical_levels'], dtype=field.dtype)
    nlev_interp = len(vertical_levels)
    logging.debug(vertical_levels)
    logging.debug(coordinate_surfaces.shape)

    """
    Setup interpolated field shape
    """
    dims_interp = list(field.dims)
    i_lev_dim = dims_interp.index(lev_dim)
    shape_interp = list(field.shape)
    shape_interp[i_lev_dim] = nlev_interp
    shape_interp = tuple(shape_interp)
    logging.debug(shape_interp)

    """
    Setup dimensions and shape for a vertical slice
    """
    dims_slice = list(field.dims)
    dims_slice.pop(i_lev_dim)
    dims_slice = tuple(dims_slice)
    logging.debug(dims_slice)
    shape_slice = list(field.shape)
    shape_slice.pop(i_lev_dim)
    shape_slice = tuple(shape_slice)
    logging.debug(shape_slice)
    field_slice = field.drop(lev_dim)
    coords_slice = field_slice.coords
    logging.debug(coords_slice)

    """
    Initialize interpolated field
    """
    field_interp = xr.DataArray(
        np.zeros(shape_interp),
        dims = field.dims,
        attrs = field.attrs)

    for k_interp, eta in zip(range(nlev_interp), vertical_levels):
        """
        Compute interpolation weights
        Todo: unit conversion
        """

        logging.debug((k_interp, eta))

        weights = xr.DataArray(
            np.zeros(field.shape),
            dims = field.dims,
            coords = field.coords)

        field_slice = xr.DataArray(
            np.zeros(field_slice.shape),
            dims = field_slice.dims,
            coords = field_slice.coords,
        )

        distances = eta - coordinate_surfaces
        above = distances < 0
        below = distances > 0

        # where bottom most layer is above eta
        layer_above = above.loc[{lev_dim: vertical_coord[0]}]
        weights.loc[{lev_dim: vertical_coord[0]}] \
            = xr.where(layer_above, 1, 0)

        field_slice = field_slice \
            + weights.loc[{lev_dim: vertical_coord[0]}] \
            * field.loc[{lev_dim: vertical_coord[0]}]

        for k in vertical_indices[1:]:
            layer_above = above.loc[{lev_dim: vertical_coord[k]}]
            layer_below = below.loc[{lev_dim: vertical_coord[k - 1]}]
            # where upper layer is above and lower layer is below eta
            mask = np.logical_and(layer_above, layer_below)

            # w(z_1) = 1 - (z_1 - eta) / (z_1 - z_0) =  (eta - z_0) / (z_1 - z_0)
            weight_above = distances.loc[{lev_dim: vertical_coord[k - 1]}] \
                / (coordinate_surfaces.loc[{lev_dim: vertical_coord[k]}]
                 - coordinate_surfaces.loc[{lev_dim: vertical_coord[k - 1]}])
            # w(z_0) = 1 - w_1 = (z_1 - eta) / (z_1 - z_0)
            weight_below = 1 - weight_above

            weights.loc[{lev_dim: vertical_coord[k]}] \
                = xr.where(mask, weight_above,
                           weights.loc[{lev_dim: vertical_coord[k]}])
            weights.loc[{lev_dim: vertical_coord[k - 1]}] \
                = xr.where(mask, weight_below,
                           weights.loc[{lev_dim: vertical_coord[k - 1]}])

            # field_slice = field_slice \
            #     + weights.loc[{lev_dim: vertical_coord[k]}] \
            #     * field.loc[{lev_dim: vertical_coord[k]}]

        # where top most layer is below eta
        layer_below = below.loc[{lev_dim: vertical_coord[nlev - 1]}]
        weights.loc[{lev_dim: vertical_coord[nlev - 1]}] \
            = xr.where(layer_below, 1, 0)

        # field_slice = field_slice \
        #     + weights.loc[{lev_dim: vertical_coord[nlev - 1]}] \
        #     * field.loc[{lev_dim: vertical_coord[nlev - 1]}]

        """
        Write fields for debugging
        """
        if (logging.root.level == logging.DEBUG):
            ds_debug = xr.Dataset(
                {'distances' : distances,
                 'weights' : weights})
            debugfile = os.path.join(args.debugdir,
                'vertical_interp_debug_' + str(int(eta)) + '.nc')
            try:
                ds_debug.to_netcdf(debugfile)
            except:
                ds_nc = nc.Dataset(debugfile, 'w')
                write_dataset(ds_debug, ds_nc)
                ds_nc.close()

    return field_interp

def height_from_pressure(config,
    surface_geopotential, surface_pressure,
    temperature, relative_humidity):
    """
    Compute height coordinate surfaces as a function of pressure.

    Arguments:
        config (dictionary) : configuration parameters
        surface_geopotential (DataArray) : surface geopotential
        surface_pressure (DataArray) : surface pressure
        temperature (DataArray) : temperature
        relative_humidity (DataArray) : relative humidity

    Returns:
        layer_height (DataArray) : layer height
    """

    ureg = pint.UnitRegistry()
    logging.info('pressure to height conversion')

    """
    Compute surface geopotential height from geopotential
    """
    surface_height = xr.DataArray(
        surface_geopotential / constants.earth_gravity.to_base_units(),
        dims = surface_geopotential.dims,
        coords = surface_geopotential.coords,
        attrs = {'long_name': 'surface geopotential height',
                 'units': 'meter'})

    """
    Get pressure coordinates
    """
    logging.debug(temperature.coords)
    logging.debug(temperature.shape)
    lev_dim = config['vertical_dim_name']
    pressure_coord = temperature.coords[lev_dim]
    nlev = len(pressure_coord)
    pressure_indices = np.arange(nlev)

    """
    Create pressure field
    """
    pressure = xr.DataArray(
        np.empty(temperature.shape),
        dims=temperature.dims,
        coords=temperature.coords,
        attrs={'long_name' : 'pressure',
               'units' : pressure_coord.attrs['units']})
    for p in pressure_coord:
        pressure.loc[{lev_dim:p}] = p

    """
    Compute mixing ratio
    """
    mixing_ratio \
        = xr.DataArray(
            calc.mixing_ratio_from_relative_humidity(
                relative_humidity, temperature, pressure),
        dims=temperature.dims,
        coords=temperature.coords,
        attrs={'long_name' : 'mixing ratio'})

    """
    Compute virtual temperature
    """
    virtual_temperature \
        = xr.DataArray(
        calc.virtual_temperature(temperature, mixing_ratio),
    dims=temperature.dims,
    coords=temperature.coords,
    attrs={'long_name' : 'virtual temperature',
           'units' : temperature.attrs['units']})

    """
    Compute layer thickness
    Z_2 - Z_1 = (R_d / g) <T_v> log(p_1 / p_2)
    R_d / g = dry_air_gas_constant / earth_gravity
    <T_v> = integral_p_2^p_1 T_v(p) (dp / p) / log(p_1 / p_2)
    """
    gas_constant_gravity_ratio \
        = (constants.dry_air_gas_constant \
        / constants.earth_gravity).to_base_units()
    logging.debug(gas_constant_gravity_ratio)
    logging.debug(surface_pressure.attrs['units'])
    logging.debug(pressure.attrs['units'])

    # pressure unit conversion
    pressure_convert = (ureg.Quantity(1, surface_pressure.attrs['units'])
        / ureg.Quantity(1, pressure.attrs['units'])).to_base_units()
    logging.debug(pressure_convert)

    layer_thickness = xr.DataArray(
        np.empty(temperature.shape),
        dims=temperature.dims,
        coords=temperature.coords,
        attrs={'long_name' : 'layer thickness',
               'units' : 'meter'})

    surface_mask = xr.DataArray(
        np.empty(temperature.shape, dtype=np.bool),
        dims=temperature.dims,
        coords=temperature.coords,
        attrs={'long_name' : 'surface mask'})

    layer_height = xr.DataArray(
        np.empty(temperature.shape),
        dims=temperature.dims,
        coords=temperature.coords,
        attrs={'long_name' : 'layer height',
               'units' : 'meter'})

    layer_thickness.loc[{lev_dim:pressure_coord[0]}] \
        = gas_constant_gravity_ratio \
        * virtual_temperature.loc[{lev_dim:pressure_coord[0]}] \
        * np.log(pressure_convert * surface_pressure
                 / pressure.loc[{lev_dim:pressure_coord[0]}])

    for k in pressure_indices[1:]:
        # logging.debug(k)
        layer_thickness.loc[{lev_dim:pressure_coord[k]}] \
            = gas_constant_gravity_ratio \
            * virtual_temperature.loc[{lev_dim:pressure_coord[k]}] \
            * np.log(pressure.loc[{lev_dim:pressure_coord[k - 1]}] \
            / pressure.loc[{lev_dim:pressure_coord[k]}])

            # * 0.5 * (virtual_temperature.loc[{lev_dim:pressure_coord[k - 1]}]
            # + virtual_temperature.loc[{lev_dim:pressure_coord[k]}]) \

    layer_thickness = layer_thickness.fillna(0)
    layer_thickness = layer_thickness.clip(min = 0)

    """
    Compute layer height and surface mask
        The surface mask value value is true if the surface pressure
        is between the lower layer and upper (current) layers.
    """
    surface_mask.loc[{lev_dim: pressure_coord[0]}] \
        = pressure.loc[{lev_dim: pressure_coord[0]}] \
        < pressure_convert * surface_pressure

    layer_height.loc[{lev_dim:pressure_coord[0]}] \
        = xr.where(surface_mask.loc[{lev_dim: pressure_coord[0]}],
            surface_height + layer_thickness.loc[{lev_dim: pressure_coord[0]}],
            np.nan)

    for k in pressure_indices[1:]:

        surface_mask.loc[{lev_dim: pressure_coord[k]}] \
            = np.logical_and(
            pressure.loc[{lev_dim: pressure_coord[k - 1]}]
            > pressure_convert * surface_pressure,
            pressure.loc[{lev_dim: pressure_coord[k]}]
            < pressure_convert * surface_pressure)

        layer_height.loc[{lev_dim:pressure_coord[k]}] \
            = xr.where(surface_mask.loc[{lev_dim: pressure_coord[k]}],
                surface_height + layer_thickness.loc[{lev_dim: pressure_coord[k]}],
                layer_height.loc[{lev_dim: pressure_coord[k - 1]}]
                + layer_thickness.loc[{lev_dim: pressure_coord[k]}])

    """
    Write fields for debugging
    Approximate layer heights
        800 hPa ~ 2 km, 500 hPa ~ 6 km, 200 hPa ~ 12 km
    """
    if (logging.root.level == logging.DEBUG):
        ds_debug = xr.Dataset(
            {'surface_geopotential' : surface_geopotential,
             'surface_height' : surface_height,
             'surface_pressure' : surface_pressure,
             'surface_mask' : surface_mask,
             'pressure' : pressure,
             'mixing_ratio' : mixing_ratio,
             'virtual_temperature' : virtual_temperature,
             'layer_thickness': layer_thickness,
             'layer_height': layer_height})
        debugfile = os.path.join(args.debugdir,
            'height_from_pressure_debug.nc')
        try:
            ds_debug.to_netcdf(debugfile)
        except:
            ds_nc = nc.Dataset(debugfile, 'w')
            write_dataset(ds_debug, ds_nc)
            ds_nc.close()

    return layer_height

def read_required_fields(config, ds):
    """
    Read required fields.

    Arguments:
        config (dictionary) : configuration parameters
        ds (DataSet) : xarray dataset

    Returns:
        surface_geopotential (DataArray) : surface geopotential
        surface_pressure (DataArray) : surface pressure
        temperature (DataArray) : temperature
        relative_humidity (DataArray) : relative humidity
    """
    surface_geopotential \
        = ds[config['surface_geopotential_name']]
    surface_pressure \
        = ds[config['surface_pressure_name']]
    temperature \
        = ds[config['temperature_name']]
    relative_humidity \
        = ds[config['relative_humidity_name']]
    return surface_geopotential, surface_pressure, \
        temperature, relative_humidity

def write_dataset(ds, ds_nc):
    """
    Write xarray Dataset to NetCDF file
    """
    for dim in ds.dims:
        logging.info('Creating dimension ' + dim)
        ds_nc.createDimension(dim, len(ds.coords[dim]))
        coord = ds_nc.createVariable(
            dim, ds.coords[dim].dtype, (dim))
        coord[:] = ds.coords[dim].values

    if 'time' in ds.coords:
        time = pd.Timestamp(ds.coords['time'].values)
        logging.debug(time)
        date_int \
            = 10000 * time.year + 100 * time.month \
            + time.day
        time_int \
            = 10000 * time.hour + 100 * time.minute \
            + time.second
        ds_nc.time = 1000000 * date_int + time_int

    if 'valid_time' in ds.coords:
        valid_time = pd.Timestamp(ds.coords['valid_time'].values)
        logging.debug(valid_time)
        date_int \
            = 10000 * valid_time.year + 100 * valid_time.month \
            + valid_time.day
        time_int \
            = 10000 * valid_time.hour + 100 * valid_time.minute \
            + valid_time.second
        ds_nc.valid_time = 1000000 * date_int + time_int

    for field in ds:
        logging.debug('Creating variable ' + field)
        var = ds_nc.createVariable(
            field, ds[field].dtype, ds[field].dims)
        var[:] = ds[field].values
        for attr in ds[field].attrs:
            logging.debug((attr, ds[field].attrs[attr]))
            setattr(var, attr, ds[field].attrs[attr])

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
    parser.add_argument('--debugdir', type=str,
        default=os.path.join(os.getenv('DATA_DIR'), 'Debug'),
        help='debug file directory (default $DATA_DIR/Debug)')
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
    Construct input and output filenames
    """
    filename_in = os.path.join(args.datadir, args.input)
    filename_out = os.path.join(args.datadir, args.output)

    """
    Read YAML configuration file
    """
    config = yaml.load(
        open(args.config), Loader=yaml.FullLoader)
    logging.info(config)

    """
    Read dataset
    """
    try:
        if (filename_in.split('.')[-1] == 'grb2'):
            logging.info('Opening GRIB2 ' + filename_in)
            ds = xr.open_dataset(filename_in, engine='cfgrib',
                backend_kwargs={'filter_by_keys':{'typeOfLevel': 'isobaricInhPa'}})
        else:
            logging.info('Opening NetCDF ' + filename_in)
            ds = xr.open_dataset(filename_in)
    except:
        logging.error('Unable to open ' + filename_in)
        logging.error(sys.exc_info()[0])

    logging.debug(ds)

    """
    Convert pressure levels to height levels
    """
    if (config['height_from_pressure']
        and config['vertical_coord_type_in'] == 'pressure'
        and config['vertical_coord_type_out'] == 'height'):

        surface_geopotential, surface_pressure, \
            temperature, relative_humidity \
                = read_required_fields(config, ds)

        layer_height = height_from_pressure(config,
            surface_geopotential, surface_pressure,
            temperature, relative_humidity)

    else:

        layer_height = ds[config['geopotential_height_name']]

    """
    Interpolate
    """
    ds_out = xr.Dataset()

    for field in config['fields']:
        logging.info(field)

        field_interp = vertical_interp(config,
            layer_height, ds[field])

        ds_out[field] = field_interp

    """
    Write dataset
    """
    try:
        logging.info('Creating with xarray ' + filename_out)
        ds_out.to_netcdf(filename_out)
    except:
        logging.error('Unable to create ' + filename_out)
        try:
            logging.info('Creating with NetCDF4 ' + filename_out)
            ds_nc = nc.Dataset(filename_out, 'w')
            write_dataset(ds_out, ds_nc)
        except:
            logging.error('Unable to create ' + filename_out)
