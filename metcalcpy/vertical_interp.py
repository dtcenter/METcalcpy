# ============================*
 # ** Copyright UCAR (c) 2020
 # ** University Corporation for Atmospheric Research (UCAR)
 # ** National Center for Atmospheric Research (NCAR)
 # ** Research Applications Lab (RAL)
 # ** P.O.Box 3000, Boulder, Colorado, 80307-3000, USA
 # ============================*
 
 
 
"""
Program Name: vertical_interp.py

A python script to vertically interpolate fields
between grids with pressure or height vertical coordinates.

Currently only pressure to height conversion is implemented.
Currently only linear interpolation is implemented.

Version  Date
0.1.0    2020/09/01  David Fillmore  Initial version

Known Issues:
    xarray does not propagate coordinate attributes to interpolation field
    need to handle unknown units

Needed Enhancements:
    multiple input files on command line or config file
    config file definition of user defined units
    config file option for interpolation weights
        above (below) the highest (lowest) coordinate surface
"""

__author__ = 'David Fillmore'
__version__ = '0.1.0'

import metpy.units

"""
Import standard modules
"""
import os
import sys
import argparse
import logging
import yaml
from datetime import datetime
import numpy as np
import xarray as xr  # http://xarray.pydata.org/
import netCDF4 as nc

"""
Import Pint and MetPy modules
    https://unidata.github.io/MetPy/
"""
import pint
import metpy.calc as calc
import metpy.constants as constants

ureg = pint.UnitRegistry()

def vertical_interp(fieldname, config,
    coordinate_surfaces, field):
    """
    Interpolate field onto coordinate surfaces.
    Linear interpolation is the only method currently implemented.

    Arguments:
        fieldname (string) : short name of field
        config (dictionary) : configuration parameters
        coordinate_surfaces (DataArray): coordinate surfaces
        field (DataArray): field

    Returns:
        field_interp (DataArray): Interpolated field
    """
    logging.info(fieldname)
    logging.debug(field.attrs)

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
    Setup interpolated field shape and coordinates
    """
    dims_interp = list(field.dims)
    i_lev_dim = dims_interp.index(lev_dim)
    logging.debug(('dims_interp:', dims_interp))
    shape_interp = list(field.shape)
    shape_interp[i_lev_dim] = nlev_interp
    shape_interp = tuple(shape_interp)
    logging.debug(shape_interp)
    coord_arrays_interp = [field.coords[dim].data for dim in dims_interp]
    dims_interp[i_lev_dim] = 'lev'
    """
    coord_arrays_interp[i_lev_dim] \
        = xr.DataArray(vertical_levels,
                       attrs={'units': config['vertical_level_units']})
    """
    coord_arrays_interp[i_lev_dim] = vertical_levels
    coords_interp = list(zip(dims_interp, coord_arrays_interp))
    logging.debug('\n\n')
    for coord_interp in coords_interp:
        logging.debug(coord_interp)
        logging.debug('\n\n')

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
        dims=dims_interp,
        coords=coords_interp,
        attrs=field.attrs)
    # coordinate attributes lost in constructor above

    # length unit conversion
    try:
        logging.debug(coordinate_surfaces.attrs['units'])
        logging.debug(config['vertical_level_units'])
        length_convert = float((ureg.Quantity(1, config['vertical_level_units'])
                       / ureg.Quantity(1, coordinate_surfaces.attrs['units'])).to_base_units())
    except pint.errors.UndefinedUnitError:
        logging.warning('Unknown unit:' + coordinate_surfaces.attrs['units'])
        length_convert = 1
    logging.debug(length_convert)

    for k_interp, eta in zip(range(nlev_interp), vertical_levels):
        """
        Compute interpolation weights
        """

        logging.debug((k_interp, eta))

        weights = xr.DataArray(
            np.zeros(field.shape),
            dims=field.dims,
            coords=field.coords)

        distances = eta - length_convert * coordinate_surfaces
        above = distances < 0
        below = distances > 0

        """
        Compute weights for linear interpolation
        """
        # where bottom most layer is above eta
        layer_above = above.loc[{lev_dim: vertical_coord[0]}]
        weights.loc[{lev_dim: vertical_coord[0]}] \
            = xr.where(layer_above, 1, 0)

        for k in vertical_indices[1:]:
            layer_above = above.loc[{lev_dim: vertical_coord[k]}]
            layer_below = below.loc[{lev_dim: vertical_coord[k - 1]}]
            # where upper layer is above and lower layer is below eta
            mask = np.logical_and(layer_above, layer_below)

            # w(z_1) = 1 - (z_1 - eta) / (z_1 - z_0) =  (eta - z_0) / (z_1 - z_0)
            weight_above = distances.loc[{lev_dim: vertical_coord[k - 1]}] \
                / ((coordinate_surfaces.loc[{lev_dim: vertical_coord[k]}]
                    - coordinate_surfaces.loc[{lev_dim: vertical_coord[k - 1]}])
                    * length_convert)
            # w(z_0) = 1 - w_1 = (z_1 - eta) / (z_1 - z_0)
            weight_below = 1 - weight_above

            weights.loc[{lev_dim: vertical_coord[k]}] \
                = xr.where(mask, weight_above,
                           weights.loc[{lev_dim: vertical_coord[k]}])
            weights.loc[{lev_dim: vertical_coord[k - 1]}] \
                = xr.where(mask, weight_below,
                           weights.loc[{lev_dim: vertical_coord[k - 1]}])

        # where top most layer is below eta
        layer_below = below.loc[{lev_dim: vertical_coord[nlev - 1]}]
        weights.loc[{lev_dim: vertical_coord[nlev - 1]}] \
            = xr.where(layer_below, 1, 0)

        """
        Compute weighted sum
        """
        field_slice = xr.DataArray(
            np.zeros(shape_slice),
            dims=dims_slice,
            coords=coords_slice,
            attrs=field.attrs)

        counts = xr.DataArray(
            np.zeros(shape_slice),
            dims=dims_slice,
            coords=coords_slice,
            attrs=field.attrs)

        for k in vertical_indices:
            weights_k = weights.loc[{lev_dim: vertical_coord[k]}]
            field_k = field.loc[{lev_dim: vertical_coord[k]}]
            mask = weights_k > 0
            field_slice = xr.where(mask,
                                   field_slice + weights_k * field_k, field_slice)
            counts = xr.where(mask, counts + 1, counts)

        mask = counts > 0
        field_slice = xr.where(mask, field_slice, np.nan)

        # field_interp[dict(isobaricInhPa=k_interp)] = field_slice
        # field_interp[{lev_dim: k_interp}] = field_slice
        field_interp[dict(lev=k_interp)] = field_slice

        """
        Write fields for debugging
        """
        if logging.root.level == logging.DEBUG:
            ds_debug = xr.Dataset(
                {'distances': distances,
                 'weights': weights,
                 fieldname: field_slice})
            debugfile = os.path.join(args.datadir, 'Debug', 'vertical_interp_debug_'
                                     + fieldname + '_' + str(int(eta)) + '.nc')
            try:
                ds_debug.to_netcdf(debugfile)
            except:
                ds_nc = nc.Dataset(debugfile, 'w')
                write_dataset(ds_debug, ds_nc)
                ds_nc.close()

    return field_interp, coords_interp


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

    logging.info('pressure to height conversion')

    """
    Compute surface geopotential height from geopotential
    """
    surface_height = xr.DataArray(
        surface_geopotential / constants.earth_gravity.to_base_units().magnitude,
        dims=surface_geopotential.dims,
        coords=surface_geopotential.coords,
        attrs={'long_name': 'surface geopotential height',
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
        attrs={'long_name': 'pressure',
               'units': pressure_coord.attrs['units']})
    for p in pressure_coord:
        pressure.loc[{lev_dim: p}] = p

    """
    Compute mixing ratio
    """
    mixing_ratio \
        = xr.DataArray(
            calc.mixing_ratio_from_relative_humidity(
                pressure, temperature, relative_humidity),
            dims=temperature.dims,
            coords=temperature.coords,
            attrs={'long_name': 'mixing ratio'})

    """
    Compute virtual temperature
    """
    virtual_temperature \
        = xr.DataArray(
            calc.virtual_temperature(temperature, mixing_ratio),
            dims=temperature.dims,
            coords=temperature.coords,
            attrs={'long_name': 'virtual temperature',
                   'units': temperature.attrs['units']})

    """
    Compute layer thickness
    Z_2 - Z_1 = (R_d / g) <T_v> log(p_1 / p_2)
    R_d / g = dry_air_gas_constant / earth_gravity
    <T_v> = integral_p_2^p_1 T_v(p) (dp / p) / log(p_1 / p_2)
    """
    gas_constant_gravity_ratio \
        = (constants.dry_air_gas_constant
        / constants.earth_gravity).to_base_units()
    logging.debug(gas_constant_gravity_ratio)
    logging.debug(surface_pressure.attrs['units'])
    logging.debug(pressure.attrs['units'])

    # pressure unit conversion
    pressure_convert = float((ureg.Quantity(1, surface_pressure.attrs['units'])
        / ureg.Quantity(1, pressure.attrs['units'])).to_base_units())
    logging.debug(pressure_convert)

    layer_thickness = xr.DataArray(
        np.empty(temperature.shape),
        dims=temperature.dims,
        coords=temperature.coords,
        attrs={'long_name': 'layer thickness',
               'units': 'meter'})

    surface_mask = xr.DataArray(
        np.empty(temperature.shape, dtype=bool),
        dims=temperature.dims,
        coords=temperature.coords,
        attrs={'long_name': 'surface mask'})

    layer_height = xr.DataArray(
        np.empty(temperature.shape),
        dims=temperature.dims,
        coords=temperature.coords,
        attrs={'long_name': 'layer height',
               'units': 'meter'})

    layer_thickness.loc[{lev_dim: pressure_coord[0]}] \
        = gas_constant_gravity_ratio \
        * virtual_temperature.loc[{lev_dim: pressure_coord[0]}] \
        * np.log(pressure_convert * surface_pressure
                 / pressure.loc[{lev_dim: pressure_coord[0]}])

    for k in pressure_indices[1:]:
        # logging.debug(k)
        layer_thickness.loc[{lev_dim:pressure_coord[k]}] \
            = gas_constant_gravity_ratio \
            * virtual_temperature.loc[{lev_dim: pressure_coord[k]}] \
            * np.log(pressure.loc[{lev_dim: pressure_coord[k - 1]}] \
            / pressure.loc[{lev_dim: pressure_coord[k]}])

    layer_thickness = layer_thickness.fillna(0)
    layer_thickness = layer_thickness.clip(min=0)

    """
    Compute layer height and surface mask
        The surface mask value value is true if the surface pressure
        is between the lower layer and upper (current) layers.
    """
    surface_mask.loc[{lev_dim: pressure_coord[0]}] \
        = pressure.loc[{lev_dim: pressure_coord[0]}] \
        < pressure_convert * surface_pressure
    layer_height.loc[{lev_dim: pressure_coord[0]}] \
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

        layer_height.loc[{lev_dim: pressure_coord[k]}] \
            = xr.where(surface_mask.loc[{lev_dim: pressure_coord[k]}],
                surface_height + layer_thickness.loc[{lev_dim: pressure_coord[k]}],
                layer_height.loc[{lev_dim: pressure_coord[k - 1]}]
                + layer_thickness.loc[{lev_dim: pressure_coord[k]}])

    """
    Write fields for debugging
    Approximate layer heights
        800 hPa ~ 2 km, 500 hPa ~ 6 km, 200 hPa ~ 12 km
    """
    if logging.root.level == logging.DEBUG:
        ds_debug = xr.Dataset(
            {'surface_geopotential': surface_geopotential,
             'surface_height': surface_height,
             'surface_pressure': surface_pressure,
             'surface_mask': surface_mask,
             'layer_pressure': pressure,
             'mixing_ratio': mixing_ratio,
             'virtual_temperature': virtual_temperature,
             'layer_thickness': layer_thickness,
             'layer_height': layer_height})
        debugfile = os.path.join(args.datadir, 'Debug',
            'height_from_pressure_debug.nc')
        
        # create the Debug directory if it doesn't already exist
        debug_path = os.path.join(args.datadir, 'Debug')
        if not os.path.exists(debug_path):
            os.makedirs(debug_path)
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
    surface_pressure \
        = ds[config['surface_pressure_name']]

    if config['zero_surface_geopotential']:
        surface_geopotential = xr.zeros_like(surface_pressure)
    else:
        surface_geopotential \
            = ds[config['surface_geopotential_name']]

    temperature \
        = ds[config['temperature_name']]
    relative_humidity \
        = ds[config['relative_humidity_name']]
    return surface_geopotential, surface_pressure, \
        temperature, relative_humidity


def write_dataset(ds, ds_nc, coords_interp=None,
    forecast_reference_time=None, create_time_dim=False):
    """
    Write xarray Dataset to NetCDF file
    """
    coord_vars = {}
    for dim in ds.dims:
        logging.info('Creating dimension ' + dim)
        ds_nc.createDimension(dim, len(ds.coords[dim]))
        dtype = ds.coords[dim].dtype
        if dtype not in ['uint32', 'uint64', 'int32', 'int64', 'float32', 'float64']:
            dtype = 'float64'
        coord = ds_nc.createVariable(
            dim, dtype, (dim))
        coord_vars[dim] = coord
        if dim != 'time':
            coord[:] = ds.coords[dim].values
        else:
            dt_array = [datetime.utcfromtimestamp(dt.astype('O')/1e9)
                        for dt in ds.coords[dim].values]
            t_array = np.array([(dt - dt_array[0]).total_seconds()
                                for dt in dt_array], dtype=np.float64)
            coord[:] = t_array

    if create_time_dim:
        ds_nc.createDimension('valid_time', 1)
        time_coord = ds_nc.createVariable(
            'valid_time', 'float64', ('valid_time'))
        dt_valid = datetime.utcfromtimestamp(
            ds['valid_time'].astype('O')/1e9)
        dt_init = datetime.utcfromtimestamp(
            ds['init_time'].astype('O')/1e9)
        time_coord[:] = (dt_valid - dt_init).total_seconds()
        time_coord.long_name = 'valid_time'
        time_coord.units = 'seconds since ' + str(dt_init)

    if coords_interp is not None:
        for dim, coord_array in coords_interp:
            logging.info('Setting coordinate attributes for ' + dim)
            """
            for attr in coord_array.attrs:
                setattr(coord_vars[dim], attr, coord_array.attrs[attr])
            """

    for field in ds:
        logging.debug('Creating variable ' + field)
        dtype = ds[field].dtype
        if dtype not in ['uint32', 'uint64', 'int32', 'int64', 'float32', 'float64']:
            dtype = 'uint64'
        if create_time_dim:
            dims_with_time = list(ds[field].dims)
            dims_with_time.insert(0, 'valid_time')
            if field != 'valid_time':
                var = ds_nc.createVariable(
                    field, dtype, tuple(dims_with_time))
                var[:] = ds[field].values
        else:
            var = ds_nc.createVariable(
                field, dtype, ds[field].dims)
            var[:] = ds[field].values

        for attr in ds[field].attrs:
            logging.debug((attr, ds[field].attrs[attr]))
            setattr(var, attr, ds[field].attrs[attr])

    for attr in ds.attrs:
        logging.debug((attr, ds.attrs[attr]))
        setattr(ds_nc, attr, ds.attrs[attr])

    if forecast_reference_time is not None:
        # setattr(ds_nc,
        #     'forecast_reference_time', forecast_reference_time)
        ds_nc.createDimension('forecast_reference_time', 1)
        ref_time_coord = ds_nc.createVariable(
            'forecast_reference_time', 'float64',
            ('forecast_reference_time'))
        ref_time_coord.long_name = 'forecast_reference_time'
        ref_time_coord.standard_name = 'forecast_reference_time'
        ref_time_coord.units = 'seconds since 1970-01-01 00:00'
        yyyymmddhh_str = str(forecast_reference_time)
        yyyy = int(yyyymmddhh_str[0:4])
        mm = int(yyyymmddhh_str[4:6])
        dd = int(yyyymmddhh_str[6:8])
        hh = int(yyyymmddhh_str[8:10])
        ref_time_obj = datetime(yyyy, mm, dd, hh)
        ref_time_coord[:] = ref_time_obj.timestamp()


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
    parser.add_argument('--create_time_dim', action='store_true',
        help='create time dimension in netcdf output')
    parser.add_argument('--ref_time_from_filename', action='store_true',
        help='extract forecast reference time from filename')
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
        if filename_in.split('.')[-1] == 'grb2' or 'pgrb2' in filename_in:
            logging.info('Opening GRIB2 ' + filename_in)
            ds = xr.open_dataset(filename_in, engine='cfgrib',
                backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa'}})
        else:
            logging.info('Opening NetCDF ' + filename_in)
            ds = xr.open_dataset(filename_in)
        logging.debug(ds)
    except:
        ds = xr.Dataset()
        logging.error('Unable to open ' + filename_in)
        logging.error(sys.exc_info()[0])

    """
    if 'valid_time' in ds:
        logging.info(datetime.utcfromtimestamp(
                     ds['valid_time'].astype('O')/1e9))
        logging.info(datetime.utcfromtimestamp(
                     ds['time'].astype('O')/1e9))
    """

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
    if 'valid_time' in ds:
        ds_out['valid_time'] = ds['valid_time'].values
        if 'time' in ds:
            ds_out['init_time'] = ds['time'].values

    ds_out['lat'] = ds['lat']
    ds_out['lon'] = ds['lon']

    for attr in ds.attrs:
        ds_out.attrs[attr] = ds.attrs[attr]

    for fieldname in config['fields']:

        field_interp, coords_interp = vertical_interp(fieldname, config,
            layer_height, ds[fieldname])

        ds_out[fieldname] = field_interp

    """
    Write dataset
    """
    try:
        logging.info('Creating with NetCDF4 ' + filename_out)
        ds_nc = nc.Dataset(filename_out, 'w')
        if args.ref_time_from_filename: 
            ref_time = filename_out.split('.')[1]
            write_dataset(ds_out, ds_nc, coords_interp=coords_interp,
                forecast_reference_time=ref_time, create_time_dim=args.create_time_dim)
        else:
            write_dataset(ds_out, ds_nc, coords_interp=coords_interp,
                create_time_dim=args.create_time_dim)
        ds_nc.close()
    except:
        logging.info('Creating with xarray ' + filename_out)
        ds_nc.to_netcdf(filename_out)

    logging.debug(os.system('ncdump -h ' + filename_out))
