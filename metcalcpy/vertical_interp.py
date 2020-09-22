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
import xarray as xr # http://xarray.pydata.org/

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

    Arguments:
        config (dictionary) : configuration parameters
        coordinate_surfaces (DataArray): coordinate surfaces
        field (DataArray): field

    Returns:
        field_interp (DataArray): Interpolated field
    """
    # not yet implemented
    lev_dim = config['vertical_dim_name']
    vertical_coord = np.array(config['vertical_levels'], dtype=field.dtype)
    nlev_interp = len(vertical_coord)

    logging.debug(vertical_coord)
    logging.debug(coordinate_surfaces.shape)

    logging.debug(field.dims)
    logging.debug(field.attrs)

    """
    Initialize interpolated field
    """
    field_interp = xr.DataArray(
        np.empty(field.shape),
        dims = field.dims,
        coords = field.coords,
        attrs = field.attrs).isel({lev_dim : slice(None, nlev_interp)})
    # Better way to resize vertical dim of an xarray?
    # Assuming nlev_interp < nlev for now,
    #     if nlev_interp > nlev consider pad method.
    field_interp.coords[lev_dim] = vertical_coord 
    # set coordinate units
    field_interp.coords[lev_dim].attrs['units'] \
        = config['vertical_level_units']
    logging.debug(field_interp.coords)

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
            * 0.5 * (virtual_temperature.loc[{lev_dim:pressure_coord[k - 1]}]
            + virtual_temperature.loc[{lev_dim:pressure_coord[k]}]) \
            * np.log(pressure.loc[{lev_dim:pressure_coord[k - 1]}] \
            / pressure.loc[{lev_dim:pressure_coord[k]}])

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
        ds_debug.to_netcdf('vertical_interp_debug.nc')

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

    filename_in = os.path.join(args.datadir, args.input)

    """
    Read YAML configuration file
    """
    config = yaml.load(
        open(args.config), Loader=yaml.FullLoader)
    logging.info(config)

    """
    Convert pressure levels to height levels
    """
    if (config['vertical_coord_type_in'] == 'pressure'
        and config['vertical_coord_type_out'] == 'height'):

        try:
            logging.info('Opening ' + filename_in)
            ds = xr.open_dataset(filename_in)
        except:
            logging.error('Unable to open ' + filename_in)

        surface_geopotential, surface_pressure, \
            temperature, relative_humidity \
                = read_required_fields(config, ds)

        layer_height = height_from_pressure(config,
            surface_geopotential, surface_pressure,
            temperature, relative_humidity)

        for field in config['fields']:
            logging.info(field)
            field_interp = vertical_interp(config,
                layer_height, ds[field])
