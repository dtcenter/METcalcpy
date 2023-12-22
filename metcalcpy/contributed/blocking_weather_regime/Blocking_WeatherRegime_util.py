# ============================*
 # ** Copyright UCAR (c) 2020
 # ** University Corporation for Atmospheric Research (UCAR)
 # ** National Center for Atmospheric Research (NCAR)
 # ** Research Applications Lab (RAL)
 # ** P.O.Box 3000, Boulder, Colorado, 80307-3000, USA
 # ** University of Illinois, Urbana-Champaign
 # ============================*
 
 
 
import os
import netCDF4
import numpy as np
import datetime
from scipy import stats


def parse_steps():
    """
    Function to parse the steps for the Blocking and weather regime Calculations
    and then return them to the driver
    :return: Lists containing the forecast and observation steps
    :rtype: List of strings
    """

    steps_param_fcst = os.environ.get('FCST_STEPS','')
    steps_list_fcst = steps_param_fcst.split("+")

    steps_param_obs = os.environ.get('OBS_STEPS','')
    steps_list_obs = steps_param_obs.split("+")

    return steps_list_fcst, steps_list_obs


def get_filenames_list(filetxt):
    """
    Function that opens a text file containing the listing of input files, remove the header
    which may say file_list and then return them for reading
    :param filetxt:
        input filename that contains the listing of input files
    :type filetxt: string
    :return: List containing the names of the input files
    :rtype: List of strings
    """

    # Read input file
    with open(filetxt) as ft:
        data_infiles = ft.read().splitlines()
    # Remove the first line if it's there
    if (data_infiles[0] == 'file_list'):
        data_infiles = data_infiles[1:]

    return data_infiles


def read_nc_met(infiles,invar,nseasons,dseasons):
    """
    Function to read in MET version netCDF data specifically for the blocking and weather regime
    calculations.  The output array needs to be in a specific format.
    :param infiles: 
        List of full paths to filenames of the data to read in
    :param invar: 
        Variable name in the file of the data to read in
    :param nseasons:
        The number of years the input data contains
    :param dseasons:
        The number of days in each year (must be equal for all input years)
    :type infiles: List of strings
    :type invar: String
    :type nseasons: Integer
    :type dseasons: Integer
    return: 4D array of data [year, day, lat, lon], latitude array, longitude array, and a time dictionary
    rtype: numpy array, numpy array, numpy array, dictionary
    """

    print("Reading in Data")

    # Check to make sure that everything is not set to missing:
    if all('missing' == fn for fn in infiles):
        raise Exception('No input files found as given, check paths to input files')

    #Find the first non empty file name so I can get the variable sizes
    locin = next(sub for sub in infiles if sub != 'missing')
    indata = netCDF4.Dataset(locin)
    lats = indata.variables['lat'][:]
    lons = indata.variables['lon'][:]
    invar_arr = indata.variables[invar][:]
    indata.close()

    var_3d = np.empty([len(infiles),len(invar_arr[:,0]),len(invar_arr[0,:])])
    init_list = []
    valid_list = []
    lead_list = []

    for i in range(0,len(infiles)):

        #Read in the data
        if (infiles[i] != 'missing'):
            indata = netCDF4.Dataset(infiles[i])
            new_invar = indata.variables[invar][:]

            init_time_str = indata.variables[invar].getncattr('init_time')
            valid_time_str = indata.variables[invar].getncattr('valid_time')
            lead_dt = datetime.datetime.strptime(valid_time_str,'%Y%m%d_%H%M%S') - datetime.datetime.strptime(init_time_str,'%Y%m%d_%H%M%S')
            leadmin,leadsec = divmod(lead_dt.total_seconds(), 60)
            leadhr,leadmin = divmod(leadmin,60)
            lead_str = str(int(leadhr)).zfill(2)+str(int(leadmin)).zfill(2)+str(int(leadsec)).zfill(2)
            indata.close()
        else:
            new_invar = np.empty((1,len(var_3d[0,:,0]),len(var_3d[0,0,:])),dtype=np.float64)
            init_time_str = ''
            valid_time_str = ''
            lead_str = ''
            new_invar[:] = np.nan
        init_list.append(init_time_str)
        valid_list.append(valid_time_str)
        lead_list.append(lead_str)
        var_3d[i,:,:] = new_invar

    var_4d = np.reshape(var_3d,[nseasons,dseasons,len(var_3d[0,:,0]),len(var_3d[0,0,:])])

    # Reshape time arrays and store them in a dictionary
    init_list_2d = np.reshape(init_list,[nseasons,dseasons])
    valid_list_2d = np.reshape(valid_list,[nseasons,dseasons])
    lead_list_2d = np.reshape(lead_list,[nseasons,dseasons])
    time_dict = {'init':init_list_2d,'valid':valid_list_2d,'lead':lead_list_2d}

    return var_4d,lats,lons,time_dict
