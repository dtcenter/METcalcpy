# ============================*
 # ** Copyright UCAR (c) 2020
 # ** University Corporation for Atmospheric Research (UCAR)
 # ** National Center for Atmospheric Research (NCAR)
 # ** Research Applications Lab (RAL)
 # ** P.O.Box 3000, Boulder, Colorado, 80307-3000, USA
 # ============================*
import os
import numpy as np


#def write_mpr_file(data_obs,data_fcst,lats_in,lons_in,obs_lead,obs_valid,fcst_lead,fcst_valid,mod_name,desc,fcst_var,fcst_unit,fcst_lev,obs_var,obs_unit,obs_lev,maskname,obslev,full_outfile):
def write_mpr_file(data_fcst,data_obs,lats_in,lons_in,fcst_lead,fcst_valid,obs_lead,obs_valid,mod_name,desc,fcst_var,fcst_unit,fcst_lev,obs_var,obs_unit,obs_lev,maskname,obsslev,outdir,outfile_prefix):

    """
    Function to write an output mpr file given a 1d array of observation and forecast data
    Parameters:
    ----------
    data_obs: 1D array float
            observation data to write to MPR file
    data_fcst: 1D array float
            forecast data to write to MPR file
    lats_in: 1D array float
            data latitudes
    lons_in: 1D array float
            data longitudes
    obs_lead: 1D array string of format HHMMSS
            observation lead time
    obs_valid: 1D array string of format YYYYmmdd_HHMMSS
            observation valid time
    fcst_lead: 1D array string of format HHMMSS
            forecast lead time
    fcst_valid: 1D array string of format YYYYmmdd_HHMMSS
            forecast valid time
    mod_name: string
            output model name (the MODEL column in MET)
    desc: string
            output description (the DESC column in MET)
    fcst_var: 1D array string
            forecast variable name
    fcst_unit: 1D array string
            forecast variable units
    fcst_lev: 1D array string
            forecast variable level
    obs_var: 1D array string
            observation variable name
    obs_unit: 1D array string
            observation variable units
    obs_lev: 1D array string
            observation variable level
    maskname: string
            name of the verification masking region
    obsslev: 1D array string
            Pressure level of the observation in hPA or accumulation
            interval in hours 
    outdir: string
            Full path including where the output data should go
    outfile_prefix: string
            Prefix to use for the output filename.  The time stamp will 
            be added in MET's format based off the first forecast time
    """

    """
    Get the data length to create the INDEX and TOTAL variables in the MPR line
    """
    dlength = len(data_obs)
    index_num = np.arange(0,dlength,1)+1

    """
    Get the length of the model, FCST_VAR, FCST_LEV, OBS_VAR, OBS_LEV, VX_MASK, etc for formatting
    """
    mname_len = str(max([5,len(mod_name)])+3)
    desc_len = str(max([4,len(desc)])+3)
    mask_len = str(max([7,len(maskname)])+3)
    fvar_len = str(max([8,max([len(l) for l in fcst_var])])+3)
    funit_len = str(max([8,max([len(l) for l in fcst_unit])])+3)
    flev_len = str(max([8,max([len(l) for l in fcst_lev])])+3)
    ovar_len = str(max([7,max([len(l) for l in obs_var])])+3)
    ounit_len = str(max([8,max([len(l) for l in obs_unit])])+3)
    olev_len = str(max([7,max([len(l) for l in obs_lev])])+3)

    """
    Set up format strings for the header (format_string) and data (format_string2)
    """
    format_string = '%-7s %-'+mname_len+'s %-'+desc_len+'s %-12s %-18s %-18s %-12s %-17s %-17s %-'+fvar_len+'s ' \
        '%-'+funit_len+'s %-'+flev_len+'s %-'+ovar_len+'s %-'+ounit_len+'s %-'+olev_len+'s %-10s %-'+mask_len+'s ' \
        '%-13s %-13s %-13s %-13s %-13s %-13s %-9s\n'
    format_string2 = '%-7s %-'+mname_len+'s %-'+desc_len+'s %-12s %-18s %-18s %-12s %-17s %-17s %-'+fvar_len+'s ' \
        '%-'+funit_len+'s %-'+flev_len+'s %-'+ovar_len+'s %-'+ounit_len+'s %-'+olev_len+'s %-10s %-'+mask_len+'s ' \
        '%-13s %-13s %-13s %-13s %-13s %-13s %-9s %-10s %-10s %-10s %-12.4f %-12.4f %-10s %-10s %-12.4f %-12.4f ' \
        '%-10s %-10s %-10s %-10s\n'

    """
    Create the output directory if it doesn't exist
    """
    if not os.path.exists(outdir):
        os.path.makedirs(outdir)

    """
    Put the timestamp on the output file
    """
    fcst_valid_str = fcst_valid[0]
    ft_stamp = fcst_lead[0]+'L_'+fcst_valid_str[0:8]+'_'+fcst_valid_str[9:15]+'V'
    full_outfile = os.path.join(outdir,outfile_prefix+'_'+ft_stamp+'.stat')

    """
    Write the file
    """
    print('Writing output MPR file: '+full_outfile)
    with open(full_outfile, 'w') as mf:
        # Write the header
        mf.write(format_string % ('VERSION', 'MODEL', 'DESC', 'FCST_LEAD', 'FCST_VALID_BEG', 'FCST_VALID_END',
            'OBS_LEAD', 'OBS_VALID_BEG', 'OBS_VALID_END', 'FCST_VAR', 'FCST_UNITS', 'FCST_LEV', 'OBS_VAR', 
            'OBS_UNITS', 'OBS_LEV', 'OBTYPE', 'VX_MASK', 'INTERP_MTHD', 'INTERP_PNTS', 'FCST_THRESH', 
            'OBS_THRESH', 'COV_THRESH', 'ALPHA', 'LINE_TYPE'))
        for dpt in range(dlength):
            # Write the data
            mf.write(format_string2 % ('V9.1',mod_name,desc,fcst_lead[dpt],fcst_valid[dpt],fcst_valid[dpt],
                obs_lead[dpt],obs_valid[dpt],obs_valid[dpt],fcst_var[dpt],fcst_unit[dpt],fcst_lev[dpt],
                obs_var[dpt],obs_unit[dpt],obs_lev[dpt],'ADPUPA',maskname,'NEAREST','1','NA','NA','NA','NA','MPR',
                str(dlength),str(index_num[dpt]),'NA',lats_in[dpt],lons_in[dpt],obsslev[dpt],'NA',data_fcst[dpt],
                data_obs[dpt],'NA','NA','NA','NA'))

