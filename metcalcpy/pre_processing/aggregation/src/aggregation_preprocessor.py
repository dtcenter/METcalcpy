import xml.etree.ElementTree as ET
import yaml
import argparse
import os 
import yaml
import subprocess
import pandas as pd
import numpy as np
import time


def write_stat_yaml(xml_spec_file, output_yaml, output_reformatted_file, output_dir='.'):
    # Create a dictionary containing the data to be written to the YAML file
    data = {
        'output_dir': output_dir,
        'output_filename': output_reformatted_file,
        'xml_spec_file': xml_spec_file
    }
    
    # Write the dictionary to a YAML file
    with open(output_yaml, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

def read_config_stat_yaml(file_path):
    with open(file_path, 'r') as file:
        config_data = yaml.safe_load(file)
        
    return config_data

def create_xml(path,output_file,
               val_names=["ensemble_stat","grid_stat","mode","point_stat","stat_analysis","wavelet_stat"]):

    # Create root element
    root = ET.Element("load_spec")
    
    # Create folder_tmpl element and add it to the root
    folder_tmpl = ET.SubElement(root, "folder_tmpl")
    folder_tmpl.text = path
    
    # Create verbose element and add it to the root
    verbose = ET.SubElement(root, "verbose")
    verbose.text = "true"
    
    # Create load_val element and add it to the root
    load_val = ET.SubElement(root, "load_val")
    
    # Create field element and add it to load_val
    field = ET.SubElement(load_val, "field", name="met_tool")
    
    # Create val elements and add them to field
    for val_name in val_names:
        val = ET.SubElement(field, "val")
        val.text = val_name
    
    # Create description element and add it to the root
    description = ET.SubElement(root, "description")
    description.text = "MET output"
    
    # Create a tree structure and write the XML to a file
    tree = ET.ElementTree(root)
    with open(output_file, "wb") as fh:
        tree.write(fh)

def check_var(config_data,varname):
    try:
        var = config_data.get(varname)
        if var is None:
            var = ""
    except:
        var = ""

    return var

def pre_req(stat_name: list) -> list:
    STATISTIC_TO_FIELDS1 = {
        'baser': ['fy_oy', 'fn_oy'],
        'acc': ['fy_oy', 'fn_on'],
        'fbias': ['fy_oy', 'fn_on', 'fn_oy', 'fy_on'],
        'fmean': ['fy_oy', 'fy_on'],
        'pody': ['fy_oy', 'fn_oy'],
        'pofd': ['fy_on', 'fn_on'],
        'podn': ['fn_on', 'fy_on'],
        'far': ['fy_on', 'fy_oy'],
        'csi': ['fy_oy', 'fy_on', 'fn_oy'],
        'gss': ['fy_oy', 'fy_on', 'fn_oy'],
        'hk': ['fy_oy', 'fn_oy', 'fy_on', 'fn_on'],
        'hss': ['fy_oy', 'fn_oy', 'fy_on', 'fn_on'],
        'odds': ['fy_oy', 'fn_oy', 'fy_on', 'fn_on'],
        'lodds': ['fy_oy', 'fn_oy', 'fy_on', 'fn_on'],
        'baggs': ['fy_oy', 'fn_oy', 'fy_on'],
        'eclv': ['fy_oy', 'fn_oy', 'fy_on', 'fn_on']
    }

    STATISTIC_TO_FIELDS2 = {
        'fbar': ['fbar'],
        'obar': ['obar'],
        'fstdev': ['fbar', 'ffbar'],
        'ostdev': ['obar', 'oobar'],
        'fobar': ['fobar'],
        'ffbar': ['ffbar'],
        'oobar': ['oobar'],
        'mae': ['mae'],
        'mbias': ['obar', 'fbar'],
        'corr': ['ffbar', 'fbar', 'oobar', 'obar', 'fobar'],
        'anom_corr': ['ffabar', 'fabar', 'ooabar', 'oabar', 'foabar'],
        'anom_corr_raw': ['ffabar', 'ooabar', 'foabar'],
        'rmsfa': ['ffabar'],
        'rmsoa': ['ooabar'],
        'me': ['fbar', 'obar'],
        'me2': ['fbar', 'obar'],
        'mse': ['ffbar', 'oobar', 'fobar'],
        'msess': ['ffbar', 'oobar', 'fobar', 'obar'],
        'rmse': ['ffbar', 'oobar', 'fobar'],
        'si': ['ffbar', 'oobar', 'fobar', 'obar'],
        'estdev': ['ffbar', 'oobar', 'fobar', 'fbar', 'obar'],
        'bcmse': ['ffbar', 'oobar', 'fobar', 'fbar', 'obar'],
        'bcrmse': ['ffbar', 'oobar', 'fobar', 'fbar', 'obar'],
        'pr_corr': ['ffbar', 'oobar', 'fobar', 'fbar', 'obar']
    }

    if stat_name in STATISTIC_TO_FIELDS1:
        var = STATISTIC_TO_FIELDS1[stat_name]
        fields = 1
    elif stat_name in STATISTIC_TO_FIELDS2:
        var = STATISTIC_TO_FIELDS2[stat_name]
        fields = 2
    else:
        raise Exception(f"Statistic named '{stat_name}' does not exist in STATISTIC_FIELDS.")

    return var, fields

def main():

    # Initialize argparse
    parser = argparse.ArgumentParser(description="Read agg_stat.yaml file.")
    
    # Add argument for file path
    parser.add_argument("-y", "--yaml", type=str, required=True, help="Path to the agg_stat.yaml file.")    
    parser.add_argument("-f", "--force", action='store_true', help="Force reformat output overwrite.")
   
    # Parse the arguments
    args = parser.parse_args()
    
    # Read agg_stat.yaml file using the provided path
    config_data = read_config_stat_yaml(args.yaml)

    # Access individual elements from the config_data dictionary
    prefix = check_var(config_data,'prefix')
    suffix = check_var(config_data,'suffix')
    dates = check_var(config_data,'dates')
    members = check_var(config_data,'members')
    group_members = config_data.get('group_members')
    group_name = config_data.get('group_name')
    output_xml_file = config_data.get('output_xml_file')
    output_yaml_file = config_data.get('output_yaml_file')
    output_reformatted_file = config_data.get('output_reformatted_file')
    metdataio_dir = config_data.get('metdataio_dir')
    fcst_var = config_data.get('fcst_var')
    fcst_thresh = config_data.get('fcst_thresh')
    list_stat = config_data.get('list_stat')
    output_aggregate_file = config_data.get('output_aggregate_file')
    log_file = config_data.get('log_file')

    stat_script = metdataio_dir + "/METreformat/write_stat_ascii.py"
    if not os.path.isfile(stat_script):
        raise Exception("METdataio script not found!")

    if os.path.isfile(log_file):
        os.system(f"rm -rf {log_file}")
    
    print("  Running reformatter...")
    start = time.time()
    if os.path.isfile(output_reformatted_file):        
        while True:
        
            if args.force:
                os.remove(output_reformatted_file)
                print(f"  - File '{output_reformatted_file}' was removed.")
                break

            user_input = input(f"  The file '{output_reformatted_file}' already exists. Do you want to delete it? (yes/no): ")
            if user_input.lower() == 'yes':
                print(f"  - File '{output_reformatted_file}' was removed.")
                os.remove(output_reformatted_file)
                break 
            elif user_input.lower() == 'no':
                break 
            else:
                print(f"'{user_input}' is not a valid option!")
    
    dirs = []
    for i in dates:
        for j in members:
            idir = prefix + i + "/" + j + suffix
            if os.path.isdir(idir):
                dirs.append(idir)
    
    
    
    if len(dirs) == 0:
        raise Exception("No directories found!")

    start = time.time()
    count = 0
    for i in dirs:
        create_xml(i,output_xml_file)
        write_stat_yaml(output_xml_file,output_yaml_file,output_reformatted_file)
        try:
            with open(log_file, 'a') as f:
                subprocess.run(["python", stat_script, output_yaml_file], stdout=f, stderr=f, check=True)
        except subprocess.CalledProcessError as e:
            raise Exception(f"An error occurred while executing the command: {e}")

        count += 1                 

    try:
        subprocess.run(["rm", "-f", output_yaml_file,output_xml_file], check=True)
    except subprocess.CalledProcessError as e:
        raise Exception(f"An error occurred while executing the command: {e}") 

    elapsed = time.time() - start

    print(f"  Done. Time Elapsed = {elapsed:.1f} seconds.\n")     

    print("  Filtering data...")
    start = time.time()
    df = pd.read_csv(output_reformatted_file,sep='\t',low_memory=False)

    if group_members:
        temp_df = df.copy()
        df['model'] = group_name
        df = pd.concat([df, temp_df], ignore_index=True)


    models = df[(df['model'] != 'model')]['model'].unique()

    df = df[df['fcst_var'].isin(fcst_var)]
    df = df[['model','fcst_init_beg','fcst_valid_beg','fcst_lead',
              'fcst_thresh','obtype','vx_mask','fcst_var','stat_name',
              'stat_ncl','stat_ncu','stat_bcl','stat_bcu',
              'line_type','stat_value','total']]
    
    df = df.sort_values(by=['model','fcst_lead'])
     
    model_df = pd.DataFrame()
    for i in models:
        for m in fcst_thresh:
            for k in fcst_var:
                ctc_df = pd.DataFrame()
                for s in list_stat:
                    ctc, fields = pre_req(s.lower())
                    if fields == 2:
                        continue
                    for l in ctc:
                        temp_df = df[
                            (df['fcst_var'] == k) & 
                            (df['model'] == i) & 
                            ((df['fcst_thresh'] == m) | pd.isna(df['fcst_thresh'])) &  # This will check for equality and NaN
                            (df['stat_name'] == l.upper())
                        ]
                        
                        if len(temp_df) == 0:
                            print(f"WARNING: No data found with the following filters: {i} {m} {k} {l.upper()}")
                            continue
                        ctc_df[l.lower()] = temp_df['stat_value'].values
                    concat_df = pd.concat([temp_df.reset_index(drop=True), ctc_df.reset_index(drop=True)], axis=1)
                    concat_df['stat_name'] = s
                    concat_df['stat_value'] = np.nan

                    model_df = pd.concat([model_df,concat_df],axis=0)

    for i in models:
        for k in fcst_var:
            ctc_df = pd.DataFrame()
            for s in list_stat:
                ctc, fields = pre_req(s.lower())
                if fields == 1:
                    continue
                for l in ctc:
                    if 'ffbar' in l:
                        temp_df = df[
                            (df['fcst_var'] == k) & 
                            (df['model'] == i) & 
                            (df['stat_name'] == "FSTDEV")
                        ].copy()
                        fstdev = pd.to_numeric(temp_df['stat_value'].values)
                        temp_df = df[
                            (df['fcst_var'] == k) & 
                            (df['model'] == i) & 
                            (df['stat_name'] == "FBAR")
                        ].copy()
                        fbar = pd.to_numeric(temp_df['stat_value'].values)
                        if fstdev.size > 0 and fbar.size > 0 and fstdev.size == fbar.size:
                            ffbar = np.square(fstdev) + np.square(fbar)
                        else:
                            print("Error while calculating ffbar.")
                            continue
                        ffbar = np.square(fstdev) + np.square(fbar)
                        temp_df['stat_name'] = "FFBAR"
                        temp_df['stat_value'] = ffbar
                        l = "ffbar"
                    elif 'oobar' in l:
                        temp_df = df[
                            (df['fcst_var'] == k) & 
                            (df['model'] == i) & 
                            (df['stat_name'] == "OSTDEV")
                        ].copy()
                        ostdev = pd.to_numeric(temp_df['stat_value'].values)
                        temp_df = df[
                            (df['fcst_var'] == k) & 
                            (df['model'] == i) & 
                            (df['stat_name'] == "OBAR")
                        ].copy()
                        obar = pd.to_numeric(temp_df['stat_value'].values)
                        if ostdev.size > 0 and obar.size > 0 and ostdev.size == obar.size:
                            ffbar = np.square(fstdev) + np.square(fbar)
                        else:
                            print("Error while calculating oobar.")
                            continue
                        oobar = np.square(ostdev) + np.square(obar)
                        temp_df['stat_name'] = "OOBAR"
                        temp_df['stat_value'] = oobar
                        l = "oobar"
                    else:         
                        temp_df = df[
                            (df['fcst_var'] == k) & 
                            (df['model'] == i) & 
                            (df['stat_name'] == l.upper())
                        ]
        
                    if len(temp_df) == 0:
                        print(f"WARNING: No data found with the following filters: {i} {k} {l.upper()}")
                        continue
                    ctc_df[l.lower()] = temp_df['stat_value'].values
                concat_df = pd.concat([temp_df.reset_index(drop=True), ctc_df.reset_index(drop=True)], axis=1)
                concat_df['stat_name'] = s

                model_df = pd.concat([model_df,concat_df],axis=0)

    model_df.fillna('NA', inplace=True)
    model_df.to_csv(output_aggregate_file,sep='\t',index=False)

    elapsed = time.time() - start

    print(f"  Done. Time Elapsed = {elapsed:.1f} seconds.\n")     


if __name__ == "__main__":
    main()
