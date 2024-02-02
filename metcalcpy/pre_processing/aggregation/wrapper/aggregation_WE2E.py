import os
import yaml
import shutil
import subprocess
import time

def read_yaml_file(yaml_file_path):

    try:
        with open(yaml_file_path, 'r') as f:
            config = yaml.safe_load(f)
            return config
    except Exception as e:
        print(f"Error reading YAML file: {e}")
        return None


def check_variables_and_files(config):

    # Check if variables are set
    required_vars = ["METCALCPY", "METPLOTPY", "AGG_STAT_YAML", "AGG_WFLOW"]
    for var in required_vars:
        if var not in config or not config[var]:
            print(f"One or more variables are not set. Missing: {var}")
            return 1
    
    # Check if paths exist
    for var, path in config.items():
        if var in required_vars and not os.path.exists(path):
            print(f"{var} points to a path that does not exist: {path}")
            return 1
    
    print("Environment variables - OK\n")
    return 0

def copy_file(ifrom,to):

    ifile = ifrom.split('/')[-1]

    if not os.path.isfile(to):
        shutil.copy(ifrom, to)
    else:
        print(f"WARNING: File '{ifile}' already exists in the work directory.")

def run_workflow(config):
    # Create the working directory if it doesn't exist
    workdir = config.get('WORKDIR', './workdir')
    os.makedirs(workdir, exist_ok=True)

    
    agg_stat = config['AGG_STAT']    
    agg_prep = config['AGG_PREP']    
    path_to_agg_prep = os.path.join(config['AGG_WFLOW'], 'src', 'aggregation_preprocessor.py')
    path_to_yaml_prep = os.path.join(config['AGG_WFLOW'], 'src', 'yaml_preprocessor.py')
    path_to_config_agg_prep = os.path.join(config['AGG_WFLOW'], 'config', 'config_aggregation_preprocessor.yaml')
    path_to_agg_stat = os.path.join(config['METCALCPY'], 'metcalcpy', 'agg_stat.py')
    path_to_config_agg_stat = config['AGG_STAT_YAML']
    path_to_config_plot_cmn = os.path.join(config['AGG_WFLOW'], 'config', "config_plot_cmn.yaml")
    performance_diagram = config["PERFORMANCE_DIAGRAM"]
    path_to_config_custom_performance_diagram = os.path.join(config['AGG_WFLOW'], 'config', 'custom_performance_diagram.yaml')
    path_to_performance_diagram = os.path.join(config['METPLOTPY'], 'metplotpy', 'plots/performance_diagram/performance_diagram.py')
    line = config["LINE"]
    path_to_config_custom_line= os.path.join(config['AGG_WFLOW'], 'config', 'custom_line.yaml')
    path_to_line = os.path.join(config['METPLOTPY'], 'metplotpy', 'plots/line/line.py')
    taylor = config["TAYLOR"]
    path_to_config_custom_taylor_diagram = os.path.join(config['AGG_WFLOW'], 'config', 'custom_taylor_diagram.yaml')
    path_to_taylor_diagram = os.path.join(config['METPLOTPY'], 'metplotpy', 'plots/taylor_diagram/taylor_diagram.py')

    # Copy files
    try:
        shutil.copy(path_to_agg_prep, f'{workdir}/.')
        shutil.copy(path_to_agg_stat, f'{workdir}/.')

        copy_file(path_to_config_agg_prep, f'{workdir}/config_aggregation_preprocessor.yaml')
        copy_file(path_to_config_agg_stat, f'{workdir}/config_agg_stat.yaml')
        
        if performance_diagram or line or taylor:
            shutil.copy(path_to_yaml_prep, f'{workdir}/.')
            copy_file(path_to_config_plot_cmn, f'{workdir}/config_plot_cmn.yaml')
 
        if performance_diagram:
            copy_file(path_to_config_custom_performance_diagram, f'{workdir}/custom_performance_diagram.yaml')
            shutil.copy(path_to_performance_diagram, f'{workdir}/.')

        if line:
            copy_file(path_to_config_custom_line, f'{workdir}/custom_line.yaml')
            shutil.copy(path_to_line, f'{workdir}/.')
    
        if taylor:
            copy_file(path_to_config_custom_taylor_diagram, f'{workdir}/custom_taylor_diagram.yaml')
            shutil.copy(path_to_taylor_diagram, f'{workdir}/.')

        os.chdir(workdir)
    except KeyError as e:
        return f"Missing configuration key: {e}"
    except Exception as e:
        return f"An error occurred while copying files: {e}"
    

    nsteps = 2
    if performance_diagram or line or taylor:
         nsteps = 3
    
    # Run Python scripts
    try:
        print(f"Step 1/{nsteps} - aggregation_preprocessor...")
        if agg_prep:
            start = time.time()
            subprocess.run(["python", "aggregation_preprocessor.py", "--yaml", "config_aggregation_preprocessor.yaml"], check=True)
            elapsed = time.time() - start
            print(f"Done. Time Elapsed = {elapsed:.1f} seconds.\n")
        else:
            print("  Step skipped...")    

        print(f"Step 2/{nsteps} - agg_stat...")
        if agg_stat:
            start = time.time()
            subprocess.run(["python", "agg_stat.py", "config_agg_stat.yaml"], check=True)
            elapsed = time.time() - start
            print(f"Done. Time Elapsed = {elapsed:.1f} seconds.\n")
        else:
            print("  Step skipped...")

        if nsteps == 3:
            print(f"Step 3/3 - plot...")
            start = time.time()
            if performance_diagram:
                subprocess.run(["python", "yaml_preprocessor.py", "config_plot_cmn.yaml", "custom_performance_diagram.yaml", "-o" , "config_performance_diagram.yaml"], check=True)
                subprocess.run(["python", "performance_diagram.py", "config_performance_diagram.yaml"], check=True)

            if line:
                subprocess.run(["python", "yaml_preprocessor.py", "config_plot_cmn.yaml", "custom_line.yaml", "-o" , "config_line.yaml"], check=True)
                subprocess.run(["python", "line.py", "config_line.yaml"], check=True)

            if taylor:
                subprocess.run(["python", "yaml_preprocessor.py", "config_plot_cmn.yaml", "custom_taylor_diagram.yaml", "-o" , "config_roc_diagram.yaml"], check=True)
                subprocess.run(["python", "taylor_diagram.py", "custom_taylor_diagram.yaml"], check=True)

            elapsed = time.time() - start
            print(f"Done. Time Elapsed = {elapsed:.1f} seconds.\n")   
    except subprocess.CalledProcessError as e:
        return f"An error occurred while running the Python scripts: {e}"

    return "Workflow completed successfully."


def main():

    yaml_file = "./environment.yaml"

    config = read_yaml_file(yaml_file)

    if config is None:
        quit()
    
    print("")
    if check_variables_and_files(config) != 0:
        quit()
    
    print(run_workflow(config))
    

if __name__ == "__main__":
    main()