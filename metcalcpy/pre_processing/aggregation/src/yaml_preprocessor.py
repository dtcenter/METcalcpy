import argparse
import yaml

def read_yaml_file(file_path):
    """Reads a YAML file and returns its contents as a dictionary."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def write_yaml_file(data, file_path):
    """Writes a dictionary to a YAML file."""
    with open(file_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

def combine_configs(base_config_path, override_config_path):
    """Combines two configurations, giving preference to the override config."""
    base_config = read_yaml_file(base_config_path)
    override_config = read_yaml_file(override_config_path)
    combined = base_config.copy()  # Start with the base config
    combined.update(override_config)   # Update with override config, overwriting base settings
    return combined

def main():
    parser = argparse.ArgumentParser(description="Combine two YAML configuration files.")
    parser.add_argument("base_config", help="The path to the base YAML configuration file.")
    parser.add_argument("override_config", help="The path to the YAML configuration file that will override the base config.")
    parser.add_argument("-o", "--output", default="combined_config.yaml",
                        help="Output path for the combined YAML configuration file.")
    
    args = parser.parse_args()
    
    # Combine both configurations using provided arguments
    combined_config = combine_configs(args.base_config, args.override_config)
    
    # Write the combined configuration to the specified output file
    write_yaml_file(combined_config, args.output)
    print(f"Combined configuration written to {args.output}")

if __name__ == "__main__":
    main()
