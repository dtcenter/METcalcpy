# ============================*
 # ** Copyright UCAR (c) 2020
 # ** University Corporation for Atmospheric Research (UCAR)
 # ** National Center for Atmospheric Research (NCAR)
 # ** Research Applications Lab (RAL)
 # ** P.O.Box 3000, Boulder, Colorado, 80307-3000, USA
 # ============================*
 
 
 
"""
Program Name: read_file.py
Copyright 2021 UCAR/NCAR/RAL
"""

__author__ = 'Hank Fisher'
__version__ = '1.0.0'


import numpy as np
import pandas as pd
import yaml
from read_env_vars_in_config import parse_config
#Two PYTHONPATH directories need to be set for the imports to work
#The parent directory for where METdatadb is installed and
#The METdbload/METdatadb/ush directory for constants in METdbload
from METdatadb import METdbLoad as dbload
from METdatadb.METdbLoad.ush import read_data_files
from METdatadb.METdbLoad.ush.read_load_xml import XmlLoadFile
from metcalcpy.util.safe_log import safe_log

class ReadMETOutput:

    def __init__(self, logger=None):
        """ Creates a output reader with a list of files to read
        """
        self.flags = {}
        self.flags['line_type_load'] = False
        self.flags['load_stat'] = True
        self.flags['load_mode'] = True
        self.flags['load_mtd'] = True
        self.flags['load_mpr'] = False
        self.flags['load_orank'] = False
        self.flags['force_dup_file'] = False
        self.flags['verbose'] = False
        self.flags['stat_header_db_check'] = True
        self.flags['tcst_header_db_check'] = True
        self.flags['mode_header_db_check'] = True
        self.flags['mtd_header_db_check'] = True
        self.flags['drop_indexes'] = False
        self.flags['apply_indexes'] = False
        self.flags['load_xml'] = True
        self.logger = logger

    def readYAMLConfig(self,configFile):
        """ Returns a file or list of files

        Args:
            configFile: A YAML formatted config file

        Returns: 
            returns a list containing a single or multiple file names including path
        """
        logger = self.logger
        try:
            # Retrieve the contents of a YAML custom config file to override
            # or augment settings defined by the default config file.
            # Use a config file parser that handles environment variables.
            files_dict = parse_config(configFile, logger=logger)

            if files_dict is None:
                safe_log(logger, "error", "Failed to parse the YAML configuration. 'files_dict' is None.")
                return []

            # parse_config returns a dictionary, read_data_files expects a list
            files = files_dict.get('files', [])

            if not files:
                safe_log(logger, "warning", "No 'files' entry found in the YAML configuration.")
            else:
                safe_log(logger, "debug", f"Files retrieved from YAML configuration: {files}")

            return files

        except Exception as e:
            safe_log(logger, "error", f"An error occurred while reading the YAML configuration: {str(e)}")
            return []

    def readXMLConfig(self,configFile):
        """ Returns a file or list of files
            Args:
                configFile: XML formatted config file
            Returns: 
                returns a list containg a single or multiple file names including path
        """
        logger = self.logger
        safe_log(logger, "debug", f"Attempting to read XML configuration from file: {configFile}")

        try:
            # Retrieve the contents of an XML custom config file to override
            # or augment settings defined by the default config file.
            # Uses XmlLoadFile from METdatadb.
            XML_LOADFILE = XmlLoadFile(configFile)
            
            safe_log(logger, "debug", "Reading XML file.")
            XML_LOADFILE.read_xml()
            
            files = XML_LOADFILE.load_files()
            safe_log(logger, "debug", f"Files retrieved from XML configuration: {files}")

            return files

        except Exception as e:
            safe_log(logger, "error", f"An error occurred while reading the XML configuration: {str(e)}")
            return []

    def readData(self,files_from_config):
        """ 
            Args:
                files_from_config: A list of MET ouptut files grabbed from a config file
            Returns: 
                a pandas DataFrame containing MET output contents
        """
        logger = self.logger
        safe_log(logger, "debug", f"Starting to read data from the files: {files_from_config}")

        try:
            # Initialize the ReadDataFiles class instance
            file_data = read_data_files.ReadDataFiles()

            # Read in the data files, with options specified by XML flags
            # Set load_flags and line_types empty so that everything is read
            line_types = []
            safe_log(logger, "debug", f"Reading data with flags: {self.flags} and line_types: {line_types}")
            
            file_data.read_data(self.flags, files_from_config, line_types)

            # Retrieve the data as a pandas DataFrame
            df = file_data.stat_data
            safe_log(logger, "debug", f"Data reading completed. DataFrame shape: {df.shape}")

            return df

        except Exception as e:
            safe_log(logger, "error", f"An error occurred while reading data: {str(e)}")
            return pd.DataFrame()  # Return an empty DataFrame in case of error

def main():
    """
    Reads in a default config file that loads sample MET output data and puts it into a pandas
    DataFrame
    """

    file_reader = ReadMETOutput()

    #Get the files to be loaded either from an XML file
    #Or a YAML file
    #xml_config_file = '../../examples/read_files.xml'
    #load_files = file_reader.readXMLConfig(xml_config_file)

    #The advantage of using YAML is that you can use environment variables to
    #reference a path to the file
    yaml_config_file = "../../examples/read_files.yaml"
    load_files = file_reader.readYAMLConfig(yaml_config_file)

    df = file_reader.readData(load_files)
    print(df)


if __name__ == "__main__":
    main()

