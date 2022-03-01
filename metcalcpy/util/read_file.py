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

class ReadMETOutput:

    def __init__(self):
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

    def readYAMLConfig(self,configFile):
        """ Returns a file or list of files

        Args:
            configFile: A YAML formatted config file

        Returns: 
            returns a list containing a single or multiple file names including path
        """
        # Retrieve the contents of a YAML custom config file to over-ride
        # or augment settings defined by the default config file.
        #Use a configure file parser that handles environment variables
        files_dict = parse_config(configFile)

        #parse_config returns a dictionary, read_data_files wants a list
        files = files_dict['files']
        return files

    def readXMLConfig(self,configFile):
        """ Returns a file or list of files
            Args:
                configFile: XML formatted config file
            Returns: 
                returns a list containg a single or multiple file names including path
        """

        # Retrieve the contents of an XML  custom config file to over-ride
        # or augment settings defined by the default config file.
        # Uses XmlLoadFile from METdatadb

        XML_LOADFILE = XmlLoadFile(configFile)
        XML_LOADFILE.read_xml()

        return XML_LOADFILE.load_files

    def readData(self,files_from_config):
        """ 
            Args:
                files_from_config: A list of MET ouptut files grabbed from a config file
            Returns: 
                a pandas DataFrame containing MET output contents
        """

        file_data = read_data_files.ReadDataFiles()

        # read in the data files, with options specified by XML flags
        #set load_flags and line_types empty so that everything is read
        line_types = []
        file_data.read_data(self.flags,
                            files_from_config,
                            line_types)


        df = file_data.stat_data
        return df

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

