"""
Program Name: read_file.py
Copyright 2021 UCAR/NCAR/RAL
"""

__author__ = 'Hank Fisher'
__version__ = '1.0.0'
__email__ = 'met_help@ucar.edu'


import numpy as np
import pandas as pd
import yaml
#Two PYTHONPATH directories need to be set for the imports to work
#The parent directory for where METdatadb is installed and
#The METdbload/METdatadb/ush directory for constants in METdbload
from METdatadb import METdbLoad as dbload
from METdatadb.METdbLoad.ush import read_data_files

class ReadMETOutput:

    def __init__(self):
        """ Creates a output reader with a list of files to read
        """

    def readConfigFile(self,configFile):
        """ Returns a file or list of files
        """

        # Retrieve the contents of the custom config file to over-ride
        # or augment settings defined by the default config file.
        with open(configFile, 'r') as stream:
            try:
               files = yaml.load(stream, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                print(exc)

        print(files)

    def readData(self):
        """ Returns a pandas DataFrame containing MET output contents
        """

        # instantiate a read data files object
        #file_data = dbload.read_data_files.ReadDataFiles
        file_data = read_data_files.ReadDataFiles()

        # read in the data files, with options specified by XML flags
        #set load_flags and line_types as empty so that everything is read
        load_flags = []
        line_types = []
        file_data.read_data(load_flags,
                            file_data,
                            line_types)


        df = file_data.stat_data
        return df

def main():
    """
    Reads in a default config file that loads sample MET output data and puts it into a pandas
    DataFrame
    """
    configFile = "./read_files.yaml"
    file_reader = ReadMETOutput()
    file_reader.readConfigFile(configFile)
    df = file_reader.readData()


if __name__ == "__main__":
    main()

