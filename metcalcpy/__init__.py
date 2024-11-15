# ============================*
 # ** Copyright UCAR (c) 2020
 # ** University Corporation for Atmospheric Research (UCAR)
 # ** National Center for Atmospheric Research (NCAR)
 # ** Research Applications Lab (RAL)
 # ** P.O.Box 3000, Boulder, Colorado, 80307-3000, USA
 # ============================*

# set value for metcalcpy.__version__ 
import importlib.metadata
__version__ = importlib.metadata.version("metcalcpy")
 
"""This module contains a variety of statistical calculations."""
GROUP_SEPARATOR = ':'
DATE_TIME_REGEX = r'\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}'
