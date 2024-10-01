****************
Logging In Guide
****************


This guide provides a comprehensive overview of the newly integrated logging capabilities 
within METcalcpy. These enhancements are designed to provide users with valuable insights 
into the application's execution, aiding in tasks such as debugging, performance monitoring, 
and understanding the operational flow of the program.


What's New
==========

Centralized Logging Configuration (**logging_config.py**):
----------------------------------------------------------

A new script, **logging_config.py**, has been introduced to centralize the management of logging 
configurations. This approach ensures consistency and simplifies the maintenance of logging 
settings across all modules within METcalcpy.


* Key Feature: :code:`setup_logging` Function

  * The :code:`setup_logging` function is the core of **logging_config.py**. It initializes 
    and configures the logger instance based on parameters specified in a YAML configuration 
    file. This function reads logging settings such as :code:`log_dir`, 
    :code:`log_filename`, and :code:`log_level` from the YAML file and sets 
    up Python's logging module accordingly.
  * By isolating the logging configuration in this script, it becomes easier to 
    manage and update logging behavior without altering the core logic of other modules.

Example Integration in **agg_stat.py**:

.. code-block:: py

  from metcalcpy.logging_config import setup_logging
  
  class AggStat:
      def __init__(self, in_params):
          self.logger = setup_logging(in_params)
          # Other initialization code...

In this example, when an :code:`AggStat object` is instantiated, it invokes the 
:code:`setup_logging` function, passing in the :code:`in_params` dictionary, 
which contains logging configurations from a YAML file such as 
**val1l2_agg_stat.yaml**. This ensures the logger is configured according to 
the user's settings.











