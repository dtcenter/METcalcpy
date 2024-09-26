# ============================*
 # ** Copyright UCAR (c) 2020
 # ** University Corporation for Atmospheric Research (UCAR)
 # ** National Center for Atmospheric Research (NCAR)
 # ** Research Applications Lab (RAL)
 # ** P.O.Box 3000, Boulder, Colorado, 80307-3000, USA
 # ============================*
 
 def safe_log(logger, log_level, message):
    """
    Safely logs a message using the provided logger and log level.
    
    Args:
        logger (logging.Logger): The logger object. If None, the message will not be logged.
        log_level (str): The logging level to use (e.g., "info", "debug").
        message (str): The message to log.
    """
    if logger:
        log_method = getattr(logger, log_level, None)
        if callable(log_method):
            log_method(message)