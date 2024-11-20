from pathlib import Path

def abs_path(rel_path):
    """Turn a relative path into abs path"""
    return str(Path(str(Path(__file__).parents[2])) / rel_path)

# Datafile access
AGG_STAT_AND_BOOT_DATA = abs_path("METcalcpy/test/data/agg_stat_and_boot_data.data")