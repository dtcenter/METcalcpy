from pathlib import Path
import numpy as np

def abs_path(rel_path):
    """Turn a relative path into abs path"""
    return str(Path(str(Path(__file__).parents[2])) / rel_path)

# Datafile access
AGG_STAT_AND_BOOT_DATA = abs_path("METcalcpy/test/data/agg_stat_and_boot_data.data")

MODE_TEST_DATA = np.array(
    [
        ["3d", 100, 1, 1, 1],
        ["3d", 30, 0, 1, 1],
        ["3d", 120, 1, 1, 0],
        ["3d", 12, 0, 1, 1],
        ["3d", 1, 1, 0, 1],
        ["3d", 17, 0, 1, 1],
        ["2d", 200, 1, 1, 1],
        ["3d", 66, 0, 1, 0],
    ]
)
