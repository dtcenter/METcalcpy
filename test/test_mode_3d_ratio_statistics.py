import pytest
from unittest.mock import patch
import numpy as np
import metcalcpy.util.mode_3d_ratio_statistics as m3rs

column_names = np.array(
    ["object_type", "volume", "fcst_flag", "simple_flag", "matched_flag"]
)

data = np.array(
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

@pytest.mark.parametrize(
    "func_name,data_values,expected",
    [
        ("ratio_fsa_asa", data, 0.3333333),
        ("ratio_osa_asa", data, 0.6666667),
        ("ratio_asm_asa", data, 0.6666667),
        ("ratio_asu_asa", data, 0.3333333),
        ("ratio_fsm_fsa", data, 1.0),
        ("ratio_fsu_fsa", data, 0.5),
        ("ratio_osm_osa", data, 0.75),
        ("ratio_osu_osa", data, 0.25),
        ("ratio_fsm_asm", data, 0.25),
        ("ratio_osm_asm", data, 0.25),
        ("ratio_fsu_asu", data, 0.5),
        ("ratio_osu_asu", data, 0.5),
        ("ratio_fsa_aaa", data, 0.2857143),
        ("ratio_osa_aaa", data, 0.5714286),
        ("ratio_fsa_faa", data, 0.6666667),
        ("ratio_fca_faa", data, 0.3333333),
        ("ratio_osa_oaa", data, 1.0),
        ("ratio_oca_oaa", data, 0.0),
        ("ratio_fca_aca", data, 1.0),
        ("ratio_fsa_osa", data, 0.5),
        ("ratio_osa_fsa", data, 2.0),
        ("ratio_asa_aca", data, 6.0),
        ("ratio_fca_fsa", data, 0.5),
        ("ratio_fsa_fca", data, 2.0),
        ("ratio_oca_osa", data, 0.0),
        ("ratio_osa_oca", data, None),
        ("objhits", data, 2.0),
        ("objmisses", data, 1.0),
        ("objfas", data, 1.0),
        ("objcsi", data, 0.5),
        ("objpody", data, 0.6666667),
        ("objfar", data, 0.3333333),
    ],
)
def test_calculate_3d_ratio(func_name, data_values, expected):
    func_str = f"calculate_3d_{func_name}"
    func = getattr(m3rs, func_str)
    actual = func(data_values, column_names)
    assert actual == expected

    # Check None is returned on Exception
    with patch.object(m3rs, "column_data_by_name_value", side_effect=TypeError):
        actual = func(data_values, column_names)
    assert actual == None
