import pytest
from unittest.mock import patch
import numpy as np
import metcalcpy.util.mode_2d_arearat_statistics as m2as

column_names = np.array(
    ["object_type", "area", "fcst_flag", "simple_flag", "matched_flag"]
)

data = np.array(
    [
        ["2d", 100, 1, 1, 1],
        ["2d", 30, 0, 1, 1],
        ["2d", 120, 1, 1, 0],
        ["2d", 12, 0, 1, 1],
        ["2d", 1, 1, 0, 1],
        ["2d", 17, 0, 1, 1],
        ["2d", 200, 1, 1, 1],
        ["3d", 66, 0, 1, 0],
    ]
)

@pytest.mark.parametrize(
    "func_name,data_values,expected",
    [
        ("arearat_fsa_asa", data, 0.8768267),
        ("arearat_osa_asa", data, 0.1231733),
        ("arearat_asm_asa", data, 0.7494781),
        ("arearat_asu_asa", data, 0.2505219),
        ("arearat_fsm_fsa", data, 0.7142857),
        ("arearat_fsu_fsa", data, 0.2857143),
        ("arearat_osm_osa", data, 1.0),
        ("arearat_osu_osa", data, None),
        ("arearat_fsm_asm", data, 0.2857143),
        ("arearat_osm_asm", data, 0.1643454),
        ("arearat_osu_asu", data, None),
        ("arearat_fsa_aaa", data, 0.875),
        ("arearat_osa_aaa", data, 0.1229167),
        ("arearat_fsa_faa", data, 0.9976247),
        ("arearat_fca_faa", data, 0.0023753),
        ("arearat_osa_oaa", data, 1.0),
        ("arearat_oca_oaa", data, None),
        ("arearat_fca_aca", data, 1.0),
        ("arearat_oca_aca", data, None),
        ("arearat_fsa_osa", data, 7.1186441),
        ("arearat_osa_fsa", data, 1.0),
        ("arearat_aca_asa", data, 0.0020877),
        ("arearat_asa_aca", data, 479.0),
        ("arearat_fca_fsa", data, 0.002381),
        ("arearat_fsa_fca", data, 420.0),
        ("arearat_oca_osa", data, None),
        ("arearat_osa_oca", data, None),
        ("objahits", data, 179.5),
        ("objamisses", data, None),
        ("objafas", data, 120.0),
        ("objacsi", data, 0.5993322),
        ("objapody", data, None),
        ("objafar", data, 0.1431981),
            ],
)
def test_calculate_3d_ratio(func_name, data_values, expected):
    func_str = f"calculate_2d_{func_name}"
    func = getattr(m2as, func_str)
    actual = func(data_values, column_names)
    assert actual == expected

    # Check None is returned on Exception
    with patch.object(m2as, "column_data_by_name_value", side_effect=TypeError):
        actual = func(data_values, column_names)
    assert actual == None
