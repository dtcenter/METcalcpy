import pytest
import numpy as np
import pandas as pd

import metcalcpy.util.mode_2d_ratio_statistics as m2rs
import metcalcpy.util.mode_2d_arearat_statistics as m2as

def prepare_data(obj_type = "2d"):
    """
    Prepare some data for testing mode ara rations.
    This uses an existing file in test/data. A more 
    robust approach would be to use a real MODE export
    from METviewer.
    """
    file_path = "test/data/ee_av_input.data"
    df = pd.read_csv(file_path, sep="\t")

    df["object_type"] = obj_type

    cols = np.array(df.columns)
    data = df.to_numpy()
    return cols, data


# Expected values
# I suspect these could be simplified
# as ratios of each other.
FSA_ASA = 0.5858572
ASM_ASA = 0.8955360
FSM_FSA = 0.9069445
OSM_OSA = 0.8793973
FSM_ASM = 0.0930555
OSM_ASM = 0.4066794
OSU_ASU = 0.4781241
OSA_AAA = 0.2184832
FSA_AAA = 0.3090720
FSA_FAA = 0.5243991
OSA_OAA = 0.5320855
OCA_ACA = 0.4066794

OBJ_HITS = 362487
OBJ_OSU = 20217
OBJ_FSU = 22067

@pytest.mark.parametrize(
        "pair, expected",
        [
            ("fsa_asa", FSA_ASA),
            ("osa_asa", 1 - FSA_ASA),
            ("asm_asa", ASM_ASA),
            ("asu_asa", 1 - ASM_ASA),
            ("fsm_fsa", FSM_FSA),
            ("fsu_fsa", 1- FSM_FSA),
            ("osm_osa", OSM_OSA),
            ("osu_osa", 1 - OSM_OSA),
            ("fsm_asm", FSM_ASM),
            ("osm_asm", OSM_ASM),
            ("osa_aaa", OSA_AAA),
            ("osu_asu", OSU_ASU),
            ("fsa_aaa", FSA_AAA),
            ("fsa_faa", FSA_FAA),
            ("fca_faa", 1 - FSA_FAA),
            ("osa_oaa", OSA_OAA),
            ("oca_oaa", 1 - OSA_OAA),
            ("fca_aca", 1 - OSM_ASM),
            ("oca_aca", OCA_ACA),
            ("fsa_osa", FSA_AAA / OSA_AAA),
            ("osa_fsa", 1), # Can this be right?
            ("aca_asa", ASM_ASA),
            ("asa_aca", 1/ASM_ASA),
            ("fca_fsa", (1-FSA_FAA)/FSA_FAA),
            ("fsa_fca", FSA_FAA/(1-FSA_FAA)),
            ("oca_osa", OSM_OSA),
            ("osa_oca", 1 / OSM_OSA),
            ("objahits", OBJ_HITS / 2),
            ("objamisses", OBJ_OSU),
            ("objafas", OBJ_FSU),
            # The values below should be derived from
            # the FSU, OSU, and HITS. However, the
            # formulas described in the doc strings
            # do not appear to give these answers.
            ("objacsi", 0.8108331),
            ("objapody", 0.8996478),
            ("objafar", 0.0295392),
        ]
)
def test_m2as(pair, expected):
    col_names, data = prepare_data()
    if pair.startswith("obj"):
        func_str = "calculate_2d_{}".format(pair)
    else:
         func_str = "calculate_2d_arearat_{}".format(pair)

    func = getattr(m2as, func_str)
    actual = func(data, col_names)
    np.testing.assert_almost_equal(actual, expected)


# Expected values for rations
rFSA_ASA = 0.4399575
rASM_ASA = 0.5855473
rFSM_FSA = 0.9541063
rFSU_FSA = 0.5458937
rOSM_OSA = 0.6166983
rFSM_ASM = 0.3411978
rOSM_ASM = 0.3666062
rFSU_ASU = 0.4820513
rFSA_AAA = 0.3236904
rOSA_AAA = 0.4120407
rFSA_FAA = 0.7101201
rOSA_OAA = 0.7571839
rFCA_ACA = 0.5
rACA_ASA = 0.3591923

@pytest.mark.parametrize(
        "pair, expected",
        [
            ("fsa_asa", rFSA_ASA),
            ("osa_asa", 1 - rFSA_ASA),
            ("asm_asa", rASM_ASA),
            ("asu_asa", 1 - rASM_ASA),
            ("fsm_fsa", rFSM_FSA),
            ("fsu_fsa", rFSU_FSA),
            ("osm_osa", rOSM_OSA),
            ("osu_osa", 1 - rOSM_OSA),
            ("fsm_asm", rFSM_ASM),
            ("osm_asm", rOSM_ASM),
            ("fsu_asu", rFSU_ASU),
            ("osu_asu", 1 - rFSU_ASU),
            ("fsa_aaa", rFSA_AAA),
            ("osa_aaa", rOSA_AAA),
            ("fsa_faa", rFSA_FAA),
            ("fca_faa", 1- rFSA_FAA),
            ("osa_oaa", rOSA_OAA),
            ("oca_oaa", rOSA_OAA),
            ("fca_aca", rFCA_ACA),
            ("oca_aca", 1 - rFCA_ACA),
            ("fsa_osa", rFSA_AAA/rOSA_AAA),
            ("osa_fsa", rOSA_AAA/rFSA_AAA),
            ("aca_asa", rACA_ASA),
            ("asa_aca", 1 / rACA_ASA),
            ("fca_fsa", (1- rFSA_FAA)/rFSA_FAA),
            ("fsa_fca", rFSA_FAA / (1- rFSA_FAA)),
            ("oca_osa", (1 - rOSA_OAA) / rOSA_OAA),
            ("osa_oca", rOSA_OAA / (1 - rOSA_OAA)),
        ]
)
def test_m2rs(pair, expected):
    col_names, data = prepare_data()
    if pair.startswith("obj"):
        func_str = "calculate_2d_{}".format(pair)
    else:
         func_str = "calculate_2d_ratio_{}".format(pair)

    func = getattr(m2rs, func_str)
    actual = func(data, col_names)
    np.testing.assert_almost_equal(actual, expected, 6)
