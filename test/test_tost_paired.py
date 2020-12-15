"""Tests the operation of METcalcpy's tost_paired code."""
import statistics
import pingouin as pg


from metcalcpy.util.tost_paired import tost_paired


def test_tost_paired():
    x = [103.4, 59.92, 68.17, 94.54, 69.48, 72.17, 74.37, 84.44, 96.74, 94.26, 48.52, 95.68]
    y = [90.11, 77.71, 77.71, 97.51, 58.21, 101.3, 79.84, 96.06, 89.3, 97.22, 61.62, 85.8]

    corr = pg.corr(x=x, y=y)['r'].tolist()[0]

    result = tost_paired(len(x), statistics.mean(x), statistics.mean(y),
                         statistics.stdev(x), statistics.stdev(y), corr,
                         -0.4, 0.4)
    assert result['dif'] == -4.225
    assert result['t'] == (0.257861, -2.5134203)
    assert result['p'] == (0.4006374, 0.0144053)
    assert result['degrees_of_freedom'] == 11
    assert result['ci_tost'] == (-10.9529216, 2.5029216)
    assert result['ci_ttest'] == (-12.4705487, 4.0205487)
    assert result['xlim'] == (-12.298506, 6.5366086)
    assert result['combined_outcome'] == 'no_diff_no_eqv'


if __name__ == "__main__":
    test_tost_paired()