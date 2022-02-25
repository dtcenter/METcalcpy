# ============================*
 # ** Copyright UCAR (c) 2020
 # ** University Corporation for Atmospheric Research (UCAR)
 # ** National Center for Atmospheric Research (NCAR)
 # ** Research Applications Lab (RAL)
 # ** P.O.Box 3000, Boulder, Colorado, 80307-3000, USA
 # ============================*
 
 
 
"""
Program Name: tost_paired.py
"""

from scipy.stats import t, nct
import sys
import math

from metcalcpy.util.utils import round_half_up, PRECISION

CODE_TO_OUTCOME_TO_MESSAGE = {
    'diff_eqv': 'statistically different from zero and statistically equivalent to zero',
    'diff_no_eqv': 'statistically different from zero and statistically not equivalent to zero',
    'no_diff_eqv': 'statistically not different from zero and statistically equivalent to zero',
    'no_diff_no_eqv': 'statistically not different from zero and statistically not equivalent to zero'
}


def pt(q, df, ncp=0, lower_tail=True):
    """
    Calculates the cumulative of the t-distribution

    Args:
        q - vector of quantiles
        df - degrees of freedom (> 0)
        ncp - array_like shape parameters
        lower_tail - if True (default), probabilities are P[X ≤ x], otherwise, P[X > x].

    Returns:
        the cumulative of the t-distribution
    """

    if ncp == 0:
        result = t.cdf(x=q, df=df, loc=0, scale=1)
    else:
        result = nct.cdf(x=q, df=df, nc=ncp, loc=0, scale=1)
    if lower_tail is False:
        result = 1 - result
    return result


def qt(p, df, ncp=0):
    """
    Calculates the quantile function of the t-distribution

     Args:
        p - array_like quantiles
        df - array_like shape parameters
        ncp - array_like shape parameters

    Returns:
        tquantile function of the t-distribution
    """
    if ncp == 0:
        result = t.ppf(q=p, df=df, loc=0, scale=1)
    else:
        result = nct.ppf(q=p, df=df, nc=ncp, loc=0, scale=1)
    return result


def tost_paired(n, m1, m2, sd1, sd2, r12, low_eqbound_dz, high_eqbound_dz, alpha=None):
    """
    TOST function for a dependent t-test (Cohen's dz). Based on Rscript function TOSTpaired

    Args:
        n: sample size (pairs)
        m1: mean of group 1
        m2: mean of group 2
        sd1: standard deviation of group 1
        sd2: standard deviation of group 2
        r12: correlation of dependent variable between group 1 and group 2
        low_eqbound_dz: lower equivalence bounds (e.g., -0.5) expressed in standardized mean difference (Cohen's dz)
        high_eqbound_dz: upper equivalence bounds (e.g., 0.5) expressed in standardized mean difference (Cohen's dz)
        alpha: alpha level (default = 0.05)

    Returns:
        Returns a dictionary with calculated TOST values
                dif - Mean Difference
                t - TOST t-values 1 and 2 as a tuple
                p - TOST p-values and 2 as a tuple
                degrees_of_freedom - degrees of freedom
                ci_tost - confidence interval TOST Lower and Upper limit as a tuple
                ci_ttest - confidence interval TTEST Lower and Upper limit as a tuple
                eqbound - equivalence bound low and high as a tuple
                xlim - limits for x-axis
                combined_outcome - outcome
                test_outcome - pt test outcome
                tist_outcome - TOST outcome

    """
    if not alpha:
        alpha = 0.05
    if low_eqbound_dz >= high_eqbound_dz:
        print(
            'WARNING: The lower bound is equal to or larger than the upper bound.'
            ' Check the plot and output to see if the bounds are specified as you intended.')

    if n < 2:
        print("The sample size should be larger than 1.")
        sys.exit()

    if 1 <= alpha or alpha <= 0:
        print("The alpha level should be a positive value between 0 and 1.")
        sys.exit()
    if sd1 <= 0 or sd2 <= 0:
        print("The standard deviation should be a positive value.")
        sys.exit()
    if 1 < r12 or r12 < -1:
        print("The correlation should be a value between -1 and 1.")
        sys.exit()

    sdif = math.sqrt(sd1 * sd1 + sd2 * sd2 - 2 * r12 * sd1 * sd2)
    low_eqbound = low_eqbound_dz * sdif
    high_eqbound = high_eqbound_dz * sdif
    se = sdif / math.sqrt(n)
    t = (m1 - m2) / se
    degree_f = n - 1

    pttest = 2 * pt(abs(t), degree_f, lower_tail=False)

    t1 = ((m1 - m2) - (low_eqbound_dz * sdif)) / se
    p1 = pt(t1, degree_f, lower_tail=False)
    t2 = ((m1 - m2) - (high_eqbound_dz * sdif)) / se
    p2 = pt(t2, degree_f, lower_tail=True)

    ll90 = ((m1 - m2) - qt(1 - alpha, degree_f) * se)
    ul90 = ((m1 - m2) + qt(1 - alpha, degree_f) * se)
    ptost = max(p1, p2)

    dif = (m1 - m2)
    ll95 = ((m1 - m2) - qt(1 - (alpha / 2), degree_f) * se)
    ul95 = ((m1 - m2) + qt(1 - (alpha / 2), degree_f) * se)
    xlim_l = min(ll90, low_eqbound) - max(ul90 - ll90, high_eqbound - low_eqbound) / 10
    xlim_u = max(ul90, high_eqbound) + max(ul90 - ll90, high_eqbound - low_eqbound) / 10

    if pttest <= alpha and ptost <= alpha:
        combined_outcome = 'diff_eqv'

    if pttest < alpha and ptost > alpha:
        combined_outcome = 'diff_no_eqv'

    if pttest > alpha and ptost <= alpha:
        combined_outcome = 'no_diff_eqv'

    if pttest > alpha and ptost > alpha:
        combined_outcome = 'no_diff_no_eqv'

    if pttest < alpha:
        test_outcome = 'significant'
    else:
        test_outcome = 'non-significant'

    if ptost < alpha:
        tost_outcome = 'significant'
    else:
        tost_outcome = 'non-significant'

    return {
        'dif': round_half_up(dif, PRECISION),
        't': (round_half_up(t1, PRECISION), round_half_up(t2, PRECISION)),
        'p': (round_half_up(p1, PRECISION), round_half_up(p2, PRECISION)),
        'degrees_of_freedom': round_half_up(degree_f, PRECISION),
        'ci_tost': (round_half_up(ll90, PRECISION), round_half_up(ul90, PRECISION) ),
        'ci_ttest': (round_half_up(ll95, PRECISION), round_half_up(ul95, PRECISION)),
        'eqbound': (round_half_up(low_eqbound, PRECISION), round_half_up(high_eqbound, PRECISION)),
        'xlim': (round_half_up(xlim_l, PRECISION), round_half_up(xlim_u, PRECISION)),
        'combined_outcome': combined_outcome,
        'test_outcome': test_outcome,
        'tost_outcome': tost_outcome
    }





