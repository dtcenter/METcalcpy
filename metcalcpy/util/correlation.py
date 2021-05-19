"""
These methods are the part of pingouin package and was moved from it to METcalcpy without changes
"""
import numpy as np
import pandas as pd
import itertools as it
import numbers
import warnings

from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.optimize import brenth
from scipy.integrate import quad
from scipy import stats

from math import pi, exp, log, lgamma


def corr(x, y, tail='two-sided', method='pearson', **kwargs):
    """(Robust) correlation between two variables.
    This method was the patrt of pingouin package and was moved from it to METcalcpy

    Parameters
    ----------
    x, y : array_like
        First and second set of observations. ``x`` and ``y`` must be
        independent.
    tail : string
        Specify whether to return ``'one-sided'`` or ``'two-sided'`` p-value.
        Note that the former are simply half the latter.
    method : string
        Correlation type:

        * ``'pearson'``: Pearson :math:`r` product-moment correlation
        * ``'spearman'``: Spearman :math:`\\rho` rank-order correlation
        * ``'kendall'``: Kendall's :math:`\\tau_B` correlation
          (for ordinal data)
        * ``'bicor'``: Biweight midcorrelation (robust)
        * ``'percbend'``: Percentage bend correlation (robust)
        * ``'shepherd'``: Shepherd's pi correlation (robust)
        * ``'skipped'``: Skipped correlation (robust)
    **kwargs : optional
        Optional argument(s) passed to the lower-level functions.

    Returns
    -------
    stats : :py:class:`pandas.DataFrame`

        * ``'n'``: Sample size (after removal of missing values)
        * ``'outliers'``: number of outliers, only if a robust method was used
        * ``'r'``: Correlation coefficient
        * ``'CI95'``: 95% parametric confidence intervals around :math:`r`
        * ``'r2'``: R-squared (:math:`= r^2`)
        * ``'adj_r2'``: Adjusted R-squared
        * ``'p-val'``: tail of the test
        * ``'BF10'``: Bayes Factor of the alternative hypothesis
          (only for Pearson correlation)
        * ``'power'``: achieved power of the test (= 1 - type II error).

    See also
    --------
    pairwise_corr : Pairwise correlation between columns of a pandas DataFrame
    partial_corr : Partial correlation
    rm_corr : Repeated measures correlation

    Notes
    -----
    The `Pearson correlation coefficient
    <https://en.wikipedia.org/wiki/Pearson_correlation_coefficient>`_
    measures the linear relationship between two datasets. Strictly speaking,
    Pearson's correlation requires that each dataset be normally distributed.
    Correlations of -1 or +1 imply a perfect negative and positive linear
    relationship, respectively, with 0 indicating the absence of association.

    .. math::
        r_{xy} = \\frac{\\sum_i(x_i - \\bar{x})(y_i - \\bar{y})}
        {\\sqrt{\\sum_i(x_i - \\bar{x})^2} \\sqrt{\\sum_i(y_i - \\bar{y})^2}}
        = \\frac{\\text{cov}(x, y)}{\\sigma_x \\sigma_y}

    where :math:`\\text{cov}` is the sample covariance and :math:`\\sigma`
    is the sample standard deviation.

    If ``method='pearson'``, The Bayes Factor is calculated using the
    :py:func:`pingouin.bayesfactor_pearson` function.

    The `Spearman correlation coefficient
    <https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient>`_
    is a non-parametric measure of the monotonicity of the relationship between
    two datasets. Unlike the Pearson correlation, the Spearman correlation does
    not assume that both datasets are normally distributed. Correlations of -1
    or +1 imply an exact negative and positive monotonic relationship,
    respectively. Mathematically, the Spearman correlation coefficient is
    defined as the Pearson correlation coefficient between the
    `rank variables <https://en.wikipedia.org/wiki/Ranking>`_.

    The `Kendall correlation coefficient
    <https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient>`_
    is a measure of the correspondence between two rankings. Values also range
    from -1 (perfect disagreement) to 1 (perfect agreement), with 0 indicating
    the absence of association. Consistent with
    :py:func:`scipy.stats.kendalltau`, Pingouin returns the Tau-b coefficient,
    which adjusts for ties:

    .. math:: \\tau_B = \\frac{(P - Q)}{\\sqrt{(P + Q + T) (P + Q + U)}}

    where :math:`P` is the number of concordant pairs, :math:`Q` the number of
    discordand pairs, :math:`T` the number of ties in x, and :math:`U`
    the number of ties in y.

    The `biweight midcorrelation
    <https://en.wikipedia.org/wiki/Biweight_midcorrelation>`_ and
    percentage bend correlation [1]_ are both robust methods that
    protects against *univariate* outliers by down-weighting observations that
    deviate too much from the median.

    The Shepherd pi [2]_ correlation and skipped [3]_, [4]_ correlation are
    both robust methods that returns the Spearman correlation coefficient after
    removing *bivariate* outliers. Briefly, the Shepherd pi uses a
    bootstrapping of the Mahalanobis distance to identify outliers, while the
    skipped correlation is based on the minimum covariance determinant
    (which requires scikit-learn). Note that these two methods are
    significantly slower than the previous ones.

    .. important:: Please note that rows with missing values (NaN) are
        automatically removed.

    References
    ----------
    .. [1] Wilcox, R.R., 1994. The percentage bend correlation coefficient.
       Psychometrika 59, 601–616. https://doi.org/10.1007/BF02294395

    .. [2] Schwarzkopf, D.S., De Haas, B., Rees, G., 2012. Better ways to
       improve standards in brain-behavior correlation analysis. Front.
       Hum. Neurosci. 6, 200. https://doi.org/10.3389/fnhum.2012.00200

    .. [3] Rousselet, G.A., Pernet, C.R., 2012. Improving standards in
       brain-behavior correlation analyses. Front. Hum. Neurosci. 6, 119.
       https://doi.org/10.3389/fnhum.2012.00119

    .. [4] Pernet, C.R., Wilcox, R., Rousselet, G.A., 2012. Robust correlation
       analyses: false positive and power validation using a new open
       source matlab toolbox. Front. Psychol. 3, 606.
       https://doi.org/10.3389/fpsyg.2012.00606

    Examples
    --------
    1. Pearson correlation

    >>> import numpy as np
    >>> import pingouin as pg
    >>> # Generate random correlated samples
    >>> np.random.seed(123)
    >>> mean, cov = [4, 6], [(1, .5), (.5, 1)]
    >>> x, y = np.random.multivariate_normal(mean, cov, 30).T
    >>> # Compute Pearson correlation
    >>> pg.corr(x, y).round(3)
              n      r         CI95%     r2  adj_r2  p-val  BF10  power
    pearson  30  0.491  [0.16, 0.72]  0.242   0.185  0.006  8.55  0.809

    2. Pearson correlation with two outliers

    >>> x[3], y[5] = 12, -8
    >>> pg.corr(x, y).round(3)
              n      r          CI95%     r2  adj_r2  p-val   BF10  power
    pearson  30  0.147  [-0.23, 0.48]  0.022  -0.051  0.439  0.302  0.121

    3. Spearman correlation (robust to outliers)

    >>> pg.corr(x, y, method="spearman").round(3)
               n      r         CI95%     r2  adj_r2  p-val  power
    spearman  30  0.401  [0.05, 0.67]  0.161   0.099  0.028   0.61

    4. Biweight midcorrelation (robust)

    >>> pg.corr(x, y, method="bicor").round(3)
            n      r         CI95%     r2  adj_r2  p-val  power
    bicor  30  0.393  [0.04, 0.66]  0.155   0.092  0.031  0.592

    5. Percentage bend correlation (robust)

    >>> pg.corr(x, y, method='percbend').round(3)
               n      r         CI95%     r2  adj_r2  p-val  power
    percbend  30  0.389  [0.03, 0.66]  0.151   0.089  0.034  0.581

    6. Shepherd's pi correlation (robust)

    >>> pg.corr(x, y, method='shepherd').round(3)
               n  outliers      r         CI95%     r2  adj_r2  p-val  power
    shepherd  30         2  0.437  [0.09, 0.69]  0.191   0.131   0.02  0.694

    7. Skipped spearman correlation (robust)

    >>> pg.corr(x, y, method='skipped').round(3)
              n  outliers      r         CI95%     r2  adj_r2  p-val  power
    skipped  30         2  0.437  [0.09, 0.69]  0.191   0.131   0.02  0.694

    8. One-tailed Pearson correlation

    >>> pg.corr(x, y, tail="one-sided", method='pearson').round(3)
              n      r          CI95%     r2  adj_r2  p-val   BF10  power
    pearson  30  0.147  [-0.23, 0.48]  0.022  -0.051   0.22  0.467  0.194

    9. Using columns of a pandas dataframe

    >>> import pandas as pd
    >>> data = pd.DataFrame({'x': x, 'y': y})
    >>> pg.corr(data['x'], data['y']).round(3)
              n      r          CI95%     r2  adj_r2  p-val   BF10  power
    pearson  30  0.147  [-0.23, 0.48]  0.022  -0.051  0.439  0.302  0.121
    """
    # Safety check
    x = np.asarray(x)
    y = np.asarray(y)
    assert x.ndim == y.ndim == 1, 'x and y must be 1D array.'
    assert x.size == y.size, 'x and y must have the same length.'
    _msg = 'tail must be "two-sided" or "one-sided".'
    assert tail in ['two-sided', 'one-sided'], _msg

    # Remove rows with missing values
    x, y = remove_na(x, y, paired=True)
    nx = x.size

    # Compute correlation coefficient
    if method == 'pearson':
        r, pval = pearsonr(x, y)
    elif method == 'spearman':
        r, pval = spearmanr(x, y, **kwargs)
    elif method == 'kendall':
        r, pval = kendalltau(x, y, **kwargs)
    elif method == 'bicor':
        r, pval = bicor(x, y, **kwargs)
    elif method == 'percbend':
        r, pval = percbend(x, y, **kwargs)
    elif method == 'shepherd':
        r, pval, outliers = shepherd(x, y, **kwargs)
    elif method == 'skipped':
        r, pval, outliers = skipped(x, y, **kwargs)
    else:
        raise ValueError(f'Method "{method}" not recognized.')

    if np.isnan(r):
        # Correlation failed -- new in version v0.3.4, instead of raising an
        # error we just return a dataframe full of NaN (except sample size).
        # This avoid sudden stop in pingouin.pairwise_corr.
        return pd.DataFrame({'n': nx, 'r': np.nan, 'CI95%': np.nan,
                             'r2': np.nan, 'adj_r2': np.nan, 'p-val': np.nan,
                             'BF10': np.nan, 'power': np.nan}, index=[method])

    # Compute r2 and adj_r2
    r2 = r ** 2
    adj_r2 = 1 - (((1 - r2) * (nx - 1)) / (nx - 3))

    # Compute the parametric 95% confidence interval and power
    ci = compute_esci(stat=r, nx=nx, ny=nx, eftype='r', decimals=6)
    pr = power_corr(r=r, n=nx, power=None, alpha=0.05, tail=tail),

    # Create dictionnary
    stats = {'n': nx,
             'r': r,
             'r2': r2,
             'adj_r2': adj_r2,
             'CI95%': [ci],
             'p-val': pval if tail == 'two-sided' else .5 * pval,
             'power': pr
             }

    if method in ['shepherd', 'skipped']:
        stats['outliers'] = sum(outliers)

    # Compute the BF10 for Pearson correlation only
    if method == 'pearson':
        stats['BF10'] = bayesfactor_pearson(r, nx, tail=tail)

    # Convert to DataFrame
    stats = pd.DataFrame.from_records(stats, index=[method])

    # Define order
    col_keep = ['n', 'outliers', 'r', 'CI95%', 'r2', 'adj_r2', 'p-val',
                'BF10', 'power']
    col_order = [k for k in col_keep if k in stats.keys().tolist()]
    return _postprocess_dataframe(stats)[col_order]


def remove_na(x, y=None, paired=False, axis='rows'):
    """Remove missing values along a given axis in one or more (paired) numpy
    arrays.

    Parameters
    ----------
    x, y : 1D or 2D arrays
        Data. ``x`` and ``y`` must have the same number of dimensions.
        ``y`` can be None to only remove missing values in ``x``.
    paired : bool
        Indicates if the measurements are paired or not.
    axis : str
        Axis or axes along which missing values are removed.
        Can be 'rows' or 'columns'. This has no effect if ``x`` and ``y`` are
        one-dimensional arrays.

    Returns
    -------
    x, y : np.ndarray
        Data without missing values

    Examples
    --------
    Single 1D array

    >>> import numpy as np
    >>> from pingouin import remove_na
    >>> x = [6.4, 3.2, 4.5, np.nan]
    >>> remove_na(x)
    array([6.4, 3.2, 4.5])

    With two paired 1D arrays

    >>> y = [2.3, np.nan, 5.2, 4.6]
    >>> remove_na(x, y, paired=True)
    (array([6.4, 4.5]), array([2.3, 5.2]))

    With two independent 2D arrays

    >>> x = np.array([[4, 2], [4, np.nan], [7, 6]])
    >>> y = np.array([[6, np.nan], [3, 2], [2, 2]])
    >>> x_no_nan, y_no_nan = remove_na(x, y, paired=False)
    """
    # Safety checks
    x = np.asarray(x)
    assert x.size > 1, 'x must have more than one element.'
    assert axis in ['rows', 'columns'], 'axis must be rows or columns.'

    if y is None:
        return _remove_na_single(x, axis=axis)
    elif isinstance(y, (int, float, str)):
        return _remove_na_single(x, axis=axis), y
    else:  # y is list, np.array, pd.Series
        y = np.asarray(y)
        # Make sure that we just pass-through if y have only 1 element
        if y.size == 1:
            return _remove_na_single(x, axis=axis), y
        if x.ndim != y.ndim or paired is False:
            # x and y do not have the same dimension
            x_no_nan = _remove_na_single(x, axis=axis)
            y_no_nan = _remove_na_single(y, axis=axis)
            return x_no_nan, y_no_nan

    # At this point, we assume that x and y are paired and have same dimensions
    if x.ndim == 1:
        # 1D arrays
        x_mask = ~np.isnan(x)
        y_mask = ~np.isnan(y)
    else:
        # 2D arrays
        ax = 1 if axis == 'rows' else 0
        x_mask = ~np.any(np.isnan(x), axis=ax)
        y_mask = ~np.any(np.isnan(y), axis=ax)

    # Check if missing values are present
    if ~x_mask.all() or ~y_mask.all():
        ax = 0 if axis == 'rows' else 1
        ax = 0 if x.ndim == 1 else ax
        both = np.logical_and(x_mask, y_mask)
        x = x.compress(both, axis=ax)
        y = y.compress(both, axis=ax)
    return x, y


def _remove_na_single(x, axis='rows'):
    """Remove NaN in a single array.
    This is an internal Pingouin function.
    """
    if x.ndim == 1:
        # 1D arrays
        x_mask = ~np.isnan(x)
    else:
        # 2D arrays
        ax = 1 if axis == 'rows' else 0
        x_mask = ~np.any(np.isnan(x), axis=ax)
    # Check if missing values are present
    if ~x_mask.all():
        ax = 0 if axis == 'rows' else 1
        ax = 0 if x.ndim == 1 else ax
        x = x.compress(x_mask, axis=ax)
    return x


def bicor(x, y, c=9):
    """
    Biweight midcorrelation.

    Parameters
    ----------
    x, y : array_like
        First and second set of observations. x and y must be independent.
    c : float
        Tuning constant for the biweight estimator (default = 9.0).

    Returns
    -------
    r : float
        Correlation coefficient.
    pval : float
        Two-tailed p-value.

    Notes
    -----
    This function will return (np.nan, np.nan) if mad(x) == 0 or mad(y) == 0.

    References
    ----------
    https://en.wikipedia.org/wiki/Biweight_midcorrelation

    https://docs.astropy.org/en/stable/api/astropy.stats.biweight.biweight_midcovariance.html

    Langfelder, P., & Horvath, S. (2012). Fast R Functions for Robust
    Correlations and Hierarchical Clustering. Journal of Statistical Software,
    46(11). https://www.ncbi.nlm.nih.gov/pubmed/23050260
    """
    from scipy.stats import t
    # Calculate median
    nx = x.size
    x_median = np.median(x)
    y_median = np.median(y)
    # Raw median absolute deviation
    x_mad = np.median(np.abs(x - x_median))
    y_mad = np.median(np.abs(y - y_median))
    if x_mad == 0 or y_mad == 0:
        # From Langfelder and Horvath 2012:
        # "Strictly speaking, a call to bicor in R should return a missing
        # value if mad(x) = 0 or mad(y) = 0." This avoids division by zero.
        return np.nan, np.nan
    # Calculate weights
    u = (x - x_median) / (c * x_mad)
    v = (y - y_median) / (c * y_mad)
    w_x = (1 - u ** 2) ** 2 * ((1 - np.abs(u)) > 0)
    w_y = (1 - v ** 2) ** 2 * ((1 - np.abs(v)) > 0)
    # Normalize x and y by weights
    x_norm = (x - x_median) * w_x
    y_norm = (y - y_median) * w_y
    denom = (np.sqrt((x_norm ** 2).sum()) * np.sqrt((y_norm ** 2).sum()))
    # Calculate r, t and two-sided p-value
    r = (x_norm * y_norm).sum() / denom
    tval = r * np.sqrt((nx - 2) / (1 - r ** 2))
    pval = 2 * t.sf(abs(tval), nx - 2)
    return r, pval


def percbend(x, y, beta=.2):
    """
    Percentage bend correlation (Wilcox 1994).

    Parameters
    ----------
    x, y : array_like
        First and second set of observations. x and y must be independent.
    beta : float
        Bending constant for omega (0 <= beta <= 0.5).

    Returns
    -------
    r : float
        Percentage bend correlation coefficient.
    pval : float
        Two-tailed p-value.

    Notes
    -----
    Code inspired by Matlab code from Cyril Pernet and Guillaume Rousselet.

    References
    ----------
    .. [1] Wilcox, R.R., 1994. The percentage bend correlation coefficient.
       Psychometrika 59, 601–616. https://doi.org/10.1007/BF02294395

    .. [2] Pernet CR, Wilcox R, Rousselet GA. Robust Correlation Analyses:
       False Positive and Power Validation Using a New Open Source Matlab
       Toolbox. Frontiers in Psychology. 2012;3:606.
       doi:10.3389/fpsyg.2012.00606.
    """
    from scipy.stats import t
    X = np.column_stack((x, y))
    nx = X.shape[0]
    M = np.tile(np.median(X, axis=0), nx).reshape(X.shape)
    W = np.sort(np.abs(X - M), axis=0)
    m = int((1 - beta) * nx)
    omega = W[m - 1, :]
    P = (X - M) / omega
    P[np.isinf(P)] = 0
    P[np.isnan(P)] = 0

    # Loop over columns
    a = np.zeros((2, nx))
    for c in [0, 1]:
        psi = P[:, c]
        i1 = np.where(psi < -1)[0].size
        i2 = np.where(psi > 1)[0].size
        s = X[:, c].copy()
        s[np.where(psi < -1)[0]] = 0
        s[np.where(psi > 1)[0]] = 0
        pbos = (np.sum(s) + omega[c] * (i2 - i1)) / (s.size - i1 - i2)
        a[c] = (X[:, c] - pbos) / omega[c]

    # Bend
    a[a <= -1] = -1
    a[a >= 1] = 1

    # Get r, tval and pval
    a, b = a
    r = (a * b).sum() / np.sqrt((a ** 2).sum() * (b ** 2).sum())
    tval = r * np.sqrt((nx - 2) / (1 - r ** 2))
    pval = 2 * t.sf(abs(tval), nx - 2)
    return r, pval


def shepherd(x, y, n_boot=200):
    """
    Shepherd's Pi correlation, equivalent to Spearman's rho after outliers
    removal.

    Parameters
    ----------
    x, y : array_like
        First and second set of observations. x and y must be independent.
    n_boot : int
        Number of bootstrap samples to calculate.

    Returns
    -------
    r : float
        Pi correlation coefficient
    pval : float
        Two-tailed adjusted p-value.
    outliers : array of bool
        Indicate if value is an outlier or not

    Notes
    -----
    It first bootstraps the Mahalanobis distances, removes all observations
    with m >= 6 and finally calculates the correlation of the remaining data.

    Pi is Spearman's Rho after outlier removal.
    """
    X = np.column_stack((x, y))
    # Bootstrapping on Mahalanobis distance
    m = bsmahal(X, X, n_boot)
    # Determine outliers
    outliers = (m >= 6)
    # Compute correlation
    r, pval = spearmanr(x[~outliers], y[~outliers])
    # (optional) double the p-value to achieve a nominal false alarm rate
    # pval *= 2
    # pval = 1 if pval > 1 else pval
    return r, pval, outliers


def skipped(x, y, corr_type='spearman'):
    """Skipped correlation (Rousselet and Pernet 2012).

    Parameters
    ----------
    x, y : array_like
        First and second set of observations. x and y must be independent.
    corr_type : str
        Method used to compute the correlation after outlier removal. Can be
        either 'spearman' (default) or 'pearson'.

    Returns
    -------
    r : float
        Skipped correlation coefficient.
    pval : float
        Two-tailed p-value.
    outliers : array of bool
        Indicate if value is an outlier or not

    Notes
    -----
    The skipped correlation involves multivariate outlier detection using a
    projection technique (Wilcox, 2004, 2005). First, a robust estimator of
    multivariate location and scatter, for instance the minimum covariance
    determinant estimator (MCD; Rousseeuw, 1984; Rousseeuw and van Driessen,
    1999; Hubert et al., 2008) is computed. Second, data points are
    orthogonally projected on lines joining each of the data point to the
    location estimator. Third, outliers are detected using a robust technique.
    Finally, Spearman correlations are computed on the remaining data points
    and calculations are adjusted by taking into account the dependency among
    the remaining data points.

    Code inspired by Matlab code from Cyril Pernet and Guillaume
    Rousselet [1]_.

    Requires scikit-learn.

    References
    ----------
    .. [1] Pernet CR, Wilcox R, Rousselet GA. Robust Correlation Analyses:
       False Positive and Power Validation Using a New Open Source Matlab
       Toolbox. Frontiers in Psychology. 2012;3:606.
       doi:10.3389/fpsyg.2012.00606.
    """
    # Check that sklearn is installed
    _is_sklearn_installed(raise_error=True)
    from scipy.stats import chi2
    from sklearn.covariance import MinCovDet
    X = np.column_stack((x, y))
    nrows, ncols = X.shape
    gval = np.sqrt(chi2.ppf(0.975, 2))
    # Compute center and distance to center
    center = MinCovDet(random_state=42).fit(X).location_
    B = X - center
    bot = (B ** 2).sum(axis=1)
    # Loop over rows
    dis = np.zeros(shape=(nrows, nrows))
    for i in np.arange(nrows):
        if bot[i] != 0:  # Avoid division by zero error
            dis[i, :] = np.linalg.norm(
                B.dot(B[i, :, None]) * B[i, :] / bot[i], axis=1)

    # Detect outliers
    def idealf(x):
        """Compute the ideal fourths IQR (Wilcox 2012).
        """
        n = len(x)
        j = int(np.floor(n / 4 + 5 / 12))
        y = np.sort(x)
        g = (n / 4) - j + (5 / 12)
        low = (1 - g) * y[j - 1] + g * y[j]
        k = n - j + 1
        up = (1 - g) * y[k - 1] + g * y[k - 2]
        return up - low

    # One can either use the MAD or the IQR (see Wilcox 2012)
    # MAD = mad(dis, axis=1)
    iqr = np.apply_along_axis(idealf, 1, dis)
    thresh = (np.median(dis, axis=1) + gval * iqr)
    outliers = np.apply_along_axis(np.greater, 0, dis, thresh).any(axis=0)
    # Compute correlation on remaining data
    if corr_type == 'spearman':
        r, pval = spearmanr(X[~outliers, 0], X[~outliers, 1])
    else:
        r, pval = pearsonr(X[~outliers, 0], X[~outliers, 1])
    return r, pval, outliers


def bsmahal(a, b, n_boot=200):
    """
    Bootstraps Mahalanobis distances for Shepherd's pi correlation.

    Parameters
    ----------
    a : ndarray (shape=(n, 2))
        Data
    b : ndarray (shape=(n, 2))
        Data
    n_boot : int
        Number of bootstrap samples to calculate.

    Returns
    -------
    m : ndarray (shape=(n,))
        Mahalanobis distance for each row in a, averaged across all the
        bootstrap resamples.
    """
    n, m = b.shape
    MD = np.zeros((n, n_boot))
    nr = np.arange(n)
    xB = np.random.choice(nr, size=(n_boot, n), replace=True)
    # Bootstrap the MD
    for i in np.arange(n_boot):
        s1 = b[xB[i, :], 0]
        s2 = b[xB[i, :], 1]
        X = np.column_stack((s1, s2))
        mu = X.mean(0)
        _, R = np.linalg.qr(X - mu)
        sol = np.linalg.solve(R.T, (a - mu).T)
        MD[:, i] = np.sum(sol ** 2, 0) * (n - 1)
    # Average across all bootstraps
    return MD.mean(1)


def bayesfactor_pearson(r, n, tail='two-sided', method='ly', kappa=1.):
    """
    Bayes Factor of a Pearson correlation.

    Parameters
    ----------
    r : float
        Pearson correlation coefficient.
    n : int
        Sample size.
    tail : float
        Tail of the alternative hypothesis. Can be *'two-sided'*,
        *'one-sided'*, *'greater'* or *'less'*. *'greater'* corresponds to a
        positive correlation, *'less'* to a negative correlation.
        If *'one-sided'*, the directionality is inferred based on the ``r``
        value (= *'greater'* if ``r`` > 0, *'less'* if ``r`` < 0).
    method : str
        Method to compute the Bayes Factor. Can be *'ly'* (default) or
        *'wetzels'*. The former has an exact analytical solution, while the
        latter requires integral solving (and is therefore slower). *'wetzels'*
        was the default in Pingouin <= 0.2.5. See Notes for details.
    kappa : float
        Kappa factor. This is sometimes called the *rscale* parameter, and
        is only used when ``method`` is *'ly'*.

    Returns
    -------
    bf : float
        Bayes Factor (BF10).
        The Bayes Factor quantifies the evidence in favour of the alternative
        hypothesis.

    See also
    --------
    corr : (Robust) correlation between two variables
    pairwise_corr : Pairwise correlation between columns of a pandas DataFrame
    bayesfactor_ttest : Bayes Factor of a T-test
    bayesfactor_binom : Bayes Factor of a binomial test

    Notes
    -----
    To compute the Bayes Factor directly from the raw data, use the
    :py:func:`pingouin.corr` function.

    The two-sided **Wetzels Bayes Factor** (also called *JZS Bayes Factor*)
    is calculated using the equation 13 and associated R code of [1]_:

    .. math::

        \\text{BF}_{10}(n, r) = \\frac{\\sqrt{n/2}}{\\gamma(1/2)}*
        \\int_{0}^{\\infty}e((n-2)/2)*
        log(1+g)+(-(n-1)/2)log(1+(1-r^2)*g)+(-3/2)log(g)-n/2g

    where :math:`n` is the sample size, :math:`r` is the Pearson correlation
    coefficient and :math:`g` is is an auxiliary variable that is integrated
    out numerically. Since the Wetzels Bayes Factor requires solving an
    integral, it is slower than the analytical solution described below.

    The two-sided **Ly Bayes Factor** (also called *Jeffreys
    exact Bayes Factor*) is calculated using equation 25 of [2]_:

    .. math::

        \\text{BF}_{10;k}(n, r) = \\frac{2^{\\frac{k-2}{k}}\\sqrt{\\pi}}
        {\\beta(\\frac{1}{k}, \\frac{1}{k})} \\cdot
        \\frac{\\Gamma(\\frac{2+k(n-1)}{2k})}{\\Gamma(\\frac{2+nk}{2k})}
        \\cdot 2F_1(\\frac{n-1}{2}, \\frac{n-1}{2}, \\frac{2+nk}{2k}, r^2)

    The one-sided version is described in eq. 27 and 28 of Ly et al, 2016.
    Please take note that the one-sided test requires the
    `mpmath <http://mpmath.org/>`_ package.

    Results have been validated against JASP and the BayesFactor R package.

    References
    ----------
    .. [1] Ly, A., Verhagen, J. & Wagenmakers, E.-J. Harold Jeffreys’s default
       Bayes factor hypothesis tests: Explanation, extension, and
       application in psychology. J. Math. Psychol. 72, 19–32 (2016).

    .. [2] Wetzels, R. & Wagenmakers, E.-J. A default Bayesian hypothesis test
       for correlations and partial correlations. Psychon. Bull. Rev. 19,
       1057–1064 (2012).

    Examples
    --------
    Bayes Factor of a Pearson correlation

    >>> from pingouin import bayesfactor_pearson
    >>> r, n = 0.6, 20
    >>> bf = bayesfactor_pearson(r, n)
    >>> print("Bayes Factor: %.3f" % bf)
    Bayes Factor: 10.634

    Compare to Wetzels method:

    >>> bf = bayesfactor_pearson(r, n, method='wetzels')
    >>> print("Bayes Factor: %.3f" % bf)
    Bayes Factor: 8.221

    One-sided test

    >>> bf10pos = bayesfactor_pearson(r, n, tail='greater')
    >>> bf10neg = bayesfactor_pearson(r, n, tail='less')
    >>> print("BF-pos: %.3f, BF-neg: %.3f" % (bf10pos, bf10neg))
    BF-pos: 21.185, BF-neg: 0.082

    We can also only pass ``tail='one-sided'`` and Pingouin will automatically
    infer the directionality of the test based on the ``r`` value.

    >>> print("BF: %.3f" % bayesfactor_pearson(r, n, tail='one-sided'))
    BF: 21.185
    """
    from scipy.special import gamma, betaln, hyp2f1
    assert method.lower() in ['ly', 'wetzels'], 'Method not recognized.'
    assert tail.lower() in ['two-sided', 'one-sided', 'greater', 'less',
                            'g', 'l', 'positive', 'negative', 'pos', 'neg']

    # Wrong input
    if not np.isfinite(r) or n < 2:
        return np.nan
    assert -1 <= r <= 1, 'r must be between -1 and 1.'

    if tail.lower() != 'two-sided' and method.lower() == 'wetzels':
        warnings.warn("One-sided Bayes Factor are not supported by the "
                      "Wetzels's method. Switching to method='ly'.")
        method = 'ly'

    if method.lower() == 'wetzels':
        # Wetzels & Wagenmakers, 2012. Integral solving

        def fun(g, r, n):
            return exp(((n - 2) / 2) * log(1 + g) + (-(n - 1) / 2)
                       * log(1 + (1 - r ** 2) * g) + (-3 / 2)
                       * log(g) + - n / (2 * g))

        integr = quad(fun, 0, np.inf, args=(r, n))[0]
        bf10 = np.sqrt((n / 2)) / gamma(1 / 2) * integr

    else:
        # Ly et al, 2016. Analytical solution.
        k = kappa
        lbeta = betaln(1 / k, 1 / k)
        log_hyperterm = log(hyp2f1(((n - 1) / 2), ((n - 1) / 2),
                                   ((n + 2 / k) / 2), r ** 2))
        bf10 = exp((1 - 2 / k) * log(2) + 0.5 * log(pi) - lbeta
                   + lgamma((n + 2 / k - 1) / 2) - lgamma((n + 2 / k) / 2) +
                   log_hyperterm)

        if tail.lower() != 'two-sided':
            # Directional test.
            # We need mpmath for the generalized hypergeometric function
            from .utils import _is_mpmath_installed
            _is_mpmath_installed(raise_error=True)
            from mpmath import hyp3f2
            hyper_term = float(hyp3f2(1, n / 2, n / 2, 3 / 2,
                                      (2 + k * (n + 1)) / (2 * k),
                                      r ** 2))
            log_term = 2 * (lgamma(n / 2) - lgamma((n - 1) / 2)) - lbeta
            C = 2 ** ((3 * k - 2) / k) * k * r / (2 + (n - 1) * k) * \
                exp(log_term) * hyper_term

            bf10neg = bf10 - C
            bf10pos = 2 * bf10 - bf10neg
            if tail.lower() in ['one-sided']:
                # Automatically find the directionality of the test based on r
                bf10 = bf10pos if r >= 0 else bf10neg
            elif tail.lower() in ['greater', 'g', 'positive', 'pos']:
                # We expect the correlation to be positive
                bf10 = bf10pos
            else:
                # We expect the correlation to be negative
                bf10 = bf10neg

    return bf10


def compute_esci(stat=None, nx=None, ny=None, paired=False, eftype='cohen',
                 confidence=.95, decimals=2):
    """Parametric confidence intervals around a Cohen d or a
    correlation coefficient.

    Parameters
    ----------
    stat : float
        Original effect size. Must be either a correlation coefficient or a
        Cohen-type effect size (Cohen d or Hedges g).
    nx, ny : int
        Length of vector x and y.
    paired : bool
        Indicates if the effect size was estimated from a paired sample.
        This is only relevant for cohen or hedges effect size.
    eftype : string
        Effect size type. Must be ``'r'`` (correlation) or ``'cohen'``
        (Cohen d or Hedges g).
    confidence : float
        Confidence level (0.95 = 95%)
    decimals : int
        Number of rounded decimals.

    Returns
    -------
    ci : array
        Desired converted effect size

    Notes
    -----
    To compute the parametric confidence interval around a
    **Pearson r correlation** coefficient, one must first apply a
    Fisher's r-to-z transformation:

    .. math:: z = 0.5 \\cdot \\ln \\frac{1 + r}{1 - r} = \\text{arctanh}(r)

    and compute the standard error:

    .. math:: \\text{SE} = \\frac{1}{\\sqrt{n - 3}}

    where :math:`n` is the sample size.

    The lower and upper confidence intervals - *in z-space* - are then
    given by:

    .. math:: \\text{ci}_z = z \\pm \\text{crit} \\cdot \\text{SE}

    where :math:`\\text{crit}` is the critical value of the normal distribution
    corresponding to the desired confidence level (e.g. 1.96 in case of a 95%
    confidence interval).

    These confidence intervals can then be easily converted back to *r-space*:

    .. math::

        \\text{ci}_r = \\frac{\\exp(2 \\cdot \\text{ci}_z) - 1}
        {\\exp(2 \\cdot \\text{ci}_z) + 1} = \\text{tanh}(\\text{ci}_z)

    A formula for calculating the confidence interval for a
    **Cohen d effect size** is given by Hedges and Olkin (1985, p86).
    If the effect size estimate from the sample is :math:`d`, then it follows a
    T distribution with standard error:

    .. math::

        \\text{SE} = \\sqrt{\\frac{n_x + n_y}{n_x \\cdot n_y} +
        \\frac{d^2}{2 (n_x + n_y)}}

    where :math:`n_x` and :math:`n_y` are the sample sizes of the two groups.

    In one-sample test or paired test, this becomes:

    .. math::

        \\text{SE} = \\sqrt{\\frac{1}{n_x} + \\frac{d^2}{2 n_x}}

    The lower and upper confidence intervals are then given by:

    .. math:: \\text{ci}_d = d \\pm \\text{crit} \\cdot \\text{SE}

    where :math:`\\text{crit}` is the critical value of the T distribution
    corresponding to the desired confidence level.

    References
    ----------
    * https://en.wikipedia.org/wiki/Fisher_transformation

    * Hedges, L., and Ingram Olkin. "Statistical models for meta-analysis."
      (1985).

    * http://www.leeds.ac.uk/educol/documents/00002182.htm

    * https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5133225/

    Examples
    --------
    1. Confidence interval of a Pearson correlation coefficient

    >>> import pingouin as pg
    >>> x = [3, 4, 6, 7, 5, 6, 7, 3, 5, 4, 2]
    >>> y = [4, 6, 6, 7, 6, 5, 5, 2, 3, 4, 1]
    >>> nx, ny = len(x), len(y)
    >>> stat = pg.compute_effsize(x, y, eftype='r')
    >>> ci = pg.compute_esci(stat=stat, nx=nx, ny=ny, eftype='r')
    >>> print(round(stat, 4), ci)
    0.7468 [0.27 0.93]

    2. Confidence interval of a Cohen d

    >>> stat = pg.compute_effsize(x, y, eftype='cohen')
    >>> ci = pg.compute_esci(stat, nx=nx, ny=ny, eftype='cohen', decimals=3)
    >>> print(round(stat, 4), ci)
    0.1538 [-0.737  1.045]
    """
    from scipy.stats import norm, t
    assert eftype.lower() in ['r', 'pearson', 'spearman', 'cohen',
                              'd', 'g', 'hedges']
    assert stat is not None and nx is not None
    assert isinstance(confidence, float)
    assert 0 < confidence < 1, 'confidence must be between 0 and 1.'

    if eftype.lower() in ['r', 'pearson', 'spearman']:
        z = np.arctanh(stat)  # R-to-z transform
        se = 1 / np.sqrt(nx - 3)
        crit = np.abs(norm.ppf((1 - confidence) / 2))
        ci_z = np.array([z - crit * se, z + crit * se])
        ci = np.tanh(ci_z)  # Transform back to r
    else:
        # Cohen d. Results are different than JASP which uses a non-central T
        # distribution. See github.com/jasp-stats/jasp-issues/issues/525
        if ny == 1 or paired:
            # One-sample or paired. Results vary slightly from the cohen.d R
            # function which uses instead:
            # >>> sqrt((n / (n / 2)^2) + .5*(dd^2 / n)) -- one-sample
            # >>> sqrt( (1/n1 + dd^2/(2*n1))*(2-2*r)); -- paired
            # where r is the correlation between samples
            # https://github.com/mtorchiano/effsize/blob/master/R/CohenD.R
            # However, Pingouin uses the formulas on www.real-statistics.com
            se = np.sqrt(1 / nx + stat ** 2 / (2 * nx))
            dof = nx - 1
        else:
            # Independent two-samples: give same results as R:
            # >>> cohen.d(..., paired = FALSE, noncentral=FALSE)
            se = np.sqrt(((nx + ny) / (nx * ny)) + (stat ** 2) / (2 * (nx + ny)))
            dof = nx + ny - 2
        crit = np.abs(t.ppf((1 - confidence) / 2, dof))
        ci = np.array([stat - crit * se, stat + crit * se])
    return np.round(ci, decimals)


def power_corr(r=None, n=None, power=None, alpha=0.05, tail='two-sided'):
    """
    Evaluate power, sample size, correlation coefficient or
    significance level of a correlation test.

    Parameters
    ----------
    r : float
        Correlation coefficient.
    n : int
        Number of observations (sample size).
    power : float
        Test power (= 1 - type II error).
    alpha : float
        Significance level (type I error probability).
        The default is 0.05.
    tail : str
        Indicates whether the test is `'two-sided'` or `'one-sided'`.

    Notes
    -----
    Exactly ONE of the parameters ``r``, ``n``, ``power`` and ``alpha`` must
    be passed as None, and that parameter is determined from the others.

    Notice that ``alpha`` has a default value of 0.05 so None must be
    explicitly passed if you want to compute it.

    :py:func:`scipy.optimize.brenth` is used to solve power equations for other
    variables (i.e. sample size, effect size, or significance level). If the
    solving fails, a nan value is returned.

    This function is a Python adaptation of the `pwr.r.test`
    function implemented in the
    `pwr <https://cran.r-project.org/web/packages/pwr/pwr.pdf>`_ R package.

    Examples
    --------
    1. Compute achieved power given ``r``, ``n`` and ``alpha``

    >>> from pingouin import power_corr
    >>> print('power: %.4f' % power_corr(r=0.5, n=20))
    power: 0.6379

    2. Compute required sample size given ``r``, ``power`` and ``alpha``

    >>> print('n: %.4f' % power_corr(r=0.5, power=0.80,
    ...                                tail='one-sided'))
    n: 22.6091

    3. Compute achieved ``r`` given ``n``, ``power`` and ``alpha`` level

    >>> print('r: %.4f' % power_corr(n=20, power=0.80, alpha=0.05))
    r: 0.5822

    4. Compute achieved alpha level given ``r``, ``n`` and ``power``

    >>> print('alpha: %.4f' % power_corr(r=0.5, n=20, power=0.80,
    ...                                    alpha=None))
    alpha: 0.1377
    """
    # Check the number of arguments that are None
    n_none = sum([v is None for v in [r, n, power, alpha]])
    if n_none != 1:
        raise ValueError('Exactly one of n, r, power, and alpha must be None')

    # Safety checks
    if r is not None:
        assert -1 <= r <= 1
        r = abs(r)
    if alpha is not None:
        assert 0 < alpha <= 1
    if power is not None:
        assert 0 < power <= 1
    if n is not None:
        if n <= 4:
            warnings.warn("Sample size is too small to estimate power "
                          "(n <= 4). Returning NaN.")
            return np.nan

    # Define main function
    if tail == 'two-sided':

        def func(r, n, power, alpha):
            dof = n - 2
            ttt = stats.t.ppf(1 - alpha / 2, dof)
            rc = np.sqrt(ttt ** 2 / (ttt ** 2 + dof))
            zr = np.arctanh(r) + r / (2 * (n - 1))
            zrc = np.arctanh(rc)
            power = stats.norm.cdf((zr - zrc) * np.sqrt(n - 3)) + \
                    stats.norm.cdf((-zr - zrc) * np.sqrt(n - 3))
            return power

    else:

        def func(r, n, power, alpha):
            dof = n - 2
            ttt = stats.t.ppf(1 - alpha, dof)
            rc = np.sqrt(ttt ** 2 / (ttt ** 2 + dof))
            zr = np.arctanh(r) + r / (2 * (n - 1))
            zrc = np.arctanh(rc)
            power = stats.norm.cdf((zr - zrc) * np.sqrt(n - 3))
            return power

    # Evaluate missing variable
    if power is None and n is not None and r is not None:
        # Compute achieved power given r, n and alpha
        return func(r, n, power=None, alpha=alpha)

    elif n is None and power is not None and r is not None:
        # Compute required sample size given r, power and alpha

        def _eval_n(n, r, power, alpha):
            return func(r, n, power, alpha) - power

        try:
            return brenth(_eval_n, 4 + 1e-10, 1e+09, args=(r, power, alpha))
        except ValueError:  # pragma: no cover
            return np.nan

    elif r is None and power is not None and n is not None:
        # Compute achieved r given sample size, power and alpha level

        def _eval_r(r, n, power, alpha):
            return func(r, n, power, alpha) - power

        try:
            return brenth(_eval_r, 1e-10, 1 - 1e-10, args=(n, power, alpha))
        except ValueError:  # pragma: no cover
            return np.nan

    else:
        # Compute achieved alpha (significance) level given r, n and power

        def _eval_alpha(alpha, r, n, power):
            return func(r, n, power, alpha) - power

        try:
            return brenth(_eval_alpha, 1e-10, 1 - 1e-10, args=(r, n, power))
        except ValueError:  # pragma: no cover
            return np.nan


def _postprocess_dataframe(df):
    """Apply some post-processing to an ouput dataframe (e.g. rounding).

    Whether and how rounding is applied is governed by options specified in
    `pingouin.options`. The default rounding (number of decimals) is
    determined by `pingouin.options['round']`. You can specify rounding for a
    given column name by the option `'round.column.<colname>'`, e.g.
    `'round.column.CI95%'`. Analogously, `'round.row.<rowname>'` also works
    (where `rowname`) refers to the pandas index), as well as
    `'round.cell.[<rolname>]x[<colname]'`. A cell-based option is used,
    if available; if not, a column-based option is used, if
    available; if not, a row-based option is used, if available; if not,
    the default is used. (Default `pingouin.options['round'] = None`,
    i.e. no rounding is applied.)

    If a round option is `callable` instead of `int`, then it will be called,
    and the return value stored in the cell.

    Post-processing is applied on a copy of the DataFrame, leaving the
    original DataFrame untouched.

    This is an internal function (no public API).

    Parameters
    ----------
    df : :py:class:`pandas.DataFrame`
        Dataframe to apply post-processing to (e.g. ANOVA summary)

    Returns
    ----------
    df : :py:class:`pandas.DataFrame`
        Dataframe with post-processing applied
    """
    df = df.copy()
    for row, col in it.product(df.index, df.columns):
        round_option = _get_round_setting_for(row, col)
        if round_option is None:
            continue
        if callable(round_option):
            newval = round_option(df.at[row, col])
            # ensure that dtype changes are processed
            df[col] = df[col].astype(type(newval))
            df.at[row, col] = newval
            continue
        if isinstance(df.at[row, col], bool):
            # No rounding if value is a boolean
            continue
        is_number = isinstance(df.at[row, col], numbers.Number)
        is_array = isinstance(df.at[row, col], np.ndarray)
        if not any([is_number, is_array]):
            # No rounding if value is not a Number or an array
            continue
        if is_array:
            is_float_array = issubclass(df.at[row, col].dtype.type,
                                        np.floating)
            if not is_float_array:
                # No rounding if value is not a float array
                continue
        df.at[row, col] = np.round(df.at[row, col], decimals=round_option)
    return df


def _format_bf(bf, precision=3, trim='0'):
    """Format BF10 to floating point or scientific notation.
    """
    if type(bf) == str:
        return bf
    if bf >= 1e4 or bf <= 1e-4:
        out = np.format_float_scientific(bf, precision=precision, trim=trim)
    else:
        out = np.format_float_positional(bf, precision=precision, trim=trim)
    return out


def _get_round_setting_for(row, col):
    options = {
        'round': None,
        'round.column.CI95%': 2,
        'round.column.BF10': _format_bf
    }
    keys_to_check = (
        'round.cell.[{}]x[{}]'.format(row, col),
        'round.column.{}'.format(col), 'round.row.{}'.format(row))
    for key in keys_to_check:
        try:
            return options[key]
        except KeyError:
            pass
    return options['round']

def _is_sklearn_installed(raise_error=False):
    """Check if sklearn is installed."""
    try:
        import sklearn  # noqa
        is_installed = True
    except IOError:  # pragma: no cover
        is_installed = False
    # Raise error (if needed) :
    if raise_error and not is_installed:  # pragma: no cover
        raise IOError("sklearn needs to be installed. Please use `pip "
                      "install scikit-learn`.")
    return is_installed
