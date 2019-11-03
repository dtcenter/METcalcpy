import warnings
import numpy as np
import pandas as pd

# CTC stat calculations
from scipy.special import lambertw


def calculate_baser(input_data, columns_names):
    warnings.filterwarnings('error')
    # if fy_oy is None or None in input_data.fn_oy.values or None in input_data.total.values:
    #     return None
    try:
        result = (sum(input_data[:, np.where(columns_names == 'fy_oy')[0][0]])
                  + sum(input_data[:, np.where(columns_names == 'fn_oy')[0][0]])) \
                 / sum(input_data[:, np.where(columns_names == 'total')[0][0]])
    except Warning:
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_acc(input_data, columns_names):
    warnings.filterwarnings('error')
    # if None in input_data.fy_oy.values or None in input_data.fn_on.values or None in input_data.total.values:
    #     return None
    try:
        result = (sum_column_data_by_name(input_data, columns_names, 'fy_oy')
                  + sum_column_data_by_name(input_data, columns_names, 'fn_on')) \
                 / sum_column_data_by_name(input_data, columns_names, 'total')
    except Warning:
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_fbias(input_data, columns_names):
    # if (is.na(d$fy_oy) | | is.na(d$fn_oy) ){
    oy = sum_column_data_by_name(input_data, columns_names, 'fy_oy') \
         + sum_column_data_by_name(input_data, columns_names, 'fn_on')
    if oy == 0:
        return None
    oyn = sum_column_data_by_name(input_data, columns_names, 'fy_oy') \
          + sum_column_data_by_name(input_data, columns_names, 'fy_on')
    return oyn / oy


def calculate_fmean(input_data, columns_names):
    # if (is.na(d$fy_oy) | | is.na(d$fn_oy) | | is.na(d$total) ){
    total = sum_column_data_by_name(input_data, columns_names, 'total')
    if total == 0:
        return None
    oyn = sum_column_data_by_name(input_data, columns_names, 'fy_oy') \
          + sum_column_data_by_name(input_data, columns_names, 'fy_on')
    return oyn / total


def calculate_pody(input_data, columns_names):
    # if(is.na(d$fy_oy) || is.na(d$fn_oy) || is.na(d$total) ){
    oy = sum_column_data_by_name(input_data, columns_names, 'fy_oy') \
         + sum_column_data_by_name(input_data, columns_names, 'fn_oy')
    if oy == 0:
        return None
    return sum_column_data_by_name(input_data, columns_names, 'fy_oy') / oy


def calculate_pofd(input_data, columns_names):
    # if(is.na(d$fy_oy) || is.na(d$fn_oy)  ){
    oy = sum_column_data_by_name(input_data, columns_names, 'fy_on') \
         + sum_column_data_by_name(input_data, columns_names, 'fn_on')
    if oy == 0:
        return None
    return sum_column_data_by_name(input_data, columns_names, 'fy_on') / oy


def calculate_podn(input_data, columns_names):
    # if(is.na(d$fy_oy) || is.na(d$fn_oy)  ){
    oy = sum_column_data_by_name(input_data, columns_names, 'fy_on') \
         + sum_column_data_by_name(input_data, columns_names, 'fn_on')
    if oy == 0:
        return None
    return sum_column_data_by_name(input_data, columns_names, 'fn_on') / oy


def calculate_far(input_data, columns_names):
    # if(is.na(d$fy_oy) || is.na(d$fn_oy)  ){
    oy = sum_column_data_by_name(input_data, columns_names, 'fy_oy') \
         + sum_column_data_by_name(input_data, columns_names, 'fy_on')
    if oy == 0:
        return None
    return sum_column_data_by_name(input_data, columns_names, 'fy_on') / oy


def calculate_csi(input_data, columns_names):
    # if(is.na(d$fy_oy) || is.na(d$fn_oy) ){
    oy = sum_column_data_by_name(input_data, columns_names, 'fy_oy') \
         + sum_column_data_by_name(input_data, columns_names, 'fy_on') \
         + sum_column_data_by_name(input_data, columns_names, 'fn_oy')
    if oy == 0:
        return None
    return sum_column_data_by_name(input_data, columns_names, 'fy_oy') / oy


def calculate_gss(input_data, columns_names):
    # if(is.na(d$fy_oy) || is.na(d$fn_oy) || is.na(d$total) ){
    total = sum_column_data_by_name(input_data, columns_names, 'total')
    if total == 0:
        return None
    dbl_c = ((sum_column_data_by_name(input_data, columns_names, 'fy_oy')
              + sum_column_data_by_name(input_data, columns_names, 'fy_on')) / total) \
            * (sum_column_data_by_name(input_data, columns_names, 'fy_oy')
               + sum_column_data_by_name(input_data, columns_names, 'fn_oy'))

    gss = ((sum_column_data_by_name(input_data, columns_names, 'fy_oy') - dbl_c)
           / (sum_column_data_by_name(input_data, columns_names, 'fy_oy')
              + sum_column_data_by_name(input_data, columns_names, 'fy_on')
              + sum_column_data_by_name(input_data, columns_names, 'fn_oy') - dbl_c))
    return gss


def calculate_hk(input_data, columns_names):
    pody = calculate_pody(input_data, columns_names)
    pofd = calculate_pofd(input_data, columns_names)
    if pody is None or pofd is None:
        return None
    return pody - pofd


def calculate_hss(input_data, columns_names):
    # if(is.na(d$fy_oy) || is.na(d$fn_oy) || is.na(d$total) ){
    total = sum_column_data_by_name(input_data, columns_names, 'total')
    if total == 0:
        return None
    dbl_c = ((sum_column_data_by_name(input_data, columns_names, 'fy_oy')
              + sum_column_data_by_name(input_data, columns_names, 'fy_on')) / total) \
            * (sum_column_data_by_name(input_data, columns_names, 'fy_oy')
               + sum_column_data_by_name(input_data, columns_names, 'fn_oy'))
    hss = ((sum_column_data_by_name(input_data, columns_names, 'fy_oy')
            + sum_column_data_by_name(input_data, columns_names, 'fn_on') - dbl_c)
           / (total - dbl_c))
    return hss


def calculate_odds(input_data, columns_names):
    pody = calculate_pody(input_data, columns_names)
    pofd = calculate_pofd(input_data, columns_names)
    if pody is None or pofd is None:
        return None
    return (pody * (1 - pofd)) / (pofd * (1 - pody))


def calculate_lodds(input_data, columns_names):
    # if(is.na(d$fy_oy) || is.na(d$fy_on) || is.na(d$fn_oy) || is.na(d$fn_on)) {
    fy_oy = sum_column_data_by_name(input_data, columns_names, 'fy_oy')
    fy_on = sum_column_data_by_name(input_data, columns_names, 'fy_on')
    fn_oy = sum_column_data_by_name(input_data, columns_names, 'fn_oy')
    fn_on = sum_column_data_by_name(input_data, columns_names, 'fn_on')
    if fy_oy is None or fy_on is None or fn_oy is None or fn_on is None:
        return None
    v = np.log(fy_oy) + np.log(fn_on) - np.log(fy_on) - np.log(fn_oy)
    return v


def calculate_baggs(input_data, columns_names):
    fy_oy = sum_column_data_by_name(input_data, columns_names, 'fy_oy')
    fn_oy = sum_column_data_by_name(input_data, columns_names, 'fn_oy')
    total = sum_column_data_by_name(input_data, columns_names, 'total')
    fy_on = sum_column_data_by_name(input_data, columns_names, 'fy_on')
    if fy_oy is None or fn_oy is None or fy_on is None or total is None:
        return None
    if fy_oy == 0 or fn_oy == 0 or total == 0:
        return None
    dbl_o = fy_oy + fn_oy
    dbl_lf = np.log(dbl_o / fn_oy)
    lambert = lambertw(dbl_o / fy_on * dbl_lf).real
    dbl_ha = dbl_o - (fy_on / dbl_lf) * lambert
    return (dbl_ha - (dbl_o ** 2 / total)) / (2 * dbl_o - dbl_ha - (dbl_o ** 2 / total))


def calculate_eclv(input_data, columns_names):
    # Implemente R version that returns an array instread of the single value
    # IT WILL NOT WORK - NEED TO CONSULT WITH STATISTICIAN
    # Build list of X-axis points between 0 and 1
    cl_step = 0.05
    cl_pts = np.arange(start=cl_step, stop=1, step=cl_step)
    fy_oy = sum_column_data_by_name(input_data, columns_names, 'fy_oy')
    fy_on = sum_column_data_by_name(input_data, columns_names, 'fy_on')
    fn_oy = sum_column_data_by_name(input_data, columns_names, 'fn_oy')
    fn_on = sum_column_data_by_name(input_data, columns_names, 'fn_on')
    eclv = calculate_economic_value(np.array([fy_oy, fy_on, fn_oy, fn_on]), cl_pts)
    common_cases_ind = pd.Series(eclv['cl']).isin(cl_pts)
    v = eclv['V'][common_cases_ind]
    return v


def calculate_economic_value(obs, cl=np.arange(start=0.05, stop=0.95, step=0.05)):
    # Calculates the economic value of a forecast based on a cost/loss ratio.
    # implementation based on Rscript value() function
    # obs An array vector of a contingency table summary of values in the form
    # c(n11, n01, n10, n00) where in nab a = obs, b = forecast.
    # cl Cost loss ratio. The relative value of being unprepared and taking a loss to that
    # of un-necessarily preparing. For example, cl = 0.1 indicates it would cost $ 1
    # to prevent a $10 loss. This defaults to the sequence 0.05 to 0.95 by 0.05.
    # vmax Maximum value
    # v array of values for each cl value
    # f Conditional false alarm rate.
    # h Conditional hit rate
    # cl array of cost loss ratios.
    # s Base rate

    # Assume data entered as c(n11, n01, n10, n00) Obs*Forecast
    if len(obs) == 4:
        n = sum(obs)
        a = obs[0]
        b = obs[1]
        c = obs[2]
        d = obs[3]
        f = b / (b + d)
        h = a / (a + c)
        s = (a + c) / n

        cl_local = np.append(cl, s)
        cl_local.sort()

        v_1 = (1 - f) - s / (1 - s) * (1 - cl_local) / cl_local * (1 - h)
        v_2 = h - (1 - s) / s * cl_local / (1 - cl_local) * f
        v = np.zeros(len(cl_local))

        indexes = cl_local < s
        v[indexes] = v_1[indexes]
        indexes = cl_local >= s
        v[indexes] = v_2[indexes]

        v_max = h - f
        result = {'vmax': v_max, 'V': v, 'F': f, 'H': h, 'cl': cl_local, 's': s, 'n': n}
        return result
    else:
        return None

    # SL1L2 stat calculations


def calculate_fbar(input_data, columns_names):
    total = sum_column_data_by_name(input_data, columns_names, 'total')
    return sum_column_data_by_name(input_data, columns_names, 'fbar') / total


def calculate_obar(input_data, columns_names):
    total = sum_column_data_by_name(input_data, columns_names, 'total')
    return sum_column_data_by_name(input_data, columns_names, 'obar') / total


def calculate_fstdev(input_data, columns_names):
    # total_column_data = get_column_data_by_name(input_data, columns_names, 'total')
    # total = sum(total_column_data)
    # fbar = sum(get_column_data_by_name(input_data, columns_names, 'fbar') * total_column_data) / total
    # obar = sum(get_column_data_by_name(input_data, columns_names, 'obar') * total_column_data) / total
    # fobar = sum(get_column_data_by_name(input_data, columns_names, 'fobar') * total_column_data) / total
    # ffbar = sum(get_column_data_by_name(input_data, columns_names, 'ffbar') * total_column_data) / total
    # oobar = sum(get_column_data_by_name(input_data, columns_names, 'oobar') * total_column_data) / total
    # mae = sum(get_column_data_by_name(input_data, columns_names, 'mae') * total_column_data) / total

    total = sum_column_data_by_name(input_data, columns_names, 'total')
    fbar = sum_column_data_by_name(input_data, columns_names, 'fbar') / total
    ffbar = sum_column_data_by_name(input_data, columns_names, 'ffbar') / total
    sum_ = fbar * total
    sum_sq = ffbar * total
    n = total
    return calculate_stddev(sum_, sum_sq, n)


def calculate_ostdev(input_data, columns_names):
    total = sum_column_data_by_name(input_data, columns_names, 'total')
    obar = sum_column_data_by_name(input_data, columns_names, 'obar') / total
    oobar = sum_column_data_by_name(input_data, columns_names, 'oobar') / total
    sum_ = obar * total
    sum_sq = oobar * total
    n = total
    return calculate_stddev(sum_, sum_sq, n)


def calculate_fobar(input_data, columns_names):
    total = sum_column_data_by_name(input_data, columns_names, 'total')
    return sum_column_data_by_name(input_data, columns_names, 'fobar') / total


def calculate_ffbar(input_data, columns_names):
    total = sum_column_data_by_name(input_data, columns_names, 'total')
    return sum_column_data_by_name(input_data, columns_names, 'ffbar') / total


def calculate_oobar(input_data, columns_names):
    total = sum_column_data_by_name(input_data, columns_names, 'total')
    return sum_column_data_by_name(input_data, columns_names, 'oobar') / total


def calculate_mae(input_data, columns_names):
    total = sum_column_data_by_name(input_data, columns_names, 'total')
    return sum_column_data_by_name(input_data, columns_names, 'mae') / total


def calculate_mbias(input_data, columns_names):
    total = sum_column_data_by_name(input_data, columns_names, 'total')
    obar = sum_column_data_by_name(input_data, columns_names, 'obar') / total
    if obar == 0:
        return None
    fbar = sum_column_data_by_name(input_data, columns_names, 'fbar') / total
    return fbar / obar


def calculate_pr_corr(input_data, columns_names):
    # f( is.na(d$total) || is.na(d$ffbar) || is.na(d$fbar) || is.na(d$oobar) || is.na(d$obar) ){
    total = sum_column_data_by_name(input_data, columns_names, 'total')
    ffbar = sum_column_data_by_name(input_data, columns_names, 'ffbar') / total
    fbar = sum_column_data_by_name(input_data, columns_names, 'fbar') / total
    oobar = sum_column_data_by_name(input_data, columns_names, 'oobar') / total
    obar = sum_column_data_by_name(input_data, columns_names, 'obar') / total
    fobar = sum_column_data_by_name(input_data, columns_names, 'fobar') / total
    v = (total ** 2 * ffbar - total ** 2 * fbar ** 2) * (total ** 2 * oobar - total ** 2 * obar ** 2)
    pr_corr = (total ** 2 * fobar - total ** 2 * fbar * obar) / np.sqrt(v)
    if 0 >= v or 1 < pr_corr:
        return None
    return pr_corr


def calculate_anom_corr(input_data, columns_names):
    # if( is.na(d$total) || is.na(d$ffbar) || is.na(d$fbar) || is.na(d$oobar) || is.na(d$obar) ){
    total = sum_column_data_by_name(input_data, columns_names, 'total')
    ffbar = sum_column_data_by_name(input_data, columns_names, 'ffbar') / total
    fbar = sum_column_data_by_name(input_data, columns_names, 'fbar') / total
    oobar = sum_column_data_by_name(input_data, columns_names, 'oobar') / total
    obar = sum_column_data_by_name(input_data, columns_names, 'obar') / total
    fobar = sum_column_data_by_name(input_data, columns_names, 'fobar') / total
    v = (total ** 2 * ffbar - total ** 2 * fbar ** 2) * (total ** 2 * oobar - total ** 2 * obar ** 2)
    if 0 >= v:
        return None
    anom_corr = (total ** 2 * fobar - total ** 2 * fbar * obar) / np.sqrt(v)
    if 1 < anom_corr:
        return None
    return anom_corr


def calculate_rmsfa(input_data, columns_names):
    total = sum_column_data_by_name(input_data, columns_names, 'total')
    ffbar = sum_column_data_by_name(input_data, columns_names, 'ffbar') / total
    if ffbar is None or 0 > ffbar:
        return None
    return np.sqrt(ffbar)


def calculate_rmsoa(input_data, columns_names):
    total = sum_column_data_by_name(input_data, columns_names, 'total')
    oobar = sum_column_data_by_name(input_data, columns_names, 'oobar') / total
    if oobar is None or 0 > oobar:
        return None
    return np.sqrt(oobar)


def calculate_me(input_data, columns_names):
    total = sum_column_data_by_name(input_data, columns_names, 'total')
    fbar = sum_column_data_by_name(input_data, columns_names, 'fbar') / total
    obar = sum_column_data_by_name(input_data, columns_names, 'obar') / total
    return fbar - obar


def calculate_me2(input_data, columns_names):
    me = calculate_me(input_data, columns_names)
    return me ** 2


def calculate_mse(input_data, columns_names):
    total = sum_column_data_by_name(input_data, columns_names, 'total')
    ffbar = sum_column_data_by_name(input_data, columns_names, 'ffbar') / total
    oobar = sum_column_data_by_name(input_data, columns_names, 'oobar') / total
    fobar = sum_column_data_by_name(input_data, columns_names, 'fobar') / total
    return ffbar + oobar - 2 * fobar


def calculate_msess(input_data, columns_names):
    ostdev = calculate_ostdev(input_data, columns_names)
    mse = calculate_mse(input_data, columns_names)
    return 1.0 - mse / ostdev ** 2


def calculate_rmse(input_data, columns_names):
    return np.sqrt(calculate_mse(input_data, columns_names))


def calculate_estdev(input_data, columns_names):
    total = sum_column_data_by_name(input_data, columns_names, 'total')
    me = calculate_me(input_data, columns_names)
    mse = calculate_mse(input_data, columns_names)
    return calculate_stddev(me * total, mse * total, total)


def calculate_bcmse(input_data, columns_names):
    mse = calculate_mse(input_data, columns_names)
    me = calculate_me(input_data, columns_names)
    return mse - me ** 2


def calculate_bcrmse(input_data, columns_names):
    return np.sqrt(calculate_bcmse(input_data, columns_names))


def calculate_stddev(sum, sum_sq, n):
    if 1 > n:
        return None
    v = (sum_sq - sum * sum / n) / (n - 1)
    if 0 > v:
        return None
    else:
        return np.sqrt(v)


def sum_column_data_by_name(input_data, columns, column_name):
    index_array = np.where(columns == column_name)[0]
    if len(index_array) == 0:
        return None
    data_array = input_data[:, index_array[0]]
    if None in data_array:
        return None
    return sum(data_array)


def get_column_data_by_name(input_data, columns, column_name):
    index = get_column_index_by_name(columns, column_name)
    if index is not None:
        return input_data[:, index]
    else:
        return None


def get_column_index_by_name(columns, column_name):
    index_array = np.where(columns == column_name)[0]
    if len(index_array) == 0:
        return None
    return index_array[0]


if __name__ == "__main__":
    obs = np.array([24617, 76390, 45157, 2099486])
    last_value = 0.95 + 0.05
    cl = np.arange(start=0.05, stop=last_value, step=0.05)
    calculate_economic_value(obs, cl)
