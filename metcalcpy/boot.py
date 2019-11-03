import time
import itertools
import numpy as np
import pandas as pd
import bootstrapped.bootstrap
from metcalcpy import event_equalize
from metcalcpy.bootstrap_custom import BootstrapDistributionResults, bootstrap_and_value
from metcalcpy.util.statistics import *
from metcalcpy.util.utils import is_string_integer, get_derived_curve_name, unique, \
    calc_derived_curve_value, intersection, is_derived_series

EXEMPTED_VARS = ['SSVAR_Spread', 'SSVAR_RMSE']
LINETYPE_TO_FIELDS = {
    'sl1l2': ['fbar', 'obar', 'fobar', 'ffbar', 'oobar', 'mae'],
    'ctc': ['fy_oy', 'fy_on', 'fn_oy', 'fn_on']
}

COLUMN_NAMES = None
STATISTIC = None
diff_name_to_values = {}


class DerivedCurveComponent:
    def __init__(self, first_component, second_component, derived_operation):
        self.first_component = first_component
        self.second_component = second_component
        self.derived_operation = derived_operation


def calc_stats(values):
    '''Calculate the statistic of values for each bootstrap sample
    Args:
        values: a np.array of values we want to calculate the statistic on
            This is actually a 2d array (matrix) of values. Each row represents
            a bootstrap resample simulation that we wish to aggregate across.
    '''
    if values is not None and values.ndim == 2:
        stat_values = [globals()['calculate_{}'.format(STATISTIC)](values, COLUMN_NAMES)]
    elif values is not None and values.ndim == 3:
        stat_values = []
        for row in values:
            stat_values.append([globals()['calculate_{}'.format(STATISTIC)](row, COLUMN_NAMES)])

        # pool = mp.Pool(mp.cpu_count())
        # stat_values = pool.map(partial(globals()['calculate_{}'.format(stat)],
        # columns_names=columns_names), [row for row in data_for_stats])
        # pool.close()
        # pool.join()

    else:
        raise KeyError("can't calculate statistic")
    return stat_values


def prepare_sl1l2_data(data_for_prepare):
    for column in LINETYPE_TO_FIELDS['sl1l2']:
        data_for_prepare[column] \
            = data_for_prepare[column].values * data_for_prepare['total'].values


def prepare_derived_data(data_1, data_2, operation, line_type):
    for column in LINETYPE_TO_FIELDS[line_type]:
        column_index = get_column_index_by_name(COLUMN_NAMES, column)
        data_1[:, column_index] = calc_derived_curve_value(
            data_1[:, column_index],
            data_2[:, column_index],
            operation)


def prepare_ctc_data(data_for_prepare):
    pass


def perform_event_equalization(input_data_for_ee, indy_vals, input_parameters):
    fix_vals = []
    output_ee_data = pd.DataFrame()

    fcst_var_val_for_ee = input_parameters['fcst_var_val']
    fixed_vars_vals_for_ee = input_parameters['fixed_vars_vals_input']
    series_val_for_ee = input_parameters['series_val']
    indy_var = input_parameters['indy_var']

    is_multi = False
    # for SSVAR use equalization of mulitple events
    if input_parameters['line_type'] == "ssvar":
        is_multi = True

    # list all fixed variables
    if fixed_vars_vals_for_ee:
        for value in fixed_vars_vals_for_ee.values():
            fix_vals.append(list(value.values()))
    # permute fix vals
    fix_vals_permuted = list(itertools.chain.from_iterable(fix_vals))
    # perform EE for each forecast variable
    for fcst_var, fcst_var_stats in fcst_var_val_for_ee.items():
        for fcst_var_stat in fcst_var_stats:
            for series_var, series_var_vals in series_val_for_ee.items():
                # ungroup series value
                series_var_vals_no_group = []
                for val in series_var_vals:
                    split_val = val.split(',')
                    series_var_vals_no_group.extend(split_val)

                series_data_for_ee = input_data_for_ee[
                    (input_data_for_ee['fcst_var'] == fcst_var)
                    & (input_data_for_ee["stat_name"] == fcst_var_stat)
                    & (input_data_for_ee[series_var].isin(series_var_vals_no_group))
                    ]
                start = time.time()
                series_data_after_ee = \
                    event_equalize(series_data_for_ee, indy_var, indy_vals, series_val_for_ee,
                                   list(fixed_vars_vals_for_ee.keys()),
                                   fix_vals_permuted, True, is_multi)
                end = time.time()
                print("one EE:" + str(end - start))

                # append EE data to result
                if output_ee_data.empty:
                    output_ee_data = series_data_after_ee
                else:
                    output_ee_data.append(series_data_after_ee)
    return output_ee_data


def get_bootsrtapped_stats_for_derived(series_subset, derived_curve_component,
                                       distributions, input_parameters):
    permute_for_first_series = derived_curve_component.first_component.copy()
    permute_for_first_series.extend(series_subset)
    permute_for_first_series = unique(permute_for_first_series)

    permute_for_second_series = derived_curve_component.second_component.copy()
    permute_for_second_series.extend(series_subset)
    permute_for_second_series = unique(permute_for_second_series)

    ds_1 = None
    ds_2 = None
    for series_to_distrib_key in distributions.keys():
        if all(elem in permute_for_first_series for elem in series_to_distrib_key):
            ds_1 = distributions[series_to_distrib_key]
        if all(elem in permute_for_second_series for elem in series_to_distrib_key):
            ds_2 = distributions[series_to_distrib_key]
        if ds_1 is not None and ds_2 is not None:
            break

    validate_series_cases_for_derived_operation(ds_1.values, input_parameters['list_stat'])
    validate_series_cases_for_derived_operation(ds_2.values, input_parameters['list_stat'])

    prepare_derived_data(ds_1.values, ds_2.values, derived_curve_component.derived_operation,
                         input_parameters['line_type'])
    num_iterations = input_parameters['num_iterations']
    if num_iterations == 1:
        stat_val = calc_stats(ds_1.values)[0]
        results = BootstrapDistributionResults(lower_bound=None,
                                               value=stat_val,
                                               upper_bound=None)
    else:
        try:
            results = bootstrap_and_value(
                ds_1.values,
                stat_func=calc_stats,
                num_iterations=input_parameters['num_iterations'],
                num_threads=input_parameters['num_threads'],
                ci_method=input_parameters['method'],
                alpha=input_parameters['alpha'])


        except KeyError as err:
            results = bootstrapped.bootstrap.BootstrapResults(None, None, None)
            print(err)
    return results


def get_bootsrtapped_stats(series_data, input_parameters):
    # can't calculate differences if  multiple values for one valid date/fcst_lead

    series_data = sort_data(series_data)

    globals()['prepare_{}_data'.format(input_parameters['line_type'])](series_data)
    data = series_data.to_numpy()

    num_iterations = input_parameters['num_iterations']
    if num_iterations == 1:
        stat_val = calc_stats(data)[0]
        results = BootstrapDistributionResults(lower_bound=None,
                                               value=stat_val,
                                               upper_bound=None)
        results.set_original_values(series_data)
    else:
        try:
            results = bootstrap_and_value(
                data,
                stat_func=calc_stats,
                num_iterations=num_iterations,
                num_threads=input_parameters['num_threads'],
                ci_method=input_parameters['method'])


        except KeyError as err:
            results = bootstrapped.bootstrap.BootstrapResults(None, None, None)
            print(err)
    return results


def validate_series_cases_for_derived_operation(series_data, list_stat):
    fcst_lead_index = np.where(COLUMN_NAMES == 'fcst_lead')[0][0]
    stat_name_index = np.where(COLUMN_NAMES == 'stat_name')[0][0]
    if "fcst_valid_beg" in COLUMN_NAMES:
        fcst_valid_ind = np.where(COLUMN_NAMES == 'fcst_valid_beg')[0][0]
    elif "fcst_valid" in COLUMN_NAMES:
        fcst_valid_ind = np.where(COLUMN_NAMES == 'fcst_valid')[0][0]
    elif "fcst_init_beg" in COLUMN_NAMES:
        fcst_valid_ind = \
            np.where(COLUMN_NAMES == 'fcst_init_beg')[0][0]
    else:
        fcst_valid_ind = \
            np.where(COLUMN_NAMES == 'fcst_init')[0][0]

    series_data_frame = \
        pd.DataFrame(
            series_data[:, [fcst_valid_ind, fcst_lead_index, stat_name_index]])
    unique_dates = len(series_data_frame.drop_duplicates().values)
    ind = \
        np.lexsort(
            (series_data[:, stat_name_index],
             series_data[:, fcst_lead_index], series_data[:, fcst_valid_ind]))
    series_data = series_data[ind]

    if len(series_data) != unique_dates \
            and list_stat not in EXEMPTED_VARS:
        raise NameError("Derived curve can't be calculated."
                        " Multiple values for one valid date/fcst_lead")


def sort_data(series_data):
    fields = series_data.keys()
    if "fcst_valid_beg" in fields:
        series_data = series_data.sort_values(by=["fcst_valid_beg", "fcst_lead", "stat_name"])
    elif "fcst_valid" in fields:
        series_data = series_data.sort_values(by=["fcst_valid", "fcst_lead", "stat_name"])
    elif "fcst_init_beg" in fields:
        series_data = series_data.sort_values(by=["fcst_init_beg", "fcst_lead", "stat_name"])
    else:
        series_data = series_data.sort_values(by=["fcst_init", "fcst_lead", "stat_name"])
    return series_data


def init_out_frame(list_static_val, series_fields, series):
    result = pd.DataFrame()
    row_number = len(series)
    for static_var in list_static_val:
        result[static_var] = [list_static_val[static_var]] * row_number
    for field_ind, field in enumerate(series_fields):
        result[field] = [row[field_ind] for row in series]
    result['stat_value'] = [None] * row_number
    result['stat_bcl'] = [None] * row_number
    result['stat_bcu'] = [None] * row_number
    result['nstats'] = [None] * row_number
    return result


def get_derived_series(series_vars, derived_series, indy_var, series_val, indy_vals):
    for derived_serie in derived_series:
        derived_val = series_val.copy()
        derived_val[series_vars[-1]] = None
        ds_1 = derived_serie[0].split(' ')
        ds_2 = derived_serie[1].split(' ')

        for var in series_vars:
            if derived_val[var] is not None \
                    and intersection(derived_val[var], ds_1) \
                    == intersection(derived_val[var], ds_1):
                derived_val[var] = intersection(derived_val[var], ds_1)
        derived_curve_name = get_derived_curve_name(derived_serie)
        derived_val[series_vars[-1]] = [derived_curve_name]
        derived_val[indy_var] = indy_vals
        global diff_name_to_values
        diff_name_to_values[derived_curve_name] \
            = DerivedCurveComponent(ds_1, ds_2, derived_serie[-1])
        if ds_1[-1] == ds_2[-1]:
            derived_val['stat_name'] = [ds_1[-1]]
        else:
            derived_val['stat_name'] = [ds_1[-1] + "," + ds_2[-1]]

        return list(itertools.product(*derived_val.values()))



def calculate_value_and_ci(input_parameters):
    input_data = pd.read_csv(
        input_parameters['input_data_file'],
        header=[0],
        sep='\t')
    global COLUMN_NAMES
    COLUMN_NAMES = input_data.columns.values

    # set random seed if present
    if input_parameters['random_seed'] is not None:
        np.random.seed(input_parameters['random_seed'])

    is_event_equal = True

    # replace thresh_i values for reliability plot
    indy_vals = input_parameters['indy_vals']
    if input_parameters['indy_var'] == 'thresh_i' and input_parameters['line_type'] == 'pct':
        indy_vals = input_data['thresh_i'].sort()
        indy_vals = np.unique(indy_vals)

    # TODO implement ungrouping!!!!
    group_to_value = {}
    series_val = input_parameters['series_val']
    if series_val:
        for index, key in enumerate(series_val.keys()):
            for val in series_val[key]:
                if ',' in val:
                    new_name = 'Group_y1_' + str(index + 1)
                    group_to_value[new_name] = val

    if is_event_equal:
        input_data = perform_event_equalization(input_data, indy_vals, input_parameters)

    # TODO contourDiff adjustments

    all_fields_values = series_val.copy()
    all_fields_values[input_parameters['indy_var']] = indy_vals
    all_fields_values['stat_name'] = input_parameters['list_stat']
    all_series = list(itertools.product(*all_fields_values.values()))

    derived_series = input_parameters['derived_series']
    if derived_series:
        all_series.extend(get_derived_series(list(series_val.keys()), derived_series,
                                             input_parameters['indy_var'], series_val, indy_vals))


    out_frame = init_out_frame(
        input_parameters['list_static_val'],
        all_fields_values.keys(),
        all_series)

    series_ind = 0
    global STATISTIC
    for stat_upper in input_parameters['list_stat']:
        STATISTIC = stat_upper.lower()
        series_to_distrib = {}
        for series in all_series:
            is_derived = is_derived_series(series)
            if not is_derived:

                # filter series data
                all_filters = []
                for field_ind, field in enumerate(all_fields_values.keys()):
                    filter_value = series[field_ind]
                    if is_string_integer(filter_value):
                        filter_value = int(filter_value)
                    all_filters.append((input_data[field] == filter_value))

                # use numpy to select the rows where any record evaluates to True
                mask = np.array(all_filters).all(axis=0)
                series_data = input_data.loc[mask]

                bootstrap_results = get_bootsrtapped_stats(series_data, input_parameters)
                series_to_distrib[series] = bootstrap_results

            else:
                bootstrap_results = get_bootsrtapped_stats_for_derived(
                    list(series[1:]),
                    diff_name_to_values[series[0]],
                    series_to_distrib,
                    input_parameters)

            out_frame['stat_value'][series_ind] = bootstrap_results.value
            out_frame['stat_bcl'][series_ind] = bootstrap_results.lower_bound
            out_frame['stat_bcu'][series_ind] = bootstrap_results.upper_bound
            out_frame['nstats'][series_ind] = 0

            print(
                "{},( {},{})".format(bootstrap_results.value, bootstrap_results.lower_bound,
                                     bootstrap_results.upper_bound))
            series_ind = series_ind + 1

    export_csv = out_frame.to_csv('/Users/tatiana/ee_testing/sl1l2_agg_stats_py.data',
                                  index=None, header=True,
                                  sep="\t", na_rep="NA")


if __name__ == "__main__":
    params = ({
        'method': 'perc',
        'num_iterations': 1000,
        'num_threads': -1,
        'alpha': 0.05,
        'input_data_file': '/Users/tatiana/ee_testing/data_sl1l2_agg_stats.data',
        'random_seed': 1,
        'line_type': 'sl1l2',
        'indy_var': "fcst_lead",
        'indy_vals': ["0", "120000", "240000", "360000", "480000"],
        'series_val': dict({'model': ["expV36", "expV36_MT10_CRIS"]}),
        'list_stat': ['FBAR'],
        'fcst_var_val': dict({'DPT': ["FBAR"]}),
        'list_static_val': dict({'fcst_var': 'DPT'}),
        'fixed_vars_vals_input': dict({
            'vx_mask': dict({'vx_mask_0': ["FULL"]}),
            'obtype': dict({'obtype_1': ["ADPUPA"]}),
            'fcst_lev': dict({'fcst_lev_2': ["P300"]})
        }),

        'derived_series': [["expV36 DPT FBAR", "expV36_MT10_CRIS DPT FBAR", "DIFF"]]
    })
    calculate_value_and_ci(params)
