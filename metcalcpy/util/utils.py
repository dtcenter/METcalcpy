OPERATION_TO_SIGN = {
    'DIFF': '-',
    'RATIO': '/',
    'SS': 'and'
}


def represents_int(possible_int):
    """Checks if the value is integer.

     Args:
         possible_int: value to check

    Returns:
        True - if the input value is an integer
        False - if the input value is not an integer
    """
    return isinstance(possible_int, int)


def is_string_integer(str):
    try:
        int(str)
        return True
    except ValueError:
        return False


def get_derived_curve_name(list_of_names):
    operation = 'DIFF'
    if len(list_of_names) == 3:
        operation = list_of_names[2]
    global OPERATION_TO_SIGN
    return "{}({}{}{})".format(operation, list_of_names[0], OPERATION_TO_SIGN[operation], list_of_names[1])


def calc_derived_curve_value(val1, val2, operation):
    result_val = None
    if 'DIFF' == operation:
        result_val = val1 - val2
    elif 'RATIO' == operation:
        result_val = val1 / val2
    elif 'SS' == operation:
        result_val = (val1 - val2) / val1
    elif 'SINGLE' == operation:
        result_val = val1
    return result_val


def unique(list1):
    # insert the list to the set
    list_set = set(list1)
    # convert the set to the list
    return (list(list_set))


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def is_derived_series(series):
    is_derived = False
    for operation in OPERATION_TO_SIGN.keys():
        for series_component in series:
            if series_component.startswith(operation):
                is_derived = True
                break
    return is_derived
