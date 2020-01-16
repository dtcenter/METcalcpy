"""
Program Name: test_validate_mv_python.py


How to use:
    python test_validate_mv_python.py <parameters_file>
        where - <parameters_file> is YAML file with parameters
        and environment variable should be set to PYTHONPATH=<path_to_METcalcpy>

    example.yaml:
    input_xml_dir:      /path/to/dir_with_xml_files
    output_xml_dir:     /path/to/dir_with_new_xml_files
    output_plots_dir:   /path/to/dir_with_new_plots_files
    r_dir:              /path/to/dir_with_r_plots
    mv_batch:           /path/to/metviewer_batch_script
    db_user:            db_user_name
    db_password:        db_pass
    start_date:         Optional. Start date of testing %Y-%m-%d. If not specified = yesterday
    end_date:           Optional. End date of testing %Y-%m-%d. If not specified = today


"""
import argparse
import os
import datetime as dt
import sys
import xml.dom.minidom
import subprocess
import yaml
from metcalcpy.compare_images import CompareImages


def replace_name(old_name, postfix):
    """Adds postfix to the end of a file name but before the extension

       Args:
           old_name: file name to be changed
           postfix: postfix
       Returns:
           file name with postfix
    """
    return old_name.replace(".", "_" + postfix + ".")


def main(params):
    """Finds METViewer generated xml files based on the values from the parameters dictionary.
        Adjust XML and reruns it using the Python version of METViewer
        Compares the original and generated images and prints the result

        Args:
            params: input parameters as a dictionary

    """
    # find XML files
    test_xml = get_test_xml(params)

    # rerun each XML with Python
    for file in test_xml:
        print(f'\nChecking {file}')
        doc = xml.dom.minidom.parse(file)

        plot_name = doc.getElementsByTagName('plot_file')[0].firstChild.nodeValue
        original_plot_path = doc.getElementsByTagName('plots')[0].firstChild.nodeValue \
                             + '/' + plot_name

        # replace the original XML with new values
        doc.getElementsByTagName('plot_file')[0].firstChild.nodeValue \
            = replace_name(plot_name, 'py')
        doc.getElementsByTagName('data_file')[0].firstChild.nodeValue \
            = replace_name(doc.getElementsByTagName('data_file')[0].firstChild.nodeValue, 'py')
        doc.getElementsByTagName('r_file')[0].firstChild.nodeValue \
            = replace_name(doc.getElementsByTagName('r_file')[0].firstChild.nodeValue, 'py')
        doc.getElementsByTagName('r_tmpl')[0].firstChild.nodeValue = params['mv_home'] + 'R_tmpl'
        doc.getElementsByTagName('r_work')[0].firstChild.nodeValue = params['mv_home'] + 'R_work'
        doc.getElementsByTagName('plots')[0].firstChild.nodeValue = params['output_plots_dir']
        doc.getElementsByTagName('user')[0].firstChild.nodeValue = params['db_user']
        doc.getElementsByTagName('password')[0].firstChild.nodeValue = params['db_pass']
        doc.getElementsByTagName('data')[0].firstChild.nodeValue = params['output_data_dir']
        doc.getElementsByTagName('scripts')[0].firstChild.nodeValue = params['output_scripts_dir']

        # save new XML
        new_xml_file = params['output_xml_dir'] + os.path.basename(replace_name(file, 'py'))
        with open(new_xml_file, "w") as xml_file:
            doc.writexml(xml_file)

        # run METviewer with the new XML and wait till it is done
        process = subprocess.Popen([params['mv_home'] + '/bin/mv_batch.sh', new_xml_file],
                                   stdout=subprocess.PIPE,
                                   universal_newlines=True)
        while True:
            return_code = process.poll()
            if return_code is not None:
                break

        new_image_path = params['output_plots_dir'] + replace_name(plot_name, 'py')

        # check if both images are present or not
        if not os.path.exists(original_plot_path) \
                and not os.path.exists(new_image_path):
            # if both images don't exist - success
            print(f'SUCCESS: For {plot_name} both images don\'t exist')
        else:
            # compare images
            try:
                compare = CompareImages(original_plot_path, new_image_path)
                ssim = compare.get_mssim()
                if ssim == 1.0:
                    print(f'SUCCESS: For {plot_name} images are identical')
                else:
                    print(f'ERROR: For {plot_name} images are different')
            except KeyError as err:
                print(f'ERROR: For {plot_name} : {err}')


def get_test_xml(params):
    """Creates a list of xml fies based on the parameters values

       Args:
           params: input parameters as a dictionary that include required key'input_xml_dir'
           and optional 'start_date' and 'end_date'
       Returns:
           a list with xml file names
    """
    # calculate testing period
    (start, end) = get_testing_period(params)

    fresh_xml = []
    # for each fie in the directory check the date and extension
    # and include it to the output list if it fits the criteria
    for file in os.listdir(params['input_xml_dir']):
        file_name = params['input_xml_dir'] + file
        filetime = dt.datetime.fromtimestamp(os.path.getctime(file_name))
        extension = os.path.splitext(file)[1]
        if end <= filetime.date() <= start and extension == '.xml':
            fresh_xml.append(file_name)
    return fresh_xml


def get_testing_period(params):
    """Using parameters forom the input dictionary creates the date range.
        If 'start_date' parameter is not present initialises it as yesterday date
        If 'end_date' parameter is not present initialises it as today date

        Args:
           params: input parameters as a dictionary that might include optional keys 'start_date'
           and 'end_date'
        Returns:
           a date range as start and end dates
        Raises: KeyError if the date range is invalid
    """
    if 'start_date' in params.keys():
        start = dt.datetime.strptime(params['start_date'], '%Y-%m-%d').date()
    else:
        # start = yesterday
        start = (dt.datetime.now() - dt.timedelta(1)).date()

    if 'end_date' in params.keys():
        end = dt.datetime.strptime(params['end_date'], '%Y-%m-%d').date()
    else:
        # end = today
        end = dt.datetime.now().date()

    if end < start:
        raise KeyError(f'Invalid start/end dates {start} - {end}')

    return start, end


if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    PARSER = argparse.ArgumentParser(description='List of validate_mv_python arguments')
    PARSER.add_argument("parameter_file", help="Path to YAML parameters file",
                        type=argparse.FileType('r'),
                        default=sys.stdin)
    ARGS = PARSER.parse_args()
    PARAMS = yaml.load(ARGS.parameter_file, Loader=yaml.FullLoader)
    main(PARAMS)
