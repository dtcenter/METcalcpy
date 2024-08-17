import pandas as pd
import argparse
'''
   Converts the uppercase MET headers into lower case for command line usage of 
   agg_stat.py and other statistics modules.
   
   Requires the full path to the input file and a full path to the output file.
   
'''

def change_header_case(input_args):
    '''

    Args:
        input_args: the argparse argument object containing the input and output filenames

    Returns:
        None.  Creates the MET input file with lower case headers.  Saved in the location specified
        from the command line arguments
    '''

    print(f"reading MET input with upper case headers:{input_args.input_file} ")
    df = pd.read_csv(input_args.input_file, sep=r"\s+")
    uc_cols = df.columns.to_list()
    lc_cols = [lc_cols.lower() for lc_cols in uc_cols]
    df.columns = lc_cols

    df.to_csv(input_args.output_file, sep="\t", index=False)


    print(f"saving MET output with lower case headers: {input_args.output_file} ")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Changes upper case MET headers into lower case")

    parser.add_argument('-i', type=str, dest='input_file', required=True)
    parser.add_argument('-o', type=str, dest='output_file', required=True)

    input_args = parser.parse_args()
    change_header_case(input_args)