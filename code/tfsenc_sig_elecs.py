import os
import csv
import glob
from datetime import datetime
import argparse
import numpy as np

def save_electrodes(args, prod, sig_elecs):

    # output file name
    filename = os.path.join(os.getcwd(), 'results', 'sig_elecs', 
                                args.output_file_name)

    # append suffix                        
    if prod: 
        filename += '_prod.csv'
    else: 
        filename += '_comp.csv'

    # write to csv file
    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(sig_elecs)

    return


def sig_elecs_from_thresh(args, prod):
    
    input_path = os.path.join(os.getcwd(), 'results', 'enc',
                                args.input_directory, '*')
    # locate all csv files in directory
    files = glob.glob(input_path)

    # consider production and comprehension separate
    if prod:
        files = [file for file in files if 'prod' in file]
    else:
        files = [file for file in files if 'comp' in file]

    # read correlation files
    sig_electrodes = []
    for file in files:
        with open(file, newline='') as csvfile:
            corr = list(map(float, csvfile.readline().strip().split(',')))

        # determine if significant 
        if np.max(np.array(corr)) > args.sig_thresh:
            sig_electrodes.append(file.split('/')[-1][:-9])

    save_electrodes(args, prod, sig_electrodes)

    return 


def sign_elecs_from_shuffle(args):

    return


def parse_arguments():
    """Read commandline arguments
    Returns:
        Namespace: input as well as default arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--sid', type=int, default=None)
    parser.add_argument('--input-directory', type=str, default=None)
    parser.add_argument('--output-file-name', type=str, default=None)

    parser.add_argument('--sig-flag', action='store_true', default=False)
    parser.add_argument('--sig-thresh', type=float, default=None)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    start_time = datetime.now()
    print(f'Start Time: {start_time.strftime("%A %m/%d/%Y %H:%M:%S")}')

    # Read command line arguments
    args = parse_arguments()

    # find significant electrodes given threshold
    if args.sig_flag:
        sig_elecs_from_thresh(args, True)
        sig_elecs_from_thresh(args, False)

    # TODO - find significant electrodes given phase shuffling

    end_time = datetime.now()
    print(f'End Time: {end_time.strftime("%A %m/%d/%Y %H:%M:%S")}')

    print(f'Total runtime: {end_time - start_time} (HH:MM:SS)')

    