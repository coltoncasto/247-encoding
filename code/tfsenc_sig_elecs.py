import os
import csv
import glob
from datetime import datetime
import argparse
import numpy as np
import pandas as pd
from statsmodels.stats import multitest

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


def sig_elecs_from_phase_shuffle(args, prod):

    subjects = sorted(
    glob.glob(
        '/scratch/gpfs/casto/247-encoding/results/sig-elecs/phase-shuffle/*'))

    lags = np.arange(-2000, 2001, 25)

    some_list = []
    for subject in subjects:
        subject_key = os.path.basename(subject)
        # if subject_key == '625': continue
        shuffle_elec_file_list = sorted(
            glob.glob(
                os.path.join(
                    '/scratch/gpfs/casto/247-encoding/results/sig-elecs/phase-shuffle',
                    os.path.basename(subject), '*.csv')))

        main_elec_file_list = sorted(
            glob.glob(
                os.path.join(
                    '/scratch/gpfs/casto/247-encoding/results/sig-elecs/no-shuffle',
                    os.path.basename(subject), '*.csv')))

        # consider production and comprehension separate
        if prod:
            suffix = '_prod'
        else:
            suffix = '_comp'

        shuffle_elec_file_list = [
            file for file in shuffle_elec_file_list if suffix in file]
        main_elec_file_list = [
            file for file in main_elec_file_list if suffix in file]
        
        a = [os.path.basename(item) for item in shuffle_elec_file_list]
        b = [os.path.basename(item) for item in main_elec_file_list]
        import pdb; pdb.set_trace()
        assert set(a) == set(b), "Mismatch: Electrode Set"

        for elec_file1, elec_file2 in zip(shuffle_elec_file_list,
                                          main_elec_file_list):
            elecname1 = os.path.split(os.path.splitext(elec_file1)[0])[1]
            elecname2 = os.path.split(os.path.splitext(elec_file2)[0])[1]
            
            assert elecname1 == elecname2, 'Mismatch: Electrode Name'

            if elecname1.startswith(('SG', 'ECGEKG', 'EEGSG')):
                continue

            perm_result = pd.read_csv(elec_file1, header=None).values
            rc_result = pd.read_csv(elec_file2, header=None).values

            # print(f'electrode {elecname1}: ' + str(np.max(np.max(perm_result, axis=1))))

            assert perm_result.shape[1] == rc_result.shape[
                1], "Mismatch: Number of Lags"

            if perm_result.shape[1] != len(lags):
                print('perm is wrong length')
            else:
                omaxs = np.max(perm_result, axis=1)

            s = 1 - (sum(np.max(rc_result) > omaxs) / perm_result.shape[0])
            some_list.append((subject_key, elecname1, s))

    df = pd.DataFrame(some_list, columns=['subject', 'electrode', 'score'])
    _, pcor, _, _ = multitest.multipletests(df.score.values,
                                            method='fdr_bh',
                                            is_sorted=False)

    # print(pcor)
    flag = np.logical_or(np.isclose(pcor, args.sig_thresh), pcor < args.sig_thresh)

    df = df[flag]
    # remove suffix from elec name
    df['electrode'] = df['electrode'].str.strip(suffix)

    # output intersection with other file
    if args.intersection:
        if prod:
            intersection(df, args.intersection_file_prod, suffix)
        else:
            intersection(df, args.intersection_file_comp, suffix)

    # output file
    output_file_name = os.path.join(os.getcwd(), 'results', 'sig-elecs',
                                        f'sig_elecs{suffix}.csv')
    df.to_csv(output_file_name,
                index=False,
                columns=['subject', 'electrode', 'score'])

    return

def intersection(df, file, suffix):

    full_file = os.path.join(os.getcwd(), 'results', 'sig-elecs', file)
    other_df = pd.read_csv(full_file)
    
    
    df['subject'] = df['subject'].astype(int)
    import pdb; pdb.set_trace()
    df_combined = something
    df_intersection = df_combined[df_combined['_merge']=='both']
    df_intersection.drop(columns=['_merge'])
                
    output_file_name = os.path.join(os.getcwd(), 'results', 'sig-elecs',
                                        f'sig_elecs{suffix}_intersection.csv')
    df_intersection.to_csv(output_file_name,
                            index=False,
                            columns=['subject', 'electrode'])
    
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

    parser.add_argument('--sig-thresh-flag', action='store_true', default=False)
    parser.add_argument('--sig-phase-shuffle-flag', action='store_true', default=False)
    parser.add_argument('--sig-thresh', type=float, default=None) # corr or alpha

    parser.add_argument('--intersection', action='store_true', default=False)
    parser.add_argument('--intersection-file-prod', type=str, default=None)
    parser.add_argument('--intersection-file-comp', type=str, default=None)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    start_time = datetime.now()
    print(f'Start Time: {start_time.strftime("%A %m/%d/%Y %H:%M:%S")}')


    # Read command line arguments
    args = parse_arguments()

    # find significant electrodes given threshold
    if args.sig_thresh_flag:
        sig_elecs_from_thresh(args, True)
        sig_elecs_from_thresh(args, False)

    # find significant electrodes given phase shuffling
    if args.sig_phase_shuffle_flag:
        sig_elecs_from_phase_shuffle(args, True)
        sig_elecs_from_phase_shuffle(args, False)


    end_time = datetime.now()
    print(f'End Time: {end_time.strftime("%A %m/%d/%Y %H:%M:%S")}')

    print(f'Total runtime: {end_time - start_time} (HH:MM:SS)')

    