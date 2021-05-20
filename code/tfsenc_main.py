import argparse
import glob
import os
import csv
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.io import loadmat

from tfsenc_pca import run_pca
from tfsenc_read_datum import read_datum
from tfsenc_utils import encoding_regression, load_header, setup_environ, write_config


def load_pickle(file):
    """Load the datum pickle and returns as a dataframe

    Args:
        file (string): labels pickle from 247-decoding/tfs_pickling.py

    Returns:
        DataFrame: pickle contents returned as dataframe
    """
    with open(file, 'rb') as fh:
        datum = pickle.load(fh)

    return datum


def parse_arguments():
    """Read commandline arguments
    Returns:
        Namespace: input as well as default arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--word-value', type=str, default='all')
    parser.add_argument('--window-size', type=int, default=200)

    group1 = parser.add_mutually_exclusive_group()
    group1.add_argument('--shuffle', action='store_true', default=False)
    group1.add_argument('--phase-shuffle', action='store_true', default=False)

    parser.add_argument('--lags', nargs='+', type=int)
    parser.add_argument('--output-prefix', type=str, default='test')
    parser.add_argument('--emb-type', type=str, default=None)
    parser.add_argument('--context-length', type=int, default=0)
    parser.add_argument('--datum-emb-fn',
                        type=str,
                        default='podcast-datum-glove-50d.csv')
    parser.add_argument('--electrodes', nargs='*', type=int)
    parser.add_argument('--npermutations', type=int, default=1)
    parser.add_argument('--min-word-freq', nargs='?', type=int, default=1)

    parser.add_argument('--sid', type=int, default=None)
    parser.add_argument('--sig-elec-file', type=str, default=None)

    parser.add_argument('--remove-glove', action='store_true', default=False)
    
    parser.add_argument('--pca-flag', action='store_true', default=False)
    parser.add_argument('--reduce-to', type=int, default=0)

    parser.add_argument('--align-with', type=str, default=None)
    parser.add_argument('--align-target-context-length', type=int, default=0)

    parser.add_argument('--split-flag', action='store_true', default=False)
    parser.add_argument('--split-by', type=str, default=None)

    parser.add_argument('--conversation-id-flag', action='store_true', default=False)
    parser.add_argument('--conversation-id', type=int, default=None)

    custom_args = [
        '--sid', '625',
        '--electrodes', '1', '--emb-type', 'glove50',
        '--context-length', '1024', '--align-with', 'gpt2-xl',
        '--align-target-context-length', '1024', '--window-size', '200',
        '--word-value', 'all', '--npermutations', '1', '--lags', '-100', '0',
        '100', '--min-word-freq', '1', '--output-prefix', 'harsha-247-test'
    ]
    args = parser.parse_args(custom_args)

    if not args.pca_flag:
        args.reduce_to = 0

    if args.pca_flag and not args.reduce_to:
        parser.error("Cannot reduce PCA to 0 dimensions")

    if not args.sid and args.electrodes:
        parser.error("--electrodes requires --sid")

    return args


def load_electrode_data(args, elec_id):
    '''Loads specific electrodes mat files
    '''
    DATA_DIR = '/projects/HASSON/247/data/conversations-car'
    convos = sorted(glob.glob(os.path.join(DATA_DIR, str(args.sid), '*')))
    # print(elec_id)
    all_signal = []
    for convo in convos:

        file = glob.glob(
            os.path.join(convo, 'preprocessed',
                         '*_' + str(elec_id) + '.mat'))[0]
        mat_signal = loadmat(file)['p1st']
        
        # print(file.split('/')[-1])

        # mat_signal = trim_signal(mat_signal)

        if mat_signal is None:
            continue
        all_signal.append(mat_signal)

    elec_signal = np.vstack(all_signal)
    return elec_signal


def process_subjects(args, datum):
    """Run encoding on particular subject (requires specifying electrodes)
    """
    electrode_info = load_pickle(
        os.path.join(args.PICKLE_DIR, str(args.sid), args.electrode_file))

    # Read in the significant electrodes
    if args.sig_elec_file:
        SIG_DIR = os.path.join(os.getcwd(), 'results', 'sig-elecs')
        sig_elec_file = os.path.join(SIG_DIR, args.sig_elec_file)
        sig_elec_list = pd.read_csv(sig_elec_file)
        sig_elec_list = sig_elec_list[sig_elec_list.subject == args.sid]


        key_list = list(electrode_info.keys())
        val_list = list(electrode_info.values())
        positions = [val_list.index(value) for value in 
                                        sig_elec_list.electrode.values]
 
        key_list = [key_list[i] for i in positions]
        electrode_info = dict(zip(key_list,sig_elec_list.electrode.values))

    # use electrodes given in Makefile
    else:
        electrode_info = {
            key: electrode_info.get(key, None)
            for key in args.electrodes
        }

    # Loop over each electrode
    for elec_id, elec_name in electrode_info.items():

        if elec_name is None:
            print(f'Electrode ID {elec_id} does not exist')
            continue
        
        elec_signal = load_electrode_data(args, elec_id)

        encoding_regression(args, args.sid, datum, elec_signal, elec_name)

    return


if __name__ == "__main__":
    start_time = datetime.now()
    print(f'Start Time: {start_time.strftime("%A %m/%d/%Y %H:%M:%S")}')

    # Read command line arguments
    args = parse_arguments()

    # Setup paths to data
    args = setup_environ(args)

    # Saving configuration to output directory
    write_config(vars(args))

    # Locate and read datum
    datum = read_datum(args)

    if args.pca_flag:
        datum = run_pca(args, datum)

    # processing individual subjects
    process_subjects(args, datum)



    end_time = datetime.now()
    print(f'End Time: {end_time.strftime("%A %m/%d/%Y %H:%M:%S")}')

    print(f'Total runtime: {end_time - start_time} (HH:MM:SS)')
