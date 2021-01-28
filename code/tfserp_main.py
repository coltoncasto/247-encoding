import argparse
import glob
import os
import pickle
import string
from datetime import datetime
from itertools import islice

import gensim.downloader as api
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data



def save_pickle(item, file_name):
    """Write 'item' to 'file_name.pkl'
    """
    add_ext = '' if file_name.endswith('.pkl') else '.pkl'

    file_name = file_name + add_ext
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    with open(file_name, 'wb') as fh:
        pickle.dump(item, fh)
    return


def load_labels_pickle(args):
    """Load the labels pickle and returns as a dataframe

    Args:
        file (string): labels pickle from 247-pickling/tfspkl_main.py

    Returns:
        DataFrame: pickle contents returned as dataframe
    """
    with open(args.label_pickle_name, 'rb') as fh:
        datum = pickle.load(fh)

    df = pd.DataFrame.from_dict(datum['labels']) # if dict
    # df = pd.DataFrame(datum) # if list

    if args.conversation_id:
        df = df[df.conversation_id == args.conversation_id]

    return df

def load_signal_pickle(args):
    """Load the signal pickle and returns as a ndarray

    Args:
        file (string): labels pickle from 247-pickling/tfspkl_main.py

    Returns:
        Ndarray: pickle contents returned as ndarray
    """
    with open(args.signal_pickle_name, 'rb') as fh:
        datum = pickle.load(fh)

    # datum is dict with keys 'full_signal', 'full_stitch_index',
    # 'electrode_ids', and 'electrode_names'
    matrix = datum['full_signal'] # if dict
    # df = pd.DataFrame(datum) # if list

    # TODO - extract only one conversation's signal
    # if args.conversation_id:
    #     df = df[df.conversation_id == args.conversation_id]

    # only extract electrode0s that will be used
    if args.electrodes == 'all':
        return matrix
    else:
        matrix = matrix[:,int(args.electrodes)-1]
        return matrix


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

def load_electrode_data(args, elec_id):
    '''Loads specific electrodes mat files
    '''
    DATA_DIR = '/projects/HASSON/247/data/conversations-car'
    convos = sorted(glob.glob(os.path.join(DATA_DIR, str(args.sid), '*')))

    all_signal = []
    for convo in convos:
        file = glob.glob(
            os.path.join(convo, 'preprocessed',
                         '*' + str(elec_id) + '.mat'))[0]
        mat_signal = loadmat(file)['p1st']

        # mat_signal = trim_signal(mat_signal)

        if mat_signal is None:
            continue
        all_signal.append(mat_signal)

    elec_signal = np.vstack(all_signal)

    return elec_signal


def seperate_prediction_labels(labels_df):
    """Split the label df into correct and incorrect gpt2 predictions

    Args:
        DataFrame: labels pickle from 247-pickling/tfspkl_main.py

    Returns:
        DataFrame (x2): one df w/ correct preds, one w/ incorrect
    """
    correct_labels = labels_df[labels_df.word == labels_df.top1_pred]
    incorrect_labels = labels_df[labels_df.word != labels_df.top1_pred]

    return correct_labels, incorrect_labels


def extract_signal(args, labels_df, all_signal):
    """Extract signals from around word onsets 

    Args:
        DataFrame: labels pickle (words x columns)
        Ndarray: all signal for all electrodes (samples x elec)

    Returns:
        ndarray: signal around word onset for given set of labels
    """
    window = args.sample_rate * args.window # in samples
    subset_signal = np.zeros(len(labels_df),window*2)

    # iterate over each word and extract signal around onset
    for i in range(0,len(labels_df)):
        onset = round(labels_df['onset'][i])
        subset_signal = all_signal[onset-window:onset+window,:]

    return subset_signal


def mean_and_sem(matrix):
    """Average matrix over rows and compute standard error

    Args:
        matrix (np.ndarray): signals for a set of words

    Returns:
        np.ndarray: average signal and sem (2xm)
    """
    output = np.zeros((2,matrix.shape[1]))
    output[0,:] = np.mean(matrix, axis=0)
    output[1,:] = np.std(matrix, axis=0, ddof=1) / np.sqrt(np.size(matrix)[1])

    return output

def process_subjects(args, datum):
    """Run encoding on particular subject (requires specifying electrodes)
    """
    electrode_info = load_pickle(
        os.path.join(args.PICKLE_DIR, str(args.sid), args.electrode_file))

    if args.electrodes:
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


def setup_environ(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device

    DATA_DIR = os.path.join(os.getcwd(), 'data')
    RESULTS_DIR = os.path.join(os.getcwd(), 'results')

    args.label_pickle_name = os.path.join(DATA_DIR, args.subject,
                                    args.subject + '_full_labels.pkl')
    args.signal_pickle_name = os.path.join(DATA_DIR, args.subject,
                                    args.subject + '_full_signal.pkl')

    args.input_dir = os.path.join(DATA_DIR, args.subject)
    args.conversation_list = sorted(os.listdir(args.input_dir))

    # if args.subject == '625':
    #     assert len(args.conversation_list) == 54
    # else:
    #     assert len(args.conversation_list) == 79

    args.gpus = torch.cuda.device_count()
    if args.gpus > 1:
        args.model = nn.DataParallel(args.model)

    stra_prod = '_'.join(['test_prod', str(args.context_length)])
    stra_comp = '_'.join(['test_comp', str(args.context_length)])

    # TODO: if multiple conversations are specified in input
    if args.conversation_id:
        output_file_name = args.conversation_list[args.conversation_id]

        # production 
        args.output_dir = os.path.join(RESULTS_DIR, args.subject, 'erp',
                                       stra_prod)
        args.output_file_prod = os.path.join(args.output_dir, output_file_name)

        # production 
        args.output_dir = os.path.join(RESULTS_DIR, args.subject, 'erp',
                                       stra_comp)
        args.output_file_comp = os.path.join(args.output_dir, output_file_name)

    return


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--electrodes', type=str, default='all')
    parser.add_argument('--subject', type=str, default='625')
    parser.add_argument('--context-length', type=int, default=1024)
    parser.add_argument('--conversation-id', type=int, default=0)
    parser.add_argument('--window', type=int, default = 2) # seconds
    parser.add_argument('--sample-rate', type=int, default = 512)

    return parser.parse_args()


def main():
    # Read command line arguments
    args = parse_arguments()

    # Setup paths to data
    args = setup_environ(args)
    
    labels_df = load_labels_pickle(args)
    signal_mat = load_signal_pickle(args)

    # remove words where gpt2 token is not root
    labels_df = labels_df[labels_df.gpt2_token_is_root]

    # seperate correct and incorrect predictions
    correct_labels, incorrect_labels = seperate_prediction_labels(
        args, labels_df)

    # seperate production and comprehension
    correct_prod_labels = correct_labels[labels_df.production==1]
    correct_comp_labels = correct_labels[labels_df.production==0]
    incorrect_prod_labels = incorrect_labels[labels_df.production==1]
    incorrect_comp_labels = incorrect_labels[labels_df.production==0]
    
    # extract signal
    correct_prod_signal = extract_signal(args, correct_prod_labels, 
        signal_mat)
    correct_comp_signal = extract_signal(args, correct_comp_labels, 
        signal_mat)
    incorrect_prod_signal = extract_signal(args, incorrect_prod_labels, 
        signal_mat)
    incorrect_comp_signal = extract_signal(args, incorrect_comp_labels, 
        signal_mat)

    # compute signal mean and standard error of mean
    correct_prod_mean_sem = mean_and_sem(correct_prod_signal)
    correct_comp_mean_sem = mean_and_sem(correct_comp_signal)
    incorrect_prod_mean_sem = mean_and_sem(incorrect_prod_signal)
    incorrect_comp_mean_sem = mean_and_sem(incorrect_comp_signal)

    # concatinate correct and inccorect 
    prod_mean_sem = np.concatenate((correct_prod_mean_sem,
        incorrect_prod_mean_sem), axis=0)
    comp_mean_sem = np.concatenate((correct_comp_mean_sem,
        incorrect_comp_mean_sem), axis=0)

    # save
    save_pickle(prod_mean_sem, args.output_file_prod)
    save_pickle(comp_mean_sem, args.output_file_comp)

    # # plot
    # if args.plot:
    #     plot_erp(args, correct_prod_mean_sem, incorrect_prod_mean_sem)
    #     plot_erp(args, correct_comp_mean_sem, incorrect_comp_mean_sem)
    
    return



if __name__ == '__main__':
    start_time = datetime.now()
    print(f'Start Time: {start_time.strftime("%A %m/%d/%Y %H:%M:%S")}')

    main()

    end_time = datetime.now()
    print(f'End Time: {end_time.strftime("%A %m/%d/%Y %H:%M:%S")}')
    print(f'Total runtime: {end_time - start_time} (HH:MM:SS)')
