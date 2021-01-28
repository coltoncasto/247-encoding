import csv
import os

import mat73
import numpy as np
from numba import jit, prange
from scipy import stats
from sklearn.model_selection import KFold
from tfsenc_phase_shuffle import phase_randomize


@jit(nopython=True)
def build_Y(onsets, brain_signal, lags, window_size):
    """[summary]

    Args:
        onsets ([type]): [description]
        brain_signal ([type]): [description]
        lags ([type]): [description]
        window_size ([type]): [description]

    Returns:
        [type]: [description]
    """

    half_window = round((window_size / 1000) * 512 / 2)
    t = len(brain_signal)

    Y1 = np.zeros((len(onsets), len(lags), 2 * half_window + 1))

    for lag in prange(len(lags)):
        lag_amount = int(lags[lag] / 1000 * 512)

        index_onsets = np.minimum(
            t - half_window - 1,
            np.maximum(half_window + 1,
                       np.round_(onsets, 0, onsets) + lag_amount))

        # subtracting 1 from starts to account for 0-indexing
        starts = index_onsets - half_window - 1
        stops = index_onsets + half_window

        # vec = brain_signal[np.array([np.arange(*item) for item in zip(starts, stops)])]

        for i, (start, stop) in enumerate(zip(starts, stops)):
            Y1[i, lag, :] = brain_signal[start:stop].reshape(-1)

    return Y1


def build_XY(args, datum, brain_signal):
    """[summary]

    Args:
        args ([type]): [description]
        datum ([type]): [description]
        brain_signal ([type]): [description]

    Returns:
        [type]: [description]
    """
    X = np.stack(datum.embeddings).astype('float64')

    onsets = datum.onset.values

    lags = np.array(args.lags)
    brain_signal = brain_signal.reshape(-1, 1)

    Y = build_Y(onsets, brain_signal, lags, args.window_size)

    return X, Y


def encode_lags_numba(args, X, Y):
    """[summary]
    Args:
        X ([type]): [description]
        Y ([type]): [description]
    Returns:
        [type]: [description]
    """
    if args.shuffle:
        np.random.shuffle(Y)

    if args.phase_shuffle:
        Y = phase_randomize(Y)

    Y = np.mean(Y, axis=-1)

    return 


def run_save_permutation(args, prod_X, prod_Y, filename):
    """[summary]

    Args:
        args ([type]): [description]
        prod_X ([type]): [description]
        prod_Y ([type]): [description]
        filename ([type]): [description]
    """
    if prod_X.shape[0]:
        perm_prod = []
        for _ in range(args.npermutations):
            perm_rc = encode_lags_numba(args, prod_X, prod_Y)
            perm_prod.append(perm_rc)

        perm_prod = np.stack(perm_prod)
        with open(filename, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(perm_prod)


def load_header(conversation_dir, subject_id):
    """[summary]

    Args:
        conversation_dir ([type]): [description]
        subject_id (string): Subject ID

    Returns:
        list: labels
    """
    misc_dir = os.path.join(conversation_dir, subject_id, 'misc')
    header_file = os.path.join(misc_dir, subject_id + '_header.mat')
    if not os.path.exists(header_file):
        return
    header = mat73.loadmat(header_file)
    labels = header.header.label

    return labels


def create_output_directory(args):
    output_prefix_add = '-'.join(args.emb_file.split('_')[:-1])

    folder_name = '-'.join([args.output_prefix, output_prefix_add])
    folder_name = folder_name + '-pca_' + str(args.reduce_to) + 'd'

    full_output_dir = os.path.join(args.output_dir, folder_name)

    os.makedirs(full_output_dir, exist_ok=True)

    return full_output_dir


def encoding_regression(args, sid, datum, elec_signal, name):

    output_dir = args.full_output_dir

    onsets = datum.onset.values
    lags = np.array(args.lags)
    brain_signal = elec_signal.reshape(-1, 1)

    # Build design matrices
    Y = build_Y(onsets, brain_signal, lags, args.window_size)

    # Split into production and comprehension
    prod_Y = Y[datum.speaker == 'Speaker1', :]
    comp_Y = Y[datum.speaker != 'Speaker1', :]

    print(f'{sid} {name} Prod: {len(prod_Y)} Comp: {len(comp_Y)}')

    # Run permutation and save results
    filename = os.path.join(output_dir, name + '_prod.csv')
    run_save_permutation(args, prod_X, prod_Y, filename)

    filename = os.path.join(output_dir, name + '_comp.csv')
    run_save_permutation(args, comp_X, comp_Y, filename)

    return


def setup_environ(args):
    """Update args with project specific directories and other flags
    """
    PICKLE_DIR = os.path.join(os.getcwd(), 'data')
    path_dict = dict(PICKLE_DIR=PICKLE_DIR)

    if args.emb_type == 'glove50':
        stra = 'cnxt_' + str(args.context_length)
        args.emb_file = '_'.join(
            [str(args.sid), args.emb_type, stra, 'embeddings.pkl'])

        # args.emb_type = args.align_with
        # args.context_length = args.align_target_context_length

        stra = 'cnxt_' + str(args.align_target_context_length)
        args.load_emb_file = '_'.join(
            [str(args.sid), args.align_with, stra, 'embeddings.pkl'])
    else:
        stra = 'cnxt_' + str(args.context_length)
        args.emb_file = '_'.join(
            [str(args.sid), args.emb_type, stra, 'embeddings.pkl'])
        args.load_emb_file = args.emb_file

    args.signal_file = '_'.join([str(args.sid), 'trimmed_signal.pkl'])
    args.electrode_file = '_'.join([str(args.sid), 'electrode_names.pkl'])

    args.output_dir = os.path.join(os.getcwd(), 'results')
    args.full_output_dir = create_output_directory(args)

    vars(args).update(path_dict)
    return args
