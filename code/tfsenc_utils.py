import csv
import os
import json

import mat73
import numpy as np
from numba import jit, prange
from scipy import stats
from sklearn.model_selection import KFold
from tfsenc_phase_shuffle import phase_randomize
from functools import partial
from multiprocessing import Pool


def encColCorr(CA, CB):
    """[summary]

    Args:
        CA ([type]): [description]
        CB ([type]): [description]

    Returns:
        [type]: [description]
    """
    df = np.shape(CA)[0] - 2

    CA -= np.mean(CA, axis=0)
    CB -= np.mean(CB, axis=0)

    r = np.sum(CA * CB, 0) / np.sqrt(np.sum(CA * CA, 0) * np.sum(CB * CB, 0))

    t = r / np.sqrt((1 - np.square(r)) / df)
    p = stats.t.sf(t, df)

    r = r.squeeze()

    if r.size > 1:
        r = r.tolist()
    else:
        r = float(r)

    return r, p, t


def cv_lm_003(X, Y, kfolds):
    """Cross-validated predictions from a regression model using sequential
        block partitions with nuisance regressors included in the training
        folds

    Args:
        X ([type]): [description]
        Y ([type]): [description]
        kfolds ([type]): [description]

    Returns:
        [type]: [description]
    """
    skf = KFold(n_splits=kfolds, shuffle=False)

    # Data size
    nSamps = X.shape[0]
    try:
        nChans = Y.shape[1]
    except E:
        nChans = 1

    # Extract only test folds
    folds = [t[1] for t in skf.split(np.arange(nSamps))]

    YHAT = np.zeros((nSamps, nChans))
    # Go through each fold, and split
    for i in range(kfolds):
        # Shift the number of folds for this iteration
        # [0 1 2 3 4] -> [1 2 3 4 0] -> [2 3 4 0 1]
        #                       ^ dev fold
        #                         ^ test fold
        #                 | - | <- train folds
        folds_ixs = np.roll(range(kfolds), i)
        test_fold = folds_ixs[-1]
        train_folds = folds_ixs[:-1]
        # print(f'\nFold {i}. Training on {train_folds}, '
        #       f'test on {test_fold}.')

        test_index = folds[test_fold]
        # print(test_index)
        train_index = np.concatenate([folds[j] for j in train_folds])

        # Extract each set out of the big matricies
        Xtra, Xtes = X[train_index], X[test_index]
        Ytra, Ytes = Y[train_index], Y[test_index]

        # Mean-center
        Xtra -= np.mean(Xtra, axis=0)
        Xtes -= np.mean(Xtes, axis=0)
        Ytra -= np.mean(Ytra, axis=0)
        Ytes -= np.mean(Ytes, axis=0)

        # Fit model
        B = np.linalg.pinv(Xtra) @ Ytra

        # Predict
        foldYhat = Xtes @ B

        # Add to data matrices
        YHAT[test_index, :] = foldYhat.reshape(-1, nChans)

    return YHAT


@jit(nopython=True)
def fit_model(X, y):
    """Calculate weight vector using normal form of regression.
    
    Returns:
        [type]: (X'X)^-1 * (X'y)
    """
    beta = np.linalg.solve(X.T.dot(X), X.T.dot(y))
    return beta


# @jit(nopython=True)
def build_Y(onsets, convo_onsets, convo_offsets, brain_signal, lags, window_size):
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

    Y1 = np.zeros((len(onsets), len(lags), 2 * half_window + 1))

    for lag in prange(len(lags)):
        lag_amount = int(lags[lag] / 1000 * 512)

        # import pdb; pdb.set_trace()
        index_onsets = np.minimum(
            convo_offsets - half_window - 1,
            np.maximum(convo_onsets + half_window + 1,
                       np.round_(onsets, 0, onsets) + lag_amount))

        # subtracting 1 from starts to account for 0-indexing
        starts = index_onsets - half_window - 1
        stops = index_onsets + half_window

        for i, (start, stop) in enumerate(zip(starts, stops)):
            Y1[i, lag, :] = brain_signal[int(start):int(stop)].reshape(-1)

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

    onsets = datum.adjusted_onset.values
    convo_onsets = datum.conversation_onset
    convo_offsets = datum.conversation_offset

    lags = np.array(args.lags)
    brain_signal = brain_signal.reshape(-1, 1)

    Y = build_Y(onsets, convo_onsets, convo_offsets, brain_signal, 
                                                lags, args.window_size)
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

    PY_hat = cv_lm_003(X, Y, 10)
    rp, _, _ = encColCorr(Y, PY_hat)

    return rp


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
        for i in range(args.npermutations):
            perm_rc = encode_lags_numba(args, prod_X, prod_Y)
            perm_prod.append(perm_rc)

        perm_prod = np.stack(perm_prod)
        # with Pool() as pool:
        #     perm_prod = pool.map(
        #         partial(encoding_mp, args=args, prod_X=prod_X, prod_Y=prod_Y),
        #         range(args.npermutations))

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

    # Build design matrices
    X, Y = build_XY(args, datum, elec_signal)

    # Split into production and comprehension
    prod_X = X[datum.speaker == 'Speaker1', :]
    comp_X = X[datum.speaker != 'Speaker1', :]

    prod_Y = Y[datum.speaker == 'Speaker1', :]
    comp_Y = Y[datum.speaker != 'Speaker1', :]

    print(f'{sid} {name} Prod: {len(prod_X)} Comp: {len(comp_X)}')

    # Run permutation and save results
    filename = os.path.join(output_dir, name + '_prod.csv')
    run_save_permutation(args, prod_X, prod_Y, filename)

    filename = os.path.join(output_dir, name + '_comp.csv')
    run_save_permutation(args, comp_X, comp_Y, filename)

    return

def write_config(dictionary):
    """Write configuration to a file
    Args:
        CONFIG (dict): configuration
    """
    json_object = json.dumps(dictionary, sort_keys=True, indent=4)

    config_file = os.path.join(dictionary['full_output_dir'], 'config.json')
    with open(config_file, "w") as outfile:
        outfile.write(json_object)


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
    args.stitch_file = '_'.join([str(args.sid), 'full_stitch_index.pkl'])

    args.output_dir = os.path.join(os.getcwd(), 'results', 'enc')
    args.full_output_dir = create_output_directory(args)

    vars(args).update(path_dict)
    return args
