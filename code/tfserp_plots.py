import argparse
import csv
import glob
import os
import pickle
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib.backends.backend_pdf import PdfPages

# TODO: this file is work in progress
plt.rcParams.update({"text.usetex": True})


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


def extract_correlations(args, directory_list, file_str=None):
    """[summary]

    Args:
        directory_list ([type]): [description]
        file_str ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """

    # Load the subject's electrode file
    # electrode_list = list(load_pickle(args.electrode_file).values())
    # if args.electrodes is not None:
    #     electrode_list = [electrode_list[idx-1] for idx in args.electrodes]

    all_corrs = []
    all_sems = []
    electrode_list = []
    for dir in directory_list:
        correlation_files = glob.glob(os.path.join(dir, "*"))
        correlation_files.sort()

        dir_corrs = []
        dir_sems = []
        for file in correlation_files:
            # file = os.path.join(dir, electrode + '_' + file_str + '.csv')
            electrode = file.split('/')[-1]
            if file_str not in electrode:
                continue
            electrode_list.append(electrode[:-9])

            with open(file, 'r') as csv_file:
                ha = list(map(float, csv_file.readline().strip().split(',')))
                ha_sem = list(map(float, csv_file.readline().strip().split(',')))
            dir_corrs.append(ha)
            dir_sems.append(ha_sem)

        all_corrs.append(dir_corrs)
        all_sems.append(dir_sems)

    # TODO - there is probably a more elegant way to do this...
    # format for combined subjects (i.e., move x dims to 2 dims)
    # assume that only 2 conditions can be given but infinite subjects
    # import pdb; pdb.set_trace()
    if args.combine_subjects:
        all_corrs_new = []
        all_sems_new = []
        num_subs = len(all_corrs)/2

        # iterate through 2 conditions
        for i in range(0,2):
            itr_list1 = all_corrs[i]
            itr_list2 = all_sems[i]

            # iterate through all subjects
            for j in range(1, int(num_subs)):
                itr_list1 += all_corrs[i+(2*j)]
                itr_list2 += all_sems[i+(2*j)]

            all_corrs_new.append(itr_list1)
            all_sems_new.append(itr_list2)

        all_corrs = all_corrs_new
        all_sems = all_sems_new


    all_corrs = np.stack(all_corrs)
    all_sems = np.stack(all_sems)

    # all_corrs.shape = [len(directory_list), 1, num_lags]
    mean_corr = np.mean(all_corrs, axis=1)
    sem_corr = np.std(all_corrs, axis=1, ddof=1) / np.sqrt(all_corrs.shape[1])

    return all_corrs, mean_corr, electrode_list, all_sems, sem_corr


def save_max_correlations(args, prod_max, comp_max, prod_list):
    """[summary]

    Args:
        args ([type]): [description]
        prod_max ([type]): [description]
        comp_max ([type]): [description]
        prod_list ([type]): [description]
    """
    df = pd.DataFrame(prod_max, columns=['production'])
    df['comprehension'] = comp_max
    df['electrode'] = [int(item.strip('elec')) for item in prod_list]
    df = df[['electrode', 'production', 'comprehension']]
    df.to_csv(args.max_corr_csv, index=False)
    return


def parse_arguments():
    """Read commandline arguments
    Returns:
        Namespace: input as well as default arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--output-prefix', type=str, default='test')
    parser.add_argument('--input-directory', nargs='*', type=str, default=None)
    parser.add_argument('--labels', nargs='*', type=str, default=None)
    parser.add_argument('--embedding-type', type=str, default=None)
    parser.add_argument('--electrodes', nargs='*', type=int, default=[])
    parser.add_argument('--output-file-name', type=str, default=None)

    parser.add_argument('--prod', action='store_true', default=False)
    parser.add_argument('--comp', action='store_true', default=False)

    parser.add_argument('--combine-subjects', action='store_true', default=False)

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--sid', nargs='?', type=int, default=None)
    group.add_argument('--sig-elec-file', nargs='?', type=str, default=None)

    args = parser.parse_args()

    if not args.sid and args.electrodes:
        parser.error("--electrodes requires --sid")

    return args


def initial_setup(args):
    assert len(args.input_directory) == len(args.labels), "Unequal number of"

    full_input_dir = os.path.join(os.getcwd(), 'Results', args.input_directory)
    args.output_pdf = os.path.join(
        os.getcwd(),
        '_'.join([str(args.sid), args.embedding_type, 'encoding.pdf']))
    args.max_corr_csv = os.path.join(
        full_input_dir,
        '_'.join([str(args.sid), args.embedding_type, 'maxCorrelations.csv']))
    return


def set_legend_labels(args):
    legend_labels = []

    for item in ['production', 'comprehension']:
        for label in args.labels:
            legend_labels.append(r'\textit{' + '-'.join([label, item]) + '}')
    return legend_labels


def plot_data(args, data, pp, sem, title=None):
    # define x vector for plotting
    sample = 1/512
    lags = np.arange(-2, 2 + sample, sample)

    fig, ax = plt.subplots()

    # import pdb; pdb.set_trace()
    # define colors given data
    rows = data.shape[0]
    if rows==1:
        if args.prod:
            colors = ['b']
        else: 
            colors = ['r']
    elif rows==2:
        if args.prod & args.comp:
            colors = ['b','r']
        elif args.prod:
            colors = ['b','c']
        else:
            colors = ['r','m']
    elif rows==4:
        colors = ['b','c','r','m']
    else:
        print('wrong dimension given to plotting function')
        return

    ax.set_prop_cycle(color=colors)
    ax.plot(lags, data.T, linewidth=.75)
    ax.legend(set_legend_labels(args), frameon=False)
    ax.set(xlabel=r'\textit{onset (s)}',
           ylabel=r'\textit{event-related potential}',
           title=title)
    # ax.set_ylim(-0.05, 0.50)
    # ax.vlines(0, -0.05, 0.50, 'k', linestyles='dashed', linewidth=.75)
    ax.set_ylim(-.2, 1)
    ax.vlines(0, -.2, 1, 'k', linestyles='dashed', linewidth=1)

    for i in range(0, rows):
        plt.fill_between(lags, data.T[:,i]-sem.T[:,i], data.T[:,i]+sem.T[:,i], 
                        color=colors[i], alpha=0.3)

    pp.savefig(fig)
    plt.close()


def plot_average_correlations_multiple(pp, prod_corr_mean, comp_corr_mean,
                                       prod_sem, comp_sem, args):

    sem = np.vstack([prod_sem, comp_sem])
    data = np.vstack([prod_corr_mean, comp_corr_mean])
    plot_data(args, data, pp, sem,
              r'\textit{Average ERP (all electrodes)}')


def plot_average_correlations_one(pp, prod_corr_mean, prod_sem, args):

    sem = np.vstack(prod_sem)
    data = np.vstack(prod_corr_mean)
    plot_data(args, data, pp, sem,
              r'\textit{Average ERP (all electrodes)}')


def plot_individual_correlation_multiple(pp, prod_corr, comp_corr, prod_sem,
                                         comp_sem, prod_list, args):

    prod_list = [item.replace('_', '\_') for item in prod_list]
    prod_corr = np.moveaxis(prod_corr, [0, 1, 2], [1, 0, 2])
    comp_corr = np.moveaxis(comp_corr, [0, 1, 2], [1, 0, 2])
    prod_sem = np.moveaxis(prod_sem, [0, 1, 2], [1, 0, 2])
    comp_sem = np.moveaxis(comp_sem, [0, 1, 2], [1, 0, 2])

    for prod_row, comp_row, sem_rowp, sem_rowc, electrode_id in zip(prod_corr, 
                                     comp_corr, prod_sem, comp_sem, prod_list):
        data = np.vstack([prod_row, comp_row])
        sem = np.vstack([sem_rowp, sem_rowc])
        plot_data(args, data, pp, sem, electrode_id)


def plot_individual_correlation_one(pp, prod_corr, prod_sem, prod_list, args):

    prod_list = [item.replace('_', '\_') for item in prod_list]
    prod_corr = np.moveaxis(prod_corr, [0, 1, 2], [1, 0, 2])
    prod_sem = np.moveaxis(prod_sem, [0, 1, 2], [1, 0, 2])

    for prod_row, sem_rowp, electrode_id in zip(prod_corr, 
                                     prod_sem, prod_list):
        data = prod_row
        sem = sem_rowp
        plot_data(args, data, pp, sem, electrode_id)


if __name__ == '__main__':
    # Parse input arguments
    args = parse_arguments()

    args.output_pdf = os.path.join(os.getcwd(), 'results', 'erp', 'figures',
                                   args.output_file_name + '.pdf')
    args.electrode_file = os.path.join(os.getcwd(), 'data', str(args.sid),
                                       str(args.sid) + '_electrode_names.pkl')

    # Results folders to be plotted
    results_dirs = [
        glob.glob(os.path.join(os.getcwd(), 'results', 'erp', directory))[0]
        for directory in args.input_directory
    ]

    if args.prod:
        prod_corr, prod_corr_mean, prod_list, prod_sem, all_semp = extract_correlations(
            args, results_dirs, 'prod')

    if args.comp:
        comp_corr, comp_corr_mean, comp_list, comp_sem, all_semc = extract_correlations(
            args, results_dirs, 'comp')

    pp = PdfPages(args.output_pdf)

    # plot both production and comprehension
    if args.prod & args.comp:
        plot_average_correlations_multiple(pp, prod_corr_mean, comp_corr_mean,
                                                all_semp, all_semc, args)

        plot_individual_correlation_multiple(pp, prod_corr, comp_corr, prod_sem, 
                                                comp_sem, prod_list, args)
    
    # only plot production 
    elif args.prod: 
        plot_average_correlations_one(pp, prod_corr_mean, all_semp, args)
        plot_individual_correlation_one(pp, prod_corr, prod_sem, prod_list, args)

    # only plot comprehension
    elif args.comp: 
        plot_average_correlations_one(pp, comp_corr_mean, all_semc, args)
        plot_individual_correlation_one(pp, comp_corr, comp_sem, comp_list, args)


    pp.close()
