import argparse
import csv
import glob
import os
import pickle

from itertools import repeat
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    electrode_list = []
    for dir in directory_list:
        correlation_files = glob.glob(os.path.join(dir, "*"))
        correlation_files.sort()

        dir_corrs = []
        for file in correlation_files:
            # file = os.path.join(dir, electrode + '_' + file_str + '.csv')
            electrode = file.split('/')[-1]
            if file_str not in electrode: # skip comp or prod
                continue
            electrode_list.append(electrode[:-9])

            with open(file, 'r') as csv_file:
                ha = list(map(float, csv_file.readline().strip().split(',')))
            dir_corrs.append(ha)

        all_corrs.append(dir_corrs)

    # TODO - make this cleaner
    subject_list = []
    subject_list += ['625'] * len(all_corrs[0]) 
    # subject_list += ['676'] * len(all_corrs[3])

    size_list = []
    size_list = list(repeat(["27,834","27,834"], len(all_corrs[0]))) # ["6,042", "21,646", "21,646"] ["9,567","33,379","33,379"] ["10,662","32,479","32,479"] ["6,532","21,223","21,223"]
    # size_list += list(repeat(["99,885","99,885"], len(all_corrs[3]))) # ["27,236", "72,162", "72,162"] ["25,189", "79,603", "79,603"] ["27,260", "77,858", "77,858"] ["29,122", "77,858", "77,858"]

    # TODO - there is probably a more elegant way to do this...
    # format for combined subjects (i.e., move x dims to 2 dims)
    # assume that only 2 conditions can be given but infinite subjects

    if args.combine_subjects:
        num_subs = 2
        conditions = int(len(directory_list)/num_subs)
        all_corrs_new = []

        electrode_list_new = []
        electrode_list_new = electrode_list[:len(all_corrs[0])]
        start = len(all_corrs[0])*2
        electrode_list_new += electrode_list[
                    start:start+len(all_corrs[conditions])]
        electrode_list = electrode_list_new

        # iterate through 2 conditions
        for i in range(0,conditions):
            itr_list1 = all_corrs[i]

            # iterate through all subjects
            for j in range(1, int(num_subs)):
                itr_list1 += all_corrs[i+(conditions*j)]

            all_corrs_new.append(itr_list1)

        all_corrs = all_corrs_new

    all_corrs = np.stack(all_corrs)

    # all_corrs.shape = [len(directory_list), 1, num_lags]
    mean_corr = np.mean(all_corrs, axis=1)
    sem_corr = np.std(all_corrs, axis=1, ddof=1) / np.sqrt(all_corrs.shape[1])

    return all_corrs, mean_corr, electrode_list, subject_list, size_list, sem_corr


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


def set_plot_styles(args):
    linestyles = ['-', '--', ':']
    color = ['b', 'r']

    linestyles = linestyles[0:len(args.labels)]# * 2
    color = np.repeat(color[0:len(args.labels)], len(args.labels))

    return (color, linestyles)


def set_legend_labels(args, size=None):
    legend_labels = []

    if args.prod & args.comp: 
        for item in ['production', 'comprehension']:
            for label in args.labels:
                legend_labels.append(r'\textit{' + '-'.join([label, item]) + '}')
    
    elif args.prod: 
        for i, label in enumerate(args.labels):
            if size is not None: 
                legend_labels.append(f'{label} word (n={str(size[i])})')
            else: 
                legend_labels.append(r'\textit{' + ' '.join([label, 'word']) + '}')

    else: 
        for i, label in enumerate(args.labels):
            if size is not None: 
                legend_labels.append(f'{label} word (n={str(size[i])})')
            else: 
                legend_labels.append(r'\textit{' + ' '.join([label, 'word']) + '}')
    
    return legend_labels


def set_color(args, data): 
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
            #colors = ['b','c']
            colors = ['midnightblue','dodgerblue']
        else:
            #colors = ['r','m']
            colors = ['darkred','orangered']
    elif rows==3:
        if args.prod:
            colors = ['b','c','limegreen']
        else:
            colors = ['r','m','darkorange']
    elif rows==4:
        colors = ['b','c','r','m']
    else:
        raise Exception('wrong dimension given to plotting function')

    return colors


def plot_data(args, data, pp, title=None, sem=None, subtitle=None, size=None):
    lags = np.arange(-2000, 2001, 25)
    # lags = np.arange(-5000, 5001, 25)

    fig, ax = plt.subplots()
    colors = set_color(args, data)
    ax.set_prop_cycle(color=colors)

    ax.plot(lags, data.T, linewidth=0.75)
    if size is not None: 
        #legend_labels = [f'correct class (n={size[0]})',f'incorrect class (actual word, n={size[1]})',f'incorrect class (predicted word, n={size[2]})']
        legend_labels = [f'GloVe embedding (n={size[0]})',f'GPT-2 embedding (n={size[1]})']
    else: 
        # legend_labels = ['correct class', 'incorrect class (actual word)', 'incorrect class (predicted word)']
        legend_labels = ['GloVe embedding', 'GPT-2 embedding']
    # ax.legend(legend_labels, frameon=False, prop={'size': 9})
    ax.set(xlabel=r'\textit{lag (ms)}',
           ylabel=r'\textit{correlation}')
    ax.set_ylim(-0.05, 0.30)
    ax.vlines(0, -0.05, 0.30, 'k', linestyles='dashed', linewidth=1)

    
    if subtitle is not None: 
        plt.suptitle(title, x = .52, y = .96)
        plt.title(subtitle, size='medium')
    else: 
        plt.title(title)
    
    if sem is not None: 
        for i in range(0, data.shape[0]):
            plt.fill_between(lags, data.T[:,i]-sem.T[:,i], data.T[:,i]+sem.T[:,i], 
                                color=colors[i], alpha=0.3)

    pp.savefig(fig)
    plt.close()


def plot_average_correlations_multiple(pp, prod_corr_mean, comp_corr_mean,
                                       prod_sem, comp_sem, args):

    sems = np.vstack([prod_sem, comp_sem])
    data = np.vstack([prod_corr_mean, comp_corr_mean])
    plot_data(args, data, pp, r'\textit{Average GloVe Encoding}',
                sems, r'(NY625, conversation 5)')

def plot_average_correlations_one(pp, prod_corr_mean, prod_sem, args):

    sem = np.vstack(prod_sem)
    data = np.vstack(prod_corr_mean)
    plot_data(args, data, pp, f'Average GloVe Encoding', 
                sem, r'(NY625, conversation 5)')

def plot_individual_correlation_multiple(pp, prod_corr, comp_corr, prod_list,
                                         args):
    prod_list = [item.replace('_', '\_') for item in prod_list]
    prod_corr = np.moveaxis(prod_corr, [0, 1, 2], [1, 0, 2])
    comp_corr = np.moveaxis(comp_corr, [0, 1, 2], [1, 0, 2])

    for prod_row, comp_row, electrode_id in zip(prod_corr, comp_corr,
                                                prod_list):
        data = np.vstack([prod_row, comp_row])
        plot_data(args, data, pp, electrode_id)


def plot_individual_correlation_one(pp, prod_corr, prod_list, sub_list, size_list, args):

    prod_list = [item.replace('_', '\_') for item in prod_list]
    prod_corr = np.moveaxis(prod_corr, [0, 1, 2], [1, 0, 2])

    for prod_row, electrode_id, sub, size in zip(prod_corr, prod_list, sub_list, size_list):
        data = prod_row
        plot_data(args, data, pp, electrode_id, subtitle=f'NY{sub}', size=size)



if __name__ == '__main__':
    # Parse input arguments
    args = parse_arguments()

    args.output_pdf = os.path.join(os.getcwd(), 'results', 'enc', 'figures',
                                   args.output_file_name + '.pdf')
    args.electrode_file = os.path.join(os.getcwd(), 'data', str(args.sid),
                                       str(args.sid) + '_electrode_names.pkl')

    # Results folders to be plotted
    results_dirs = [
        glob.glob(os.path.join(os.getcwd(), 'results', 'enc', directory))[0]
        for directory in args.input_directory
    ]

    if args.prod:
        prod_corr, prod_corr_mean, prod_list, sub_list, size_list, prod_sem = extract_correlations(
            args, results_dirs, 'prod')

    if args.comp:
        comp_corr, comp_corr_mean, comp_list, sub_list, size_list, comp_sem = extract_correlations(
            args, results_dirs, 'comp')

    pp = PdfPages(args.output_pdf)

    # plot both production and comprehension
    if args.prod & args.comp:
        plot_average_correlations_multiple(pp, prod_corr_mean, comp_corr_mean,
                                                prod_sem, comp_sem, args)

        plot_individual_correlation_multiple(pp, prod_corr, comp_corr,
                                                prod_list, args)
    
    # only plot production 
    elif args.prod: 
        plot_average_correlations_one(pp, prod_corr_mean, prod_sem, args)
        plot_individual_correlation_one(pp, prod_corr, prod_list,
                                                        sub_list, size_list, args)

    # only plot comprehension
    elif args.comp: 
        plot_average_correlations_one(pp, comp_corr_mean, comp_sem, args)
        plot_individual_correlation_one(pp, comp_corr, comp_list, 
                                                        sub_list, size_list, args)

    pp.close()
