import numpy as np
import pylab as plt
import pandas as pd
import os

from lasso import matthew_coeff, select_conf_mat

import network_util as nu

plt.ion()
colormap = 'jet'

fpath = '/home/tnieus/Projects/RESULTS/Lasso/' #paper/20nrn_16exc_4inh/'
# fpath_fig = os.path.join(fpath, 'noise/added_noise/fig')

folder_names = ['100nrn_80exc_20inh', '100nrn_80exc_20inh_spatial',
                '100nrn_80exc_20inh_spatial_4clusters']
folder_labels = ['random', 'Gauss', '4 clusters']

#number = len(noise_types)
figsize = (4, 4)
fs_lab = 14
fs_ticks = 12
fs_legend = 12
ms = 10  # markersize
fig_ext = 'png'
fpath_fig = '/home/tnieus/Projects/PAPERS/lasso/figures/figure_distance/'

def collect(fpath, idx_rng=np.arange(10)):
    """Collect data."""
    # trials
    mat = {}
    for label in folder_labels:
        mat[label] = {}
        for syn_type in ['all', 'exc', 'inh']:
            mat[label][syn_type] = []
        mat[label]['lambda'] = []
    #
    for folder, label in zip(folder_names, folder_labels):
        for idx in idx_rng:
            fn_csv = os.path.join(fpath, folder, nu.snum(idx),
                                  'confusion_mat.csv')
            df = pd.read_csv(fn_csv)
            lamb = 1/df['regularization_strength'].to_numpy()[::-1]
            mat[label]['lambda'].append(lamb)
            for syn_type in ['all', 'exc', 'inh']:
                mc = matthew_coeff(select_conf_mat(df, syn_type))[::-1]
                mat[label][syn_type].append(mc)
            print(label, idx, len(mc))
    return mat


def plot_collect(mat):
    """Plot collection."""
    col = {}
    col['Gauss'] = 'k'
    col['4 clusters'] = 'g'
    col['random'] = 'c'
    for syn_type in ['all', 'exc', 'inh']:
        plt.figure(figsize=figsize)
        for label in folder_labels:
            mc = np.array(mat[label][syn_type])
            mean_mc = mc.mean(axis=0)
            std_mc = mc.std(axis=0)
            plt.errorbar(x=mat[label]['lambda'][0], y=mean_mc, yerr=std_mc,
                         linewidth=2, elinewidth=2, label=label, c=col[label])
        plt.legend(loc=0, fontsize=14)
        plt.tight_layout(pad=1)
        plt.legend(loc=0, fontsize=fs_legend)
        plt.xlabel(r'$\lambda$', fontsize=fs_lab)
        plt.ylabel('MCC', fontsize=fs_lab)
        plt.xticks(fontsize=fs_ticks)
        plt.yticks(fontsize=fs_ticks)
        plt.tight_layout(pad=1)
        plt.savefig(os.path.join(fpath_fig,
                                 'MCC_vs_lambda_%s.%s' % (syn_type, fig_ext)))


"""Sparseness

sparse_lst = []
for i in range(10):
    fn = os.path.join(fpath, '100nrn_80exc_20inh', nu.snum(i),'output.npy')
    d = nu.load_dict(fn)
    if 'num_neurons' in d:
        num_neurons = d['num_neurons']
    else:
        num_neurons = len([x for x in d if isinstance(x, int)])
    sparseness = 100 * len(d['params_netw']['conn_mat']) / num_neurons **2
    sparse_lst.append(sparseness)
print(np.mean(sparse_lst), np.std(sparse_lst))


"""
