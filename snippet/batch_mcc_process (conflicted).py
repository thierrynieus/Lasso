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


def collect_mcc(fpath, idx_rng=np.arange(10)):
    """Collect data."""
    # trials
    mat_dict = {}
    for label in folder_labels:
        mat_dict[label] = {}
        for syn_type in ['all', 'exc', 'inh']:
            mat_dict[label][syn_type] = []
    #
    for folder, label in zip(folder_names, folder_labels):
        for idx in idx_rng:
            fn_csv = os.path.join(fpath, folder, nu.snum(idx),
                                  'confusion_mat.csv')
            df = pd.read_csv(fn_csv)
            for syn_type in ['all', 'exc', 'inh']:
                mc = matthew_coeff(select_conf_mat(df, syn_type))[::-1]
                mat_dict[label][syn_type].append(mc)
    return mat_dict
