import numpy as np
import pylab as plt
import pandas as pd
import os
import glob

#import network_util as nu
from lasso import matthew_coeff, select_conf_mat

syn_type_all = ['exc', 'inh', 'all']


def statistics_mcc(fpath_csv, fn_csv='confusion_mat.csv'):
    """Plot it."""
    fname_csv_all = glob.glob(os.path.join(fpath_csv, '**', fn_csv),
                              recursive=True)
    dout = {}
    dout['fpath'] = []
    for syn_type in syn_type_all:
        dout['MCCpeak_%s' % syn_type] = []
        dout['lamb_MCCpeak_%s' % syn_type] = []
    for fname_csv in fname_csv_all:
        df = pd.read_csv(fname_csv)
        dout['fpath'].append(os.path.dirname(fname_csv).replace(fpath_csv, ''))
        lambd = 1 / df['regularization_strength'].to_numpy()[::-1]
        for syn_type in syn_type_all:
            mcc_syn = matthew_coeff(select_conf_mat(df, syn_type))[::-1]
            idx = np.argmax(mcc_syn)
            dout['MCCpeak_%s' % syn_type].append(mcc_syn[idx])
            dout['lamb_MCCpeak_%s' % syn_type].append(lambd[idx])
    return pd.DataFrame(dout)
