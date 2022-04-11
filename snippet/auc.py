import pandas as pd
import numpy as np


def area_under_curve(fname_conf_mat):
    """Compute area under curve.

    note:
        add (FPR,TPR)=(1,1)
    """
    df = pd.read_csv(fname_conf_mat)
    n = len(df)
    roc = {}
    roc['all'] = np.zeros((n+1, 2))
    roc['exc'] = np.zeros((n+1, 2))
    roc['inh'] = np.zeros((n+1, 2))
    auc = {}
    for syn_type in ['all', 'exc', 'inh']:
        roc[syn_type][n, :] = 1
        if syn_type == 'all':
            key_add = ''
        else:
            key_add = '_%s' % syn_type
        #
        tp = df['tp%s' % key_add].to_numpy()
        fp = df['fp%s' % key_add].to_numpy()
        tn = df['tn%s' % key_add].to_numpy()
        fn = df['fn%s' % key_add].to_numpy()

        roc[syn_type][:n, 0] = fp / (fp + tn)  # false positive rate
        roc[syn_type][:n, 1] = tp / (tp + fn)  # true positive rate

        # sort wrt to false positive rate
        idx_sort = np.argsort(roc[syn_type][:n, 0])
        roc[syn_type][:n, 0] = roc[syn_type][:n, 0][idx_sort]
        roc[syn_type][:n, 1] = roc[syn_type][:n, 1][idx_sort]

        # numerical integration
        auc[syn_type] = np.trapz(roc[syn_type][:, 1], roc[syn_type][:, 0])
    return auc
