import pylab as plt
import network_analysis as na
import network_util as nu
import pandas as pd
import numpy as np
import glob
import os
from lasso import matthew_coeff, select_conf_mat

fpath = '/home/tnieus/Projects/RESULTS/Lasso/paper/20nrn_16exc_4inh/0001'
figsize = (4, 4)

colormap = 'jet'
fs_lab = 14
fs_ticks = 12
fs_legend = 12
ms = 1  # markersize
fig_type = '.png'


def plot_dale(fpath, plot_legend=False, plot_fig=False):
    """Plot dale."""
    df = na.batch_dale(fpath)
    if plot_fig:
        plt.figure(figsize=figsize)
    plt.plot(df['lambda'], df['dale_precision_inh'], 'bo-', lw=2,
             markersize=ms, label='inhibitory')
    plt.plot(df['lambda'], df['dale_precision_exc'], 'ro-', lw=2,
             markersize=ms, label='excitatory')
    plt.xticks(fontsize=fs_ticks)
    plt.yticks(fontsize=fs_ticks)
    plt.xlabel(r'$\lambda$', fontsize=fs_lab)
    plt.ylabel('dale precision', fontsize=fs_lab)
    if plot_legend:
        plt.legend(loc=0, fontsize=fs_legend)
    plt.tight_layout(pad=1)


def compute_dale(fpath, idx_plateau=0):
    """Analyze dale and MCCpeak."""
    df = na.batch_dale(fpath)
    lambda_val = df['lambda'].to_numpy()
    dale_exc_1 = np.where(df['dale_precision_exc'] == 1)[0]
    dale_inh_1 = np.where(df['dale_precision_inh'] == 1)[0]
    return lambda_val[dale_exc_1[idx_plateau]], lambda_val[dale_inh_1[idx_plateau]]


def batch_compute_dale(fpath_base, num_arr=np.arange(1, 11), idx_plateau=0):
    """Batch compute dale."""
    dout = {}
    dout['fname'] = []
    dout['lambda_dale_exc'] = []
    dout['lambda_dale_inh'] = []
    dout['MCC(lambda_dale_exc)'] = []
    dout['MCC(lambda_dale_inh)'] = []

    dout['lambda_MCC_exc'] = []
    dout['lambda_MCC_inh'] = []
    dout['MCCpeak_exc'] = []
    dout['MCCpeak_inh'] = []

    for num in num_arr:
        fpath = os.path.join(fpath_base, nu.snum(num))
        lamb_dale_exc, lamb_dale_inh = compute_dale(fpath, idx_plateau)
        dout['fname'].append('/'+nu.snum(num))
        dout['lambda_dale_exc'].append(lamb_dale_exc)
        dout['lambda_dale_inh'].append(lamb_dale_inh)
        # get the corresponding MCC
        fn_csv = os.path.join(fpath_base, nu.snum(num), 'confusion_mat.csv')
        df = pd.read_csv(fn_csv)
        lambd = 1 / df['regularization_strength'].to_numpy()
        mc_exc = matthew_coeff(select_conf_mat(df, 'exc'))
        mc_inh = matthew_coeff(select_conf_mat(df, 'inh'))
        idx_exc = np.where(lambd == lamb_dale_exc)[0][0]
        idx_inh = np.where(lambd == lamb_dale_inh)[0][0]
        dout['MCC(lambda_dale_exc)'].append(mc_exc[idx_exc])
        dout['MCC(lambda_dale_inh)'].append(mc_inh[idx_inh])
        dout['MCCpeak_exc'].append(np.max(mc_exc))
        dout['MCCpeak_inh'].append(np.max(mc_inh))
        dout['lambda_MCC_exc'].append(lambd[np.argmax(mc_exc)])
        dout['lambda_MCC_inh'].append(lambd[np.argmax(mc_inh)])

    return pd.DataFrame(dout)


def group_curves(fpath_tmp, idx_rng=np.arange(1, 11)):
    """Plot it.

    fpath_tmp = '/home/tnieus/Projects/RESULTS/Lasso/paper/20nrn_16exc_4inh/0002/added_noise/inhibitory_as_excitatory_in_sim/%s'
    """
    mat = {}
    for syn_type in ['exc', 'inh']:
        mat[syn_type] = []
        for idx in idx_rng:
            df = na.batch_dale(fpath_tmp % nu.snum(idx))
            data = df['dale_precision_%s' % syn_type].to_numpy()
            mat[syn_type].append(data)
    mat['lambda'] = df['lambda'].to_numpy()
    return mat


def plot_group(fpath, idx_rng=np.arange(1, 11)):
    """Plot group.

    '/home/tnieus/Projects/RESULTS/Lasso/paper/20nrn_16exc_4inh/0002/added_noise/inhibitory_as_excitatory_in_sim/%s'

    """
    plt.figure(figsize=figsize)
    mat = group_curves(fpath, idx_rng)
    for k in range(10):
        if k == 0:
            plt.plot(mat['lambda'], mat['exc'][k], 'r-', label='exc')
            plt.plot(mat['lambda'], mat['inh'][k], 'b-', label='inh')
        else:
            plt.plot(mat['lambda'], mat['exc'][k], 'r-')
            plt.plot(mat['lambda'], mat['inh'][k], 'b-')
    plt.legend(loc=0, fontsize=fs_legend)
    plt.xticks(fontsize=fs_ticks)
    plt.yticks(fontsize=fs_ticks)
    plt.xlabel(r'$\lambda$', fontsize=fs_lab)
    plt.ylabel('dale precision', fontsize=fs_lab)
    plt.tight_layout(pad=1)


def summary_dale_plot(fpath_csv, fn_csv='dale.csv'):
    """Search dale.csv files."""
    from math import sqrt
    fname_csv_all = glob.glob(os.path.join(fpath_csv, '**', fn_csv),
                              recursive=True)
    out = {}
    out2 = {}
    for measure in ['dale', 'peak']:
        out[measure] = {}
        out2[measure] = {}
        for syn_type in ['exc', 'inh']:
            out[measure][syn_type] = {}
            out[measure][syn_type]['mean'] = []
            out[measure][syn_type]['err'] = []
            out2[measure][syn_type] = {}
            out2[measure][syn_type]['mean'] = []
            out2[measure][syn_type]['err'] = []

    for fname_csv in fname_csv_all:
        df = pd.read_csv(fname_csv)
        for syn_type in ['exc', 'inh']:
            mcc_peak = df['MCCpeak_%s' % syn_type].to_numpy()
            mean_mcc_peak = np.mean(mcc_peak)
            err_mcc_peak = np.std(mcc_peak) / sqrt(len(mcc_peak))
            mcc_dale = df['MCC(lambda_dale_%s)' % syn_type].to_numpy()
            mean_mcc_dale = np.mean(mcc_dale)
            err_mcc_dale = np.std(mcc_dale) / sqrt(len(mcc_dale))
            # update out list
            out['peak'][syn_type]['mean'].append(mean_mcc_peak)
            out['peak'][syn_type]['err'].append(err_mcc_peak)
            out['dale'][syn_type]['mean'].append(mean_mcc_dale)
            out['dale'][syn_type]['err'].append(err_mcc_dale)
            #
            mcc_peak = df['lambda_MCC_%s' % syn_type].to_numpy()
            mean_mcc_peak = np.mean(mcc_peak)
            err_mcc_peak = np.std(mcc_peak) / sqrt(len(mcc_peak))
            mcc_dale = df['lambda_dale_%s' % syn_type].to_numpy()
            mean_mcc_dale = np.mean(mcc_dale)
            err_mcc_dale = np.std(mcc_dale) / sqrt(len(mcc_dale))
            # update out2 list
            out2['peak'][syn_type]['mean'].append(mean_mcc_peak)
            out2['peak'][syn_type]['err'].append(err_mcc_peak)
            out2['dale'][syn_type]['mean'].append(mean_mcc_dale)
            out2['dale'][syn_type]['err'].append(err_mcc_dale)

    # plot 1
    plt.figure(figsize=figsize)
    xnoise = np.linspace(0, 0.01, 10)  # random.rand(len(mcc_dale)) / 50
    # exc
    plt.errorbar(x=out['peak']['exc']['mean'] + xnoise,
                 y=out['dale']['exc']['mean'],
                 xerr=out['peak']['exc']['err'],
                 yerr=out['dale']['exc']['err'],
                 ecolor='r', elinewidth=1, color='r', lw=0, fmt='o',
                 label='exc')
    # inh
    plt.errorbar(x=out['peak']['inh']['mean'] + xnoise,
                 y=out['dale']['inh']['mean'],
                 xerr=out['peak']['inh']['err'],
                 yerr=out['dale']['inh']['err'],
                 ecolor='b', elinewidth=1, color='b', lw=0, fmt='o',
                 label='inh')

    plt.legend(loc=0, fontsize=fs_legend)
    plt.xticks(fontsize=fs_ticks)
    plt.yticks(fontsize=fs_ticks)
    plt.xlabel('MCC peak', fontsize=fs_lab)
    plt.ylabel('MCC dale', fontsize=fs_lab)
    plt.tight_layout(pad=1)

    # plot 2
    plt.figure(figsize=figsize)
    plt.errorbar(x=out2['peak']['exc']['mean'],
                 y=out2['dale']['exc']['mean'],
                 xerr=out2['peak']['exc']['err'],
                 yerr=out2['dale']['exc']['err'],
                 ecolor='r', elinewidth=1, color='r', lw=0, fmt='o',
                 label='exc')
    # xnoise = np.random.rand(len(mcc_dale)) / 50
    plt.errorbar(x=out2['peak']['inh']['mean'],
                 y=out2['dale']['inh']['mean'],
                 xerr=out2['peak']['inh']['err'],
                 yerr=out2['dale']['inh']['err'],
                 ecolor='b', elinewidth=1, color='b', lw=0, fmt='o',
                 label='inh')
    xymin = min(np.min(out2['peak']['exc']['mean']),
                np.min(out2['peak']['inh']['mean']),
                np.min(out2['dale']['exc']['mean']),
                np.min(out2['dale']['inh']['mean']))

    xymax = max(np.max(out2['peak']['exc']['mean']),
                np.max(out2['peak']['inh']['mean']),
                np.max(out2['dale']['exc']['mean']),
                np.max(out2['dale']['inh']['mean']))

    plt.legend(loc=0, fontsize=fs_legend)
    plt.xticks(fontsize=fs_ticks)
    plt.yticks(fontsize=fs_ticks)
    plt.xlabel(r'$\lambda MCC\ peak$', fontsize=fs_lab)
    plt.ylabel(r'$\lambda MCC\ dale$', fontsize=fs_lab)
    plt.plot([xymin, xymax], [xymin, xymax], 'g--')
    plt.tight_layout(pad=1)
    return out
