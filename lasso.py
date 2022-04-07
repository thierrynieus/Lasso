import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, block_diag
from sklearn.linear_model import LogisticRegression

import os
import pandas as pd

from network_util import load_dict

resultsfolder = '/home/tnieus/Projects/RESULTS/Lasso/'
configfolder = '/home/tnieus/Projects/CODE/Lasso/config/'
figsize = (10, 10)

params_lasso = {'regularization_strength': 1, 'rel_path_results': 'test',
                'fname_RSmat': 'RSmat.npy',
                'fname_lasso': 'RSmat_lasso.npy',
                'skip_existent': True, 'max_iter': 3000, 'n_jobs': None,
                'tol': 0.0001, 'solver': 'saga', 'warm_start': False}

"""
'solver': 'liblinear' good for small datasets, 'saga' for big datasets, 'saga'
used also for Ridge and ElasticNet
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
"""

params_conf_mat = {'regularization_strength': 0.1, 'rel_path_results': 'test',
                   'fname_lasso': 'RSmat_lasso.npy',
                   'fname_netw': 'params_netw.npy',
                   'rel_path_config': '20nrn_16exc_4inh/0000',
                   'fname_conf_mat': 'RSmat_conf_mat.npy'}

params_roc = {'rel_path_results': 'test',
              'fname_conf_mat': 'RSmat_conf_mat.npy',
              'reg_vect': [0.0001, 0.001, 0.005, 0.0075, 0.01, 0.025, 0.05,
                           0.075, 0.1, 0.2, 0.3, 0.4, 0.5]}


def calc_mat_processes(fn_data, fn_out, time_trim=(100, 5000), dt_dis=1):
    """Compute the matrices of the internal (S) and external (R) events.

    data[k]['spikes_nrn']
    data[k]['spikes_noise']
    conn a list of connections (src,dst)
    time_trim discard outside interval
    dt_dist discretization
    """
    def discretize(spk, tmin, tmax, dt_dis):
        spk1 = spk.copy()
        spk1 = spk1[spk1 < tmax]
        spk1 = spk1[spk1 > tmin] - tmin
        return np.trunc(spk1 / dt_dis).astype(int)

    print(fn_data)
    data = load_dict(os.path.join(resultsfolder, fn_data))
    # config = load_dict(os.path.join(configfolder, fn_cfg))

    conn_mat = np.array(data['params_netw']['conn_mat'])
    if 'num_neurons' in data:
        num_neurons = data['num_neurons']
    else:
        # in old fn_data num_neurons was not available!
        num_neurons = len([x for x in data if isinstance(x, int)])

    tmin, tmax = time_trim
    ntime = int((tmax - tmin) / dt_dis)
    mat_ext = np.zeros((num_neurons, ntime), dtype=np.int8)
    mat_int = np.zeros((num_neurons, ntime), dtype=np.int8)
    # mat_ext_dic = {} # lists, dictionary for sparse representation

    # external (spikes)
    print('num neurons: ', num_neurons)
    for k in range(num_neurons):
        idx = discretize(data[k]['spikes_nrn'], tmin, tmax, dt_dis)
        # print(k,idx, idx.max(),len(idx))
        mat_ext[k, idx] = 1

    # internal (epsp, ipsp, noise epsp)
    for k in range(num_neurons):
        #  process endogeneous noise epsp events
        idx = discretize(data[k]['spikes_noise'], tmin, tmax, dt_dis)
        mat_int[k, idx] = 1
        #  all inputs to neuron k
        idx_in_conn = np.where(conn_mat[:, 1] == k)[0]
        for idx_in in idx_in_conn:
            nrn_in = conn_mat[idx_in, 0]  # nrn_in to k
            sign = 1 if nrn_in in data['params_netw']['exc'] else -1
            # conn_mat[idx_in, 0] connections to neuron k
            idx_nz = np.where(mat_ext[nrn_in, :])[0]
            mat_int[k, idx_nz] = sign
    dout = {'R': mat_int, 'S': mat_ext, 'time_trim': time_trim,
            'dt_dis': dt_dis}
    np.save(os.path.join(resultsfolder, fn_out), dout)


def lasso(params):
    """Perform logistic regression with lasso penalization.

    note:
        add possibility to load a model !!!
    """
    import time
    t0 = time.time()
    # Inverso del coefficiente di regolarizzazione (lambda^{-1})
    reg_strength = params['regularization_strength']

    # output folder
    fpath_out = os.path.join(resultsfolder, params['rel_path_results'],
                             'reg_%g' % reg_strength)
    if not(os.path.isdir(fpath_out)):
        os.mkdir(fpath_out)
    fname_lasso = os.path.join(fpath_out, params['fname_lasso'])
    # skip the analysis if skip_existent is TRUE and file already exists
    if os.path.isfile(fname_lasso) & params['skip_existent']:
        return None

    # load data
    fn = os.path.join(resultsfolder, params['rel_path_results'],
                      params['fname_RSmat'])
    data = np.load(fn, allow_pickle=1).item()
    R = data['R']
    S = data['S']

    n, m = S.shape   # number of neurons (N) and observations (M)
    y_vect = R.reshape(-1, 1)  # response vector

    s_mat_transpose = np.transpose(S)
    # define block diagonal matrix based on S
    x_block_mat = block_diag([coo_matrix(s_mat_transpose) for i in range(n)])

    nm = n * m
    omega0 = np.count_nonzero(y_vect) / nm
    omega1 = (nm - np.count_nonzero(y_vect == 1)) / nm
    omegam1 = (nm - np.count_nonzero(y_vect == -1)) / nm
    print('omega(0)=%g omega(1)=%g omega(-1)=%g' % (omega0, omega1, omegam1))

    model = LogisticRegression(
        penalty='l1',  # equal to l1_ratio=1
        class_weight={0: omega0, 1: omega1, -1: omegam1},
        # if sum weights<>1 change penalization parameter C
        solver=params['solver'],
        multi_class='multinomial',
        max_iter=params['max_iter'],
        n_jobs=params['n_jobs'],
        tol=params['tol'],
        warm_start=params['warm_start'],
        C=reg_strength)  # reg_strength is the inverse of lambda

    model.fit(x_block_mat, y_vect.ravel())  # apply coo_matrix on y_vect ?
    """
    # access as a dict
    dx = model.__dict__
    dx.keys()
    """
    #  save the model
    fname_model = os.path.join(fpath_out, 'model.npy')
    np.save(fname_model, model)

    # coefficients of the model
    alpha = model.coef_
    #  alpha relative to the classes -1, 0, 1
    alpha_minus_one = np.transpose(alpha[0].reshape(n, n))
    alpha_zero = np.transpose(alpha[1].reshape(n, n))
    alpha_one = np.transpose(alpha[2].reshape(n, n))

    # Coefficienti beta
    beta_uno = np.zeros(shape=(n, n), dtype='int8')
    beta_zero = np.zeros(shape=(n, n), dtype='int8')
    beta_meno_uno = np.zeros(shape=(n, n), dtype='int8')

    #  <--- REFINE FROM HERE ..

    """ Selezioniamo i valori positivi (in alternativa selezioniamo i valori
    del range 30% più alto) """
    coef_threshold = 0
    beta_meno_uno[alpha_minus_one > coef_threshold] = 1
    beta_zero[alpha_zero > coef_threshold] = 1
    beta_uno[alpha_one > coef_threshold] = 1

    '''
    beta dev'essere -1 per beta_meno_uno = 1
        e dev'essere 1 per beta_meno_uno = 0 and beta_uno = 1
    beta = beta_uno - beta_meno_uno
    beta = np.zeros(shape=(n, n))
    '''
    beta = np.zeros(shape=(n, n), dtype='int8')
    beta[beta_meno_uno == 1] = -1
    beta[np.logical_and(beta_meno_uno == 0, beta_uno == 1)] = 1

    ### ---> .. TO HERE

    dout = {}
    dout['beta'] = beta
    dout['regularization_strength'] = reg_strength
    dout['computation time (s)'] = time.time()-t0
    # save data
    np.save(fname_lasso, dout)


def confusion_matrix(params):
    """Build confusion matrix."""
    fname_netw = os.path.join(configfolder, params['rel_path_config'],
                              params['fname_netw'])

    folder_reg = 'reg_%g' % params['regularization_strength']
    fname_lasso = os.path.join(resultsfolder, params['rel_path_results'],
                               folder_reg, params['fname_lasso'])
    data_netw = np.load(fname_netw, allow_pickle=1).item()
    data_lasso = np.load(fname_lasso, allow_pickle=1).item()

    # build adjancency matrix
    src, dst = np.array(data_netw['conn_mat']).T  # -1, 1, 0
    idx_exc = np.isin(src, data_netw['exc'])
    idx_inh = np.isin(src, data_netw['inh'])
    n = data_netw['num_neurons']
    adj_mat = np.zeros((n, n), dtype='int8')
    adj_mat[src[idx_exc], dst[idx_exc]] = 1
    adj_mat[src[idx_inh], dst[idx_inh]] = -1

    beta = data_lasso['beta']
    reg_strength = data_lasso['regularization_strength']

    values = [-1, 0, 1]
    count_ab = {}
    for k in values:
        for j in values:
            str_kj = '(%d,%d)' % (k, j)
            count_ab[str_kj] = np.count_nonzero(np.logical_and(adj_mat == k,
                                                               beta == j))
    fp_exc = count_ab['(0,1)'] + count_ab['(-1,1)']
    fp_inh = count_ab['(0,-1)'] + count_ab['(1,-1)']
    fn_exc = count_ab['(1,0)'] + count_ab['(1,-1)']
    fn_inh = count_ab['(-1,0)'] + count_ab['(-1,1)']

    nexc = 0
    for nrn in data_netw['exc']:
        nexc += np.count_nonzero(src == nrn)
    ninh = 0
    for nrn in data_netw['inh']:
        ninh += np.count_nonzero(src == nrn)
    nzero = n**2 - nexc - ninh

    tp_exc = nexc - fn_exc
    tp_inh = ninh - fn_inh

    # tn_exc = nzero + tp_inh + count_ab['(-1,0)'] - count_ab['(0,-1)']
    tn_exc = nzero + tp_inh + count_ab['(-1,0)'] - count_ab['(0,1)']
    tn_inh = nzero + tp_exc + count_ab['(1,0)'] - count_ab['(0,-1)']

    #
    tp = tp_exc + tp_inh    # tp is just the sum of exc and inh tp
    fp = count_ab['(0,1)'] + count_ab['(0,-1)']
    fn = count_ab['(1,0)'] + count_ab['(-1,0)']
    tn = count_ab['(0,0)']

    # save results
    dout = {}
    dout['regularization_strength'] = reg_strength
    # all
    dout['tp'] = tp
    dout['tn'] = tn
    dout['fp'] = fp
    dout['fn'] = fn
    # exc
    dout['tp_exc'] = tp_exc
    dout['tn_exc'] = tn_exc
    dout['fp_exc'] = fp_exc
    dout['fn_exc'] = fn_exc
    # inh
    dout['tp_inh'] = tp_inh
    dout['tn_inh'] = tn_inh
    dout['fp_inh'] = fp_inh
    dout['fn_inh'] = fn_inh

    fname_conf_mat = os.path.join(resultsfolder, params['rel_path_results'],
                                  folder_reg,
                                  params['fname_conf_mat'])

    np.save(fname_conf_mat, dout)

    return dout


def plot_roc(params):
    """Plot Receiver Operative Curve."""
    def ratio(a, b):
        s = a + b
        return a/s if s else 1.
    # sensitivity
    sensitivity = []
    sensitivity_exc = []
    sensitivity_inh = []
    # specificity
    specificity = []
    specificity_exc = []
    specificity_inh = []
    #
    for reg in params['reg_vect']:
        fname_lasso = os.path.join(resultsfolder, params['rel_path_results'],
                                   'reg_%g' % reg, params['fname_conf_mat'])
        d = np.load(fname_lasso, allow_pickle=1).item()
        # all
        sensitivity.append(ratio(d['tp'], d['fn']))
        specificity.append(ratio(d['tn'], d['fp']))
        print(reg, 1-specificity[-1], sensitivity[-1])
        # exc
        sensitivity_exc.append(ratio(d['tp_exc'], d['fn_exc']))
        specificity_exc.append(ratio(d['tn_exc'], d['fp_exc']))
        # inh
        sensitivity_inh.append(ratio(d['tp_inh'], d['fn_inh']))
        specificity_inh.append(ratio(d['tn_inh'], d['fp_inh']))

    plt.figure(figsize=(6, 6))
    # all
    oneminus_specificity = 1-np.array(specificity)
    plt.plot(oneminus_specificity, sensitivity, 'ks-', markersize=10,
             label='all', lw=2)
    # exc
    oneminus_specificity_exc = 1-np.array(specificity_exc)
    plt.plot(oneminus_specificity_exc, sensitivity_exc, 'ro--', markersize=10,
             label='excitatory')
    # inh
    oneminus_specificity_inh = 1-np.array(specificity_inh)
    plt.plot(oneminus_specificity_inh, sensitivity_inh, 'bo--', markersize=10,
             label='inhibitory')
    # decorate
    plt.xlabel('false positive rate', fontsize=16)
    plt.ylabel('true positive rate', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.tight_layout(pad=1)

    fname_png = os.path.join(resultsfolder, params['rel_path_results'],
                             'plot_ROC.png')
    plt.savefig(fname_png)


def plot_matthew_vs_regularization(params_roc, label='', lw=2, fignum=1):
    """Plot Matthew coefficient versus regularization.

    params_roc

    """
    def copy_data(df, dtype='all'):
        str_add = ''  # default
        if dtype == 'exc':
            str_add = '_exc'
        if dtype == 'inh':
            str_add = '_inh'
        lun = len(df)
        conf_mat = np.zeros((lun, 4), dtype='int')
        conf_mat[:, 0] = df['tp%s' % str_add].to_numpy()
        conf_mat[:, 1] = df['fp%s' % str_add].to_numpy()
        conf_mat[:, 2] = df['tn%s' % str_add].to_numpy()
        conf_mat[:, 3] = df['fn%s' % str_add].to_numpy()
        return conf_mat

    fname_lasso = os.path.join(resultsfolder, params_roc['rel_path_results'],
                               'confusion_mat.csv')
    df = pd.read_csv(fname_lasso)
    # plot
    plt.figure(fignum, figsize=figsize)

    # all
    mc = matthews_coeff(copy_data(df, 'all'))
    plt.plot(df['regularization_strength'], mc, 'ks-', markersize=10,
             label='all %s' % label, lw=lw)

    # exc
    mc = matthews_coeff(copy_data(df, 'exc'))
    plt.plot(df['regularization_strength'], mc, 'ro--', markersize=10,
             label='excitatory %s' % label, lw=lw)
    # inh
    mc = matthews_coeff(copy_data(df, 'inh'))
    plt.plot(df['regularization_strength'], mc, 'bo--', markersize=10,
             label='inhibitory %s' % label, lw=lw)
    plt.legend(loc=0, fontsize=14)
    #  decorate
    plt.xlabel('regularization strength', fontsize=16)
    plt.ylabel('Matthew coefficient', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.tight_layout(pad=1)
    fname_png = os.path.join(resultsfolder, params_roc['rel_path_results'],
                             'plot_Matthew.png')
    plt.savefig(fname_png)


def run_all(rel_path_config='0000', rel_path_results='', time_trim=(100, 5000),
            dt_dis=1):
    """Perform all steps.

    npte: rel_path_config potrebbe essere letto dal file di output ..
    """
    # lasso
    params_lasso['rel_path_results'] = rel_path_results
    params_lasso['fname_RSmat'] = 'RSmat.npy'
    params_lasso['fname_lasso'] = 'RSmat_lasso.npy'

    # confusion mat
    params_conf_mat['rel_path_results'] = rel_path_results
    params_conf_mat['fname_lasso'] = 'RSmat_lasso.npy'
    params_conf_mat['rel_path_config'] = rel_path_config
    params_conf_mat['fname_conf_mat'] = 'RSmat_conf_mat.npy'

    # roc
    params_roc['rel_path_results'] = rel_path_results
    params_roc['fname_conf_mat'] = 'RSmat_conf_mat.npy'

    # step 0 R,S mat
    calc_mat_processes('%s/output.npy' % (rel_path_results),
                       '%s/RSmat.npy' % (rel_path_results),
                       time_trim=time_trim, dt_dis=dt_dis)
    keys = ['regularization_strength', 'tp', 'tn', 'fp', 'fn', 'tp_exc',
            'tn_exc', 'fp_exc', 'fn_exc', 'tp_inh', 'tn_inh', 'fp_inh',
            'fn_inh']
    dict_out = {}
    for key in keys:
        dict_out[key] = []
    for reg in params_roc['reg_vect']:
        # lasso regularization (step 1)
        print(reg)
        params_lasso['regularization_strength'] = reg
        lasso(params_lasso)
        # confusion matrix (step 2)
        params_conf_mat['regularization_strength'] = reg
        dout = confusion_matrix(params_conf_mat)
        for key in keys:
            dict_out[key].append(dout[key])
    df = pd.DataFrame(dict_out)
    df['tp+tn+fp+fn'] = df['tp'] + df['tn'] + df['fp'] + df['fn']
    df['(tp+tn+fp+fn)_exc'] = df['tp_exc'] + df['tn_exc'] + df['fp_exc'] + \
        df['fn_exc']
    df['(tp+tn+fp+fn)_inh'] = df['tp_inh'] + df['tn_inh'] + df['fp_inh'] + \
        df['fn_inh']
    conn = load_dict(os.path.join(configfolder,
                                  '%s/params_netw.npy' % rel_path_config))
    src = np.array(conn['conn_mat'])[:, 0]
    nexc = 0
    for nrn in conn['exc']:
        nexc += np.count_nonzero(src == nrn)
    ninh = 0
    for nrn in conn['inh']:
        ninh += np.count_nonzero(src == nrn)
    df['total_exc'] = nexc
    df['total_inh'] = ninh
    df.to_csv(os.path.join(resultsfolder, rel_path_results,
                           'confusion_mat.csv'))
    plot_roc(params_roc)


def subsample_output(fn_out='', num_nrn_sample=20):
    """Subsample existing results.

    fn_out: path to the output.npy file
    num_nrn_sample: number of neurons to sample

    notes:
        consider to distinguish the selection based on #exc/#inh neurons
    """
    d = np.load(fn_out, allow_pickle=1).item()
    num_neurons = d['params_neurons']['num_neurons']
    num_exc = d['params_neurons']['num_exc_neurons']
    #
    nrn_sample = np.sort(np.random.choice(np.arange(num_neurons),
                                          num_nrn_sample,
                                          replace=False))
    nrn_sample_lst = nrn_sample.tolist()
    num_exc_sample = np.where(nrn_sample < num_exc)[0][-1] + 1

    # build the new output.py
    dnew = {}
    # connections
    dnew['params_netw'] = {}
    dnew['params_netw']['conn_mat'] = []
    dnew['params_netw']['num_neurons'] = num_nrn_sample
    dnew['params_netw']['exc'] = np.arange(num_exc_sample)
    dnew['params_netw']['inh'] = num_exc_sample + np.arange(num_nrn_sample -
                                                            num_exc_sample)
    for src, dst in d['params_netw']['conn_mat']:
        if (src in nrn_sample_lst) & (dst in nrn_sample_lst):
            src_new = nrn_sample_lst.index(src)
            dst_new = nrn_sample_lst.index(dst)
            dnew['params_netw']['conn_mat'].append((src_new, dst_new))
    # spike trains
    for idx_nrn, nrn in enumerate(nrn_sample_lst):
        dnew[idx_nrn] = d[nrn]
    # parameters
    dnew['params_neurons'] = {}
    dnew['params_neurons']['num_neurons'] = num_nrn_sample
    dnew['params_neurons']['num_exc_neurons'] = num_exc_sample
    dnew['nrn_sample'] = nrn_sample  # the sampled neurons
    return dnew


def matthews_coeff(conf_mat):
    """Compute the Matthews cross-correlation coefficient.

    used on binary matrices

    reference:
    """
    mcc = np.zeros(conf_mat.shape[0])
    tp = conf_mat[:, 0]
    fp = conf_mat[:, 1]
    tn = conf_mat[:, 2]
    fn = conf_mat[:, 3]
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    idx = np.where(den == 0)[0]
    mcc[idx] = -1
    idx = np.where(den)[0]
    mcc[idx] = (tp[idx] * tn[idx] - fp[idx] * fn[idx]) / np.sqrt(den[idx])
    return mcc


def area_under_curve(fname_conf_mat):
    """Compute area under curve."""
    df = pd.read_csv(fname_conf_mat)
    n = len(df)
    roc = {}
    roc['all'] = np.zeros((n+1, 2))
    roc['exc'] = np.zeros((n+1, 2))
    roc['inh'] = np.zeros((n+1, 2))
    auc = {}
    for syn_type in ['all', 'exc', 'inh']:
        roc[syn_type][n, :] = 1  # ensures the ROC goes to (1,1)
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
