import numpy as np
import os


def confusion_matrix(params):
    '''
    '''
    # load data

    fname_netw = os.path.join(configfolder, params['foldercfg'], params['fname_netw'])
    folder_reg = 'reg_%g' % params['regularization_strength']
    fname_lasso = os.path.join(resultsfolder, params['subfolder'], folder_reg,
                               params['fname_lasso'])
    data_netw = np.load(fname_netw, allow_pickle=1).item()
    data_lasso = np.load(fname_lasso, allow_pickle=1).item()

    # ADJ = dIN_adj['matriceADJ']
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
            str_kj ='(%d,%d)' % (k,j)
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

    tp_exc = nexc - fn_exc  # ok
    tp_inh = ninh - fn_inh  # ok

    tn_exc = nzero + tp_inh + count_ab['(-1,0)'] - count_ab['(0,-1)']
    tn_inh = nzero + tp_exc + count_ab['(1,0)'] - count_ab['(0,-1)']

    #  tn_exc = tp_inh + tp_0 + count_ab['(0,-1)'] + count_ab['(-1,0)']
    #  tn_inh = tp_exc + tp_0 + count_ab['(0,1)'] + count_ab['(1,0)']

    #
    tp = tp_exc + tp_inh    # OK tp_1 + tp_-1
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

    fname_conf_mat = os.path.join(resultsfolder, params['subfolder'],
                                  folder_reg,
                                  params['fname_conf_mat'])

    np.save(fname_conf_mat, dout)
    return dout
