import network_util as nu

import numpy as np
import os
import pandas as pd


def export_mat(fpath_reg, fpath_out):
    """Do it."""
    reg_arr = nu.get_regularization_factor(fpath_reg)
    if not(os.path.isdir(fpath_out)):
        os.makedirs(fpath_out)
    dout = {}
    dout['lambda'] = np.sort(1 / reg_arr)
    fn_out_lambda = os.path.join(fpath_out, 'lambda_list.csv')
    df = pd.DataFrame(dout)
    df.to_csv(fn_out_lambda, sep='\t', index=False)
    for reg in reg_arr:
        lamb = 1/reg
        fn_data = os.path.join(fpath_reg, 'reg_%g' % reg, 'RSmat_lasso.npy')
        beta = nu.load_dict(fn_data)['beta']
        # +1
        r_1, c_1 = np.where(beta == 1)
        dout = {}
        dout['from'] = r_1
        dout['to'] = c_1
        df = pd.DataFrame(dout)
        fn_out_one = os.path.join(fpath_out, 'lambda=%g_exc.csv' % lamb)
        df.to_csv(fn_out_one, sep='\t', index=False)
        # -1
        r_m1, c_m1 = np.where(beta == -1)
        dout = {}
        dout['from'] = r_m1
        dout['to'] = c_m1
        df = pd.DataFrame(dout)
        fn_out_minusone = os.path.join(fpath_out, 'lambda=%g_inh.csv' % lamb)
        df.to_csv(fn_out_minusone, sep='\t', index=False)
    # structural
    struct = nu.load_dict(os.path.join(fpath_reg, 'output.npy'))
    dout = {}
    conn_mat = np.array(struct['params_netw']['conn_mat'])
    dout['from'] = conn_mat[:, 0]
    dout['to'] = conn_mat[:, 1]
    df = pd.DataFrame(dout)
    df.to_csv(os.path.join(fpath_out, 'struct_connmat.csv'), sep='\t', index=False)
    # neuron type
    exc_idx = list(struct['params_netw']['exc'])
    inh_idx = list(struct['params_netw']['inh'])
    nrn_type = list(np.repeat(1, len(exc_idx))) + list(np.repeat(-1, len(inh_idx)))
    dout = {}
    dout['neuron_idx'] = exc_idx + inh_idx
    dout['neuron_type'] = list(np.repeat(1, len(exc_idx))) + list(np.repeat(-1, len(inh_idx)))
    df = pd.DataFrame(dout)
    df.to_csv(os.path.join(fpath_out, 'struct_neuron_type.csv'), sep='\t', index=False)
