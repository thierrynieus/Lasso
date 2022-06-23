import numpy as np
import pylab as plt
import os
import network_util as nu
import network_analysis as na


def graph_metrics(fpath_out, reg_vect=None, what={'sparseness': None}):
    """Compute graph metrics.

    what:
        - intergroup: list of groups (arrays)
        - in degree: no params
        - out degree: no params
        - sparseness: no params
    """
    data = nu.load_dict(os.path.join(fpath_out, 'output.npy'))
    num_neurons = data['num_neurons']
    conn_mat_struct = data['params_netw']['conn_mat']
    #  pos = [(x, y) for x, y in zip(data['params_netw']['x'], data['params_netw']['y'])]
    if reg_vect is None:
        reg_vect = nu.get_regularization_factor(fpath_out)

    f_graphs = {}
    for reg in reg_vect:
        f_graphs[reg] = {}
        beta = nu.load_dict(os.path.join(fpath_out, 'reg_%g' % reg,
                                         'RSmat_lasso.npy'))['beta']
        # functional graph
        r_all, c_all = np.where(beta)
        f_graphs[reg]['conn_mat'] = [(r, c) for r, c in zip(r_all, c_all)]
        f_graphs[reg]['sign_syn'] = beta[r_all, c_all]
        #  graph = nx.DiGraph(conn_mat_funct)
    dout = {}
    for key in ['sparseness', 'intergroup', 'indegree', 'outdegree']:
        if key in what.keys():
            dout[key] = []
    for reg in reg_vect:
        if 'sparseness' in what.keys():
            dout['sparseness'].append(100 * len(f_graphs[reg]['conn_mat']) /
                                      num_neurons**2)
        if 'intergroup' in what.keys():
            i_gr, a_gr = na.inter_group_connections(f_graphs[reg]['conn_mat'],
                                                    what['intergroup'])
            i_gr_tot = np.sum(list(i_gr.values()))
            dout['intergroup'].append(100 * a_gr / i_gr_tot)

    dout['struct'] = {}
    if 'sparseness' in what.keys():
        dout['struct']['sparseness'] = 100 * len(conn_mat_struct) / num_neurons**2
    if 'intergroup' in what.keys():
        i_gr, a_gr = na.inter_group_connections(conn_mat_struct,
                                                what['intergroup'])
        i_gr_tot = np.sum(list(i_gr.values()))
        dout['struct']['intergroup'] = 100 * a_gr / i_gr_tot
    dout['reg_vect'] = reg_vect
    return dout


def plot_graph_metrics(dout, item='sparseness', fpath_out=''):
    """Plot graph metrics."""
    plt.figure(figsize=(10, 10))
    plt.plot(dout['reg_vect'], dout[item], 'g-', lw=2, label='functional')
    x = [dout['reg_vect'][0], dout['reg_vect'][-1]]
    y = [dout['struct'][item], dout['struct'][item]]
    plt.plot(x, y, 'k--', label='structural')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('regularization strength', fontsize=18)
    plt.ylabel(item, fontsize=18)
    plt.legend(loc=0, fontsize=18)
    plt.tight_layout(pad=1)
    plt.savefig(os.path.join(fpath_out, '%s.png' % item))
