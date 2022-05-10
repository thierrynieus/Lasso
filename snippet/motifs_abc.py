"""

d=dout['paths3']
n=d.shape[0]

#
count_c_b_a = 0
for k in range(n):
  idx = np.where((d[k,0] == d[:,2]) & (d[k,2] == d[:,0]))[0]
  L = len(idx)
  if L:
    count_c_b_a += 1



dL = [tuple(dr) for dr in d]
dL_swap = [tuple(dr) for dr in d[:,[2,1,0]]]
"""
import numpy as np
import os
import networkx as nx
import network_util as nu
import pylab as plt


def analyze_triplets(fpath_out, reg):
    """Analyze triplets."""
    data = nu.load_dict(os.path.join(fpath_out, 'output.npy'))
    # conn_mat_struct = data['params_netw']['conn_mat']
    beta = nu.load_dict(os.path.join(fpath_out, 'reg_%g' % reg,
                                     'RSmat_lasso.npy'))['beta']
    """
    pos = [(x, y) for x, y in zip(data['params_netw']['x'],
                                  data['params_netw']['y'])]
    """
    # structural graph
    conn_structural = nx.DiGraph(data['params_netw']['conn_mat'])
    conn_structural_sp = nx.shortest_path(conn_structural)
    conn_structural_paths3 = [tuple(path) for k in range(data['num_neurons'])
                              for path in list(conn_structural_sp[k].values())
                              if len(path) == 3]

    conn_structural_no_paths2 = [(conn[0], conn[2])
                                 for conn in conn_structural_paths3]

    conn_structural_paths2 = [tuple(path) for k in range(data['num_neurons'])
                              for path in list(conn_structural_sp[k].values())
                              if len(path) == 2]
    # functional graph
    r_all, c_all = np.where(beta)
    conn_functional = nx.DiGraph([(r, c) for r, c in zip(r_all, c_all)])
    conn_functional_sp = nx.shortest_path(conn_functional)
    """
    conn_functional_paths3 = [tuple(path) for k in range(data['num_neurons'])
                              for path in list(conn_functional_sp[k].values())
                              if len(path) == 3]
    """
    conn_functional_paths2 = [tuple(path) for k in range(data['num_neurons'])
                              for path in list(conn_functional_sp[k].values())
                              if len(path) == 2]

    """ search for functional links (x,y):
            1) in the fake structural x(->z)->y
            2) in the real structural x->y
    """
    count_fake = 0
    count_real = 0
    for conn_func in conn_functional_paths2:
        if conn_func in conn_structural_no_paths2:
            count_fake += 1
        if conn_func in conn_structural_paths2:
            count_real += 1

    lun_funct2 = len(conn_functional_paths2)
    lun_struct2 = len(conn_structural_paths2)

    """
    lun_funct2-count_real vs count_fake
    """

    return count_fake, count_real, lun_funct2, lun_struct2


def plot_fake_real(dout, reg):
    """Do it now."""
    bs_all = dout.keys()
    colors={}
    colors[1] = 'k'
    colors[5] = 'r'
    colors[10] = 'b'
    for bs in bs_all:
        mat = np.array(dout[bs])
        plt.plot(reg[bs], mat[:, 2]-mat[:, 1], 'o-', c=colors[bs], lw=2, label='#diff(func,real)(%d)'%bs)
        plt.plot(reg[bs], mat[:, 0], 'o--', c=colors[bs], lw=2, label='#indirect(%d)'%bs)
    plt.legend(loc=0, fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('regularization strength', fontsize=18)
    plt.ylabel('# connections', fontsize=18)
