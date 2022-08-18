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


def analyze_motifs(fpath_out, reg_lst, num_hopes=8):
    """Analyze motifs."""
    #  structural connectome
    data = nu.load_dict(os.path.join(fpath_out, 'output.npy'))
    if 'num_neurons' in data:
        num_neurons = data['num_neurons']
    else:
        # in old fn_data num_neurons was not available!
        num_neurons = len([x for x in data if isinstance(x, int)])

    conn_structural = nx.DiGraph(data['params_netw']['conn_mat'])
    conn_structural_sp = nx.shortest_path(conn_structural)
    mat_count_paths = np.zeros((num_hopes - 1, len(reg_lst)))
    struct_paths = {}

    for hope in range(2, num_hopes+1):
        struct_paths[hope] = [tuple(path) for k in range(num_neurons)
                              for path in list(conn_structural_sp[k].values())
                              if len(path) == hope]
        print('hopes %d in the structural connectome with %d paths' %
              (hope, len(struct_paths[hope])))

    for idx_col, reg in enumerate(reg_lst):
        beta = nu.load_dict(os.path.join(fpath_out, 'reg_%g' % reg,
                                         'RSmat_lasso.npy'))['beta']
        # functional connectome
        r_all, c_all = np.where(beta)
        conn_functional = nx.DiGraph([(r, c) for r, c in zip(r_all, c_all)])
        conn_functional_sp = nx.shortest_path(conn_functional)
        """
        for k in range(num_neurons):
            if not(k in conn_functional_sp.keys()):
                conn_functional_sp[k] = []
        """
        count_real_lst = []
        for hope in range(2, num_hopes+1):
            funct_path = [tuple(path) for k in range(num_neurons)
                          for path in list(conn_functional_sp[k].values())
                          if len(path) == hope]
            num_funct_path = len(funct_path)
            #  print(hope, num_funct_path)
            count_real = 0
            for conn_func in funct_path:
                if conn_func in struct_paths[hope]:
                    count_real += 1
            num_struct_path = len(struct_paths[hope])
            print(reg, hope, count_real, num_struct_path, num_funct_path)
            if num_funct_path * num_struct_path:
                count_real_lst.append(count_real / num_struct_path)
            else:
                count_real_lst.append(0)
            #  if count_real:
            #    print(hope, count_real, num_funct_path)
        mat_count_paths[:, idx_col] = np.array(count_real_lst)
    return mat_count_paths, struct_paths


def analyze_motifs_v2(fpath_out, reg_lst, num_hopes=6):
    """Analyze motifs."""
    #  structural connectome
    data = nu.load_dict(os.path.join(fpath_out, 'output.npy'))
    if 'num_neurons' in data:
        num_neurons = data['num_neurons']
    else:
        # in old fn_data num_neurons was not available!
        num_neurons = len([x for x in data if isinstance(x, int)])

    conn_structural = nx.DiGraph(data['params_netw']['conn_mat'])

    conn_structural_paths = []
    for nrn_src in range(num_neurons):
        for nrn_dst in range(num_neurons):
            if nrn_src != nrn_dst:
                simp_paths = nx.all_simple_paths(conn_structural, nrn_src,
                                                 nrn_dst, cutoff=num_hopes)
                conn_structural_paths.extend(simp_paths)
    lun_conn_structural_paths = [len(conn) for conn in conn_structural_paths]
    return conn_structural_paths, lun_conn_structural_paths
    """
    nx._all_simple_paths_graph

    conn_structural_sp = nx.shortest_path(conn_structural)
    mat_count_paths = np.zeros((num_hopes - 1, len(reg_lst)))
    struct_paths = {}

    for hope in range(2, num_hopes+1):
        struct_paths[hope] = [tuple(path) for k in range(num_neurons)
                              for path in list(conn_structural_sp[k].values())
                              if len(path) == hope]
        print('hopes %d in the structural connectome with %d paths' %
              (hope, len(struct_paths[hope])))

    for idx_col, reg in enumerate(reg_lst):
        beta = nu.load_dict(os.path.join(fpath_out, 'reg_%g' % reg,
                                         'RSmat_lasso.npy'))['beta']
        # functional connectome
        r_all, c_all = np.where(beta)
        conn_functional = nx.DiGraph([(r, c) for r, c in zip(r_all, c_all)])
        conn_functional_sp = nx.shortest_path(conn_functional)
        count_real_lst = []
        for hope in range(2, num_hopes+1):
            funct_path = [tuple(path) for k in range(num_neurons)
                          for path in list(conn_functional_sp[k].values())
                          if len(path) == hope]
            num_funct_path = len(funct_path)
            #  print(hope, num_funct_path)
            count_real = 0
            for conn_func in funct_path:
                if conn_func in struct_paths[hope]:
                    count_real += 1
            num_struct_path = len(struct_paths[hope])
            print(reg, hope, count_real, num_struct_path, num_funct_path)
            if num_funct_path * num_struct_path:
                count_real_lst.append(count_real / num_struct_path)
            else:
                count_real_lst.append(0)
            #  if count_real:
            #    print(hope, count_real, num_funct_path)
        mat_count_paths[:, idx_col] = np.array(count_real_lst)
    return mat_count_paths, struct_paths
    """

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


'''
plt.plot(1/reg_lst, mat_count_paths[0,::-1], 'ko-', label='hope 2')
plt.plot(1/reg_lst, mat_count_paths[1,::-1], 'ro-', label='hope 3')
plt.plot(1/reg_lst, mat_count_paths[2,::-1], 'bo-', label='hope 4')
plt.plot(1/reg_lst, mat_count_paths[3,::-1], 'go-', label='hope 5')
plt.xlabel(r'$\lambda$', fontsize=18)
plt.ylabel('fraction correct', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(loc='lower left', fontsize=16)
plt.tight_layout(pad=1)
'''
