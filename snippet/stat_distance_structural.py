import numpy as np
import pylab as plt
from math import sqrt
import networkx as nx
import network_util as nu

norm = True
figsize = (4, 4)

"""
Refer also to:
    snippet/plot_dist_distrib.py
"""


def distance_structural(fn_out):
    """Report statistics on structural link lenght.

    input:
        fn_out  output.py

    output:
        dist
    """
    from scipy.spatial.distance import pdist, squareform
    d = np.load(fn_out, allow_pickle=1).item()
    num_neurons = d['params_netw']['num_neurons']
    conn_mat = np.array(d['params_netw']['conn_mat'], 'int')
    print(conn_mat.shape[0])
    pos = np.zeros((num_neurons, 2))
    pos[:, 0] = d['params_netw']['x']
    pos[:, 1] = d['params_netw']['y']
    dist_mat = squareform(pdist(pos))
    dist = {}
    for syn_type in ['exc', 'inh']:
        dist[syn_type] = []
    #
    for syn_type in ['exc', 'inh']:
        for nrn in d['params_netw'][syn_type]:
            idx_nrn_src = np.where(conn_mat[:, 0] == nrn)[0]
            idx_nrn_dst = conn_mat[idx_nrn_src, 1]
            dist[syn_type].extend(dist_mat[nrn, idx_nrn_dst])
    return dist


def hist_distance(dist, bins=np.linspace(0, sqrt(2), 50)):
    """Link distribution."""
    hist_conn = {}
    for syn_type in ['exc', 'inh']:
        hist_conn[syn_type] = np.histogram(dist[syn_type], bins=bins)[0]
    hist_conn['bins'] = bins[:-1] + .5 * (bins[1] - bins[0])
    return hist_conn


def plot_graph(fn_out, node_size=400, arrsize=20, ew=1):
    """Plot the structural graph."""
    data = nu.load_dict(fn_out)
    conn_mat = data['params_netw']['conn_mat']
    nrn_src = np.array(conn_mat)[:, 0]
    sign_syn = [1 if nrn_src[k] in data['params_netw']['exc']
                else -1 for k in range(len(nrn_src))]

    graph = nx.DiGraph(conn_mat)
    color_map = ['red' if node in data['params_netw']['exc'] else 'blue'
                 for node in graph]
    edge_widths = []
    edge_colors = []
    pos = [(x, y) for x, y in zip(data['params_netw']['x'],
                                  data['params_netw']['y'])]
    for sign in sign_syn:
        edge_widths.append(ew)
        col = 'r' if sign == 1 else 'b'
        edge_colors.append(col)
        #  edge_colors.append('k')

    # draw graph
    plt.figure(figsize=figsize)
    nx.draw(graph, with_labels=False, pos=pos, node_color=color_map,
            width=edge_widths, node_size=node_size, edge_color=edge_colors,
            edgecolors='k', arrowsize=arrsize)


def inter_cluster_connectivity(fname_params_netw, num_clust=4,
                               num_nrn_clust=25):
    """Quantify connections in/across clusters."""
    data = nu.load_dict(fname_params_netw)
    conn_mat = np.array(data['conn_mat'])
    nrn_idx = data['idx_nrn_type']['idx']  # nrn indexes
    nrn_clust_i = 0
    nrn_clust_s = num_nrn_clust
    idx_clusters = []
    in_cluster_tot, out_cluster_tot = 0, 0
    for c_clust in range(num_clust):
        idx_clust = np.where((nrn_idx>=nrn_clust_i) & (nrn_idx<nrn_clust_s))[0]
        idx_clusters.append(idx_clust)
        nrn_clust_i += num_nrn_clust
        nrn_clust_s += num_nrn_clust
        # iterate over each neuron of the cluster
        in_cluster, out_cluster = 0, 0
        for idx_nrn_c in idx_clust:
            idx_s = np.where(conn_mat[:, 0] == idx_nrn_c)[0]
            nrn_d = conn_mat[idx_s, 1]
            for x in nrn_d:
                if x in idx_clust:
                    in_cluster += 1
                else:
                    out_cluster += 1
        in_cluster_tot += in_cluster
        out_cluster_tot += out_cluster
        print(c_clust, in_cluster, out_cluster)
    print(in_cluster_tot, out_cluster_tot, in_cluster_tot+out_cluster_tot,
          conn_mat.shape[0])
    return in_cluster_tot, out_cluster_tot
