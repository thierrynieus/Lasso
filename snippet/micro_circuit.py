import networkx as nx
import network_util as nu
import numpy as np

fn_out = '/home/tnieus/Projects/RESULTS/Lasso/100nrn_80exc_20inh_spatial/0001/output.npy'

d = nu.load_dict(fn_out)

G = nx.DiGraph(d['params_netw']['conn_mat'])
pos = [(x, y) for x, y in zip(d['params_netw']['x'], d['params_netw']['y'])]
out = nx.shortest_path(G)

paths3 = np.array([path for k in range(d['num_neurons'])
                   for path in list(out[k].values()) if len(path) == 3])
n3 = paths3.shape[0]
idx = np.random.randint(n3)

nrn_exclude = paths3[idx, 1]

idx = np.where(paths3[:, 1] == nrn_exclude)[0]

# np.unique(paths3[:,1],return_counts=True)

print(len(idx))

nrn_src = paths3[idx, 0]
nrn_dst = paths3[idx, 2]
nrn_src_dst = np.hstack((nrn_src, nrn_dst))
nrn_src_dst_with_exclude = np.intersect1d(nrn_src_dst, nrn_exclude)

print(len(np.unique(nrn_src)), len(np.unique(nrn_dst)),
      len(np.unique(nrn_src_dst)), len(nrn_src_dst_with_exclude))

nrn_2_monitor = np.union1d(np.unique(nrn_src), np.unique(nrn_dst))
