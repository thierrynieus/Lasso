import numpy as np
import os
import json
import glob
import pylab as plt
plt.ion()

fpath_results = '/home/tnieus/Projects/RESULTS/Lasso/'
fpath_cfg = '/home/tnieus/Projects/CODE/Lasso/config'


def snum(num, lun_str=4):
    """Generate a fixed-lenght string from a number."""
    str_num = '%d' % num
    while len(str_num) < lun_str:
        str_num = '0' + str_num
    return str_num


def save_json(data, fpath, fname):
    """Save a json file.

    comment: unused
    """
    count = 0
    while os.path.exists(os.path.join(fpath, snum(count))):
        count += 1
    print(count)
    fpath_new = os.path.join(fpath, snum(count))
    os.mkdir(fpath_new)
    fn_json = os.path.join(fpath_new, snum(count), fname)
    with open(fn_json, 'w') as outfile:
        json.dump(data, outfile)


def load_json(fname, fpath, count):
    """Load a json file.

    comment: unused
    """
    fn_json = os.path.join(fpath, snum(count), fname)
    f = open(fn_json)
    data = json.load(f)
    return data


def save_npy(data, fpath, fname):
    """Save file and prevent overwriting."""
    count = 0
    while os.path.isdir(os.path.join(fpath, snum(count))):
        count += 1
    print(count)
    os.makedirs(os.path.join(fpath, snum(count)))
    fn_npy = os.path.join(fpath, snum(count), fname)
    np.save(fn_npy, data)


def load_dict(fname_py):
    """Load a dictionary from a file.

    comment: a simple and less cumbersome way to load it
    """
    return np.load(fname_py, allow_pickle=1).item()


def save_config(params_syn, params_noise, params_neurons, params_netw):
    """Save configuration.

    comment: unused
    """
    save_json(params_syn, fpath_cfg, 'params_syn.json')
    save_json(params_noise, fpath_cfg, 'params_noise.json')
    save_json(params_neurons, fpath_cfg, 'params_neurons.json')
    save_npy(params_netw, fpath_cfg, 'params_netw.npy')


def load_config(count=0):
    """Load configuration.

    comment: unused
    """
    params_syn = load_json('params_syn.json', fpath_cfg, count)
    params_noise = load_json('params_noise.json', fpath_cfg, count)
    params_neurons = load_json('params_neurons.json', fpath_cfg, count)
    fn_npy = os.path.join(fpath_cfg, snum(count), 'params_netw.npy')
    params_netw = np.load(fn_npy, allow_pickle=1).item()
    return params_syn, params_noise, params_neurons, params_netw


def create_conn_mat(params_neurons):
    """Build a connectome without spatial information."""
    n = params_neurons['num_neurons']
    ne = params_neurons['num_exc_neurons']
    params_netw = {}
    params_netw['num_neurons'] = n
    params_netw['exc'] = np.arange(ne)
    params_netw['inh'] = np.arange(ne, n)
    params_netw['conn_mat'] = []
    for src in range(n):
        for dst in range(n):
            if src != dst:
                rnd = np.random.rand()
                if (src in params_netw['exc']) & (rnd < params_neurons['pe']):
                    params_netw['conn_mat'].append((src, dst))
                if (src in params_netw['inh']) & (rnd < params_neurons['pi']):
                    params_netw['conn_mat'].append((src, dst))
    if params_neurons['spatial_coordinates']:
        params_netw['x'] = np.random.rand(n)
        params_netw['y'] = np.random.rand(n)
    return params_netw


def create_conn_mat_spatial(params_neurons, pos=None):
    """Build a connectome with spatial information.

    comment:
        - possibile additional argument (x,y) (default empty)
        - additional prefixed links (default empty)
    """
    n = params_neurons['num_neurons']
    ne = params_neurons['num_exc_neurons']
    params_netw = {}
    params_netw['num_neurons'] = n
    params_netw['exc'] = np.arange(ne)
    params_netw['inh'] = np.arange(ne, n)
    params_netw['conn_mat'] = []

    params_neurons['spatial_coordinates'] = True
    if pos is None:
        pos = np.random.rand((n, 2))
    params_netw['x'] = pos[:, 0]
    params_netw['y'] = pos[:, 1]

    sigma = params_neurons['sigma'] if 'sigma' in params_neurons else 0.25

    # taken from utils.py Lonardoni et al. 2018
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import chi2
    dist_mat = squareform(pdist(pos))
    dist_mat = chi2(1).cdf(dist_mat / sigma)
    dist_mat += np.eye(len(dist_mat)) * 10
    prob_mat = np.random.random(dist_mat.shape)
    edges = np.where(dist_mat < prob_mat)
    for edge in np.array(edges).T:
        params_netw['conn_mat'].append(tuple(edge))
    return params_netw


def get_regularization_factor(fpath):
    """Get the analyzed regularization coefficients."""
    fn_search = os.path.join(fpath, 'reg_*')
    fn_lst = glob.glob(fn_search, recursive=True)
    reg_lst = [float(fn.split('/')[-1].replace('reg_', '')) for fn in fn_lst]
    return np.sort(reg_lst)
