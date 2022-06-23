import numpy as np
import network_util as nu


def compute_variance_connectivity(fn_data):
    """Compute variance connectivity."""
    data = nu.load_dict(fn_data)
    data_conn = np.array(data['params_netw']['conn_mat'])
    nrn_dst_unique = np.unique(data_conn[:, 1])
    num_conn_dst = {}
    for nrn_dst in nrn_dst_unique:
        idx = np.where(data_conn[:, 1] == nrn_dst)[0]
        num_conn_dst[nrn_dst] = len(idx)
    return num_conn_dst
