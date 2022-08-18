



import os
import numpy as np

sub_folders = {}
sub_folders['less'] = 'subsample/20_less_connected'
sub_folders['most'] = 'subsample/20_most_connected'

def most_less_connected(fpath_out, num_select = 20):
    """
    """
    fn_out = os.path.join(fpath_out, 'output.npy')
    d = nu.load_dict(fn_out)
    a = np.unique(np.array(d['params_netw']['conn_mat'])[:,0], return_counts=True)
    idx_numconn = a[1].argsort()
    # less
    idx_less = idx_numconn[:num_select]
    print(a[1][idx_less])    
    dnew_less = subsample_output2(fn_out, idx_less)
    fpath_less = os.path.join(fpath_out, sub_folders['less'])
    if not(os.path.isdir(fpath_less)):
        os.makedirs(fpath_less)
    np.save(os.path.join(fpath_less, 'output.npy'), dnew_less)            
    # most
    idx_most = idx_numconn[-num_select:]
    print(a[1][idx_most])    
    dnew_most = subsample_output2(fn_out, idx_most)    
    fpath_most = os.path.join(fpath_out, sub_folders['most'])
    if not(os.path.isdir(fpath_most)):
        os.makedirs(fpath_most)
    np.save(os.path.join(fpath_most, 'output.npy'), dnew_most)    
