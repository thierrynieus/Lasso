import numpy as np
import pylab as plt
import os

import network_analysis as na

fpath='/home/tnieus/Projects/RESULTS/Lasso/100nrn_80exc_20inh_spatial/0001/'
fname='output.npy'

sname={}
sname[1]=''
sname[1.1]='change_weight/0000'
sname[1.2]='change_weight/0001'
sname[1.3]='change_weight/0002'
sname[1.4]='change_weight/0003'
sname[1.5]='change_weight/0004'
sname[2]='change_weight/0005'
slabel= []

plt.figure(figsize=(8,8))

def spike_synchrony(fn_out, time_trim=(100, 5000)):
    """Compute Kreuz spike synchrony measure."""
    data = np.load(fn_out, allow_pickle=1).item()
    if 'num_neurons' in data:
        num_neurons = data['num_neurons']
    else:
        # in old fn_data num_neurons was not available!
        num_neurons = len([x for x in data if isinstance(x, int)])  # robust?
    # 
    ti, te = time_trim        
    spk_trains = []
    for nrn in range(num_neurons):
        spk = data[nrn]['spikes_nrn']
        spk = spk[(spk>ti) & (spk<te)]
        spk_trains.append(pyspike.SpikeTrain(spk, edges=1e12))  # edges works ??
    #  
    sync_idx=[]    
    for idx_src in range(num_neurons):
        for idx_dst in range(idx_src+1,num_neurons):
                sync_idx.append(pyspike.spike_sync(spk_trains[idx_src], spk_trains[idx_dst]))
    return np.array(sync_idx)
    
sync_all = []
    
for key in sname.keys():
    fn_data = os.path.join(fpath,sname[key],fname)
    sync = spike_synchrony(fn_data, time_trim=(100, 10000))
    sync_all.append(sync)    
    slabel.append('x %g' % key)
    
lun=len(sname)
plt.boxplot(sync_all, whis=[5,95])
plt.xticks(np.arange(1,lun+1), slabel, fontsize=14)
plt.xlabel('synaptic coupling', fontsize=16)
plt.ylabel('spike synchrony', fontsize=16)

plt.tight_layout(pad=1)    
    
