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
mfr_all = []
plt.figure(figsize=(8,8))

for key in sname.keys():
    fn_data = os.path.join(fpath,sname[key],fname)
    mfr = na.mean_firing_rate(fn_data, time_trim=(500,10000))
    mfr_all.append(mfr)    
    slabel.append('x %g' % key)
    
lun=len(sname)
plt.boxplot(mfr_all, whis=[5,95])
plt.xticks(np.arange(1,lun+1), slabel, fontsize=14)
plt.xlabel('synaptic coupling', fontsize=16)
plt.ylabel('mean firing rate (Hz)', fontsize=16)

plt.tight_layout(pad=1)



