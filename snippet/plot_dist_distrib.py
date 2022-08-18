import os
import numpy as np
import pylab as plt

plt.figure(figsize=(4, 4))

fpath = '/home/tnieus/Projects/RESULTS/Lasso/100nrn_80exc_20inh_spatial/'
hist_lst = []
for num in range(10):
    dist = distance_structural(os.path.join(fpath, nu.snum(num), 'output.npy'))
    hist_conn = hist_distance(dist)
    hist_lst.append(hist_conn['exc']+hist_conn['inh'])
hist = np.array(hist_lst)
hs=hist.sum(axis=1)
h=hist/hs[:,None]
x = hist_conn['bins']
y = h.mean(axis=0)
yerr = h.std(axis=0)
#for i in range(10):
#    plt.plot(hist_conn['bins'],h[i,:],'k-')
plt.errorbar(x=x,y=y,yerr=yerr,elinewidth=2,linewidth=2,ecolor='k',color='k',label='Gauss')

fpath = '/home/tnieus/Projects/RESULTS/Lasso/100nrn_80exc_20inh_spatial_4clusters/'
hist_lst = []
for num in range(10):
    dist = distance_structural(os.path.join(fpath, nu.snum(num), 'output.npy'))
    hist_conn = hist_distance(dist)
    hist_lst.append(hist_conn['exc']+hist_conn['inh'])
hist = np.array(hist_lst)
hs=hist.sum(axis=1)
h=hist/hs[:,None]
x = hist_conn['bins']
y = h.mean(axis=0)
yerr = h.std(axis=0)
plt.errorbar(x=x,y=y,yerr=yerr,elinewidth=2,linewidth=2,ecolor='g',color='g',label='4 clusters')
plt.legend(loc=0)

plt.xlim(0,1.1)
plt.grid()
plt.xlabel('distance',fontsize=14)
plt.ylabel('frequency',fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout(pad=1)
