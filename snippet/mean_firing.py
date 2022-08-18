fn_out='/home/tnieus/Projects/RESULTS/Lasso/100nrn_80exc_20inh_spatial_4clusters/%s/output.npy'
mfr_lst = []
for i in range(10):
    mfr=na.mean_firing_rate(fn_out%nu.snum(i),time_trim=(0,5000))
    #print(i,np.mean(mfr),np.std(mfr))
    mfr_lst.append(np.mean(mfr))
print(np.mean(mfr_lst),np.std(mfr_lst))

fn_out='/home/tnieus/Projects/RESULTS/Lasso/100nrn_80exc_20inh_spatial/%s/output.npy'
mfr_lst = []
for i in range(10):
    mfr=na.mean_firing_rate(fn_out%nu.snum(i),time_trim=(0,5000))
    #print(i,np.mean(mfr),np.std(mfr))
    mfr_lst.append(np.mean(mfr))    
print(np.mean(mfr_lst),np.std(mfr_lst))
