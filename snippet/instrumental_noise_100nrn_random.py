
# update RSmat with noise_amount
for sFolders in ['inhibitory_as_excitatory_in_sim']:
    for i in range(10):
        fn_data='100nrn_80exc_20inh/%s/output.npy' % nu.snum(i)
        for j in range(1,11): # j temporal noise
            fpath_out = '/home/tnieus/Projects/RESULTS/Lasso/100nrn_80exc_20inh/%s/added_noise/%s/%s' % (nu.snum(i), sFolders, nu.snum(j))
            if not(os.path.isdir(fpath_out)):
                os.makedirs(fpath_out)
            calc_mat_processes(fn_data,'100nrn_80exc_20inh/%s/RSmat_new.npy'% nu.snum(i))

# save noise_amount.npy
for sFolders in ['inhibitory_as_excitatory_in_sim']:
    for i in range(10):
        fname = '/home/tnieus/Projects/RESULTS/Lasso/100nrn_80exc_20inh/%s/RSmat_new.npy'% nu.snum(i)
        d = nu.load_dict(fname)        
        anoise={}
        anoise['exc']=np.zeros(100, 'int')
        anoise['inh']=d['noise_amount']
        fname_out = '/home/tnieus/Projects/RESULTS/Lasso/100nrn_80exc_20inh/%s/added_noise/%s/noise_amount.npy' % (nu.snum(i), sFolders)
        np.save(fname_out, anoise)
        
# create appropriate RSmat files
for sFolders in ['inhibitory_as_excitatory_in_sim']:
    for i in range(10):
        fn_rs = '/home/tnieus/Projects/RESULTS/Lasso/100nrn_80exc_20inh/%s/RSmat_new.npy'% nu.snum(i)
        fn_noise = '/home/tnieus/Projects/RESULTS/Lasso/100nrn_80exc_20inh/%s/added_noise/%s/noise_amount.npy' % (nu.snum(i), sFolders)
        anoise = nu.load_dict(fn_noise)
        for j in range(1,11): # j temporal noise
            dnew = add_instrumental_noise_v2(fn_rs, anoise)
            fn_out = '/home/tnieus/Projects/RESULTS/Lasso/100nrn_80exc_20inh/%s/added_noise/%s/%s/RSmat.npy' % (nu.snum(i), sFolders, nu.snum(j))
            np.save(fn_out, dnew)
