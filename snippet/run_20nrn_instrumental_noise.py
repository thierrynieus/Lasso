

d = nu.load_dict('/home/tnieus/Projects/RESULTS/Lasso/paper/20nrn_16exc_4inh/%s/RSmat_new.npy'% nu.snum(i))




for sFolders in ['exc_inh_x2', 'exc_inh_x4', 'exc_inh_x16']:
    for i in range(2,11): # i network
        #i=1
        fname_noise = '/home/tnieus/Projects/RESULTS/Lasso/paper/20nrn_16exc_4inh/%s/added_noise/%s/noise_amount.npy' % (nu.snum(i),sFolders)
        ne = np.load(fname_noise, allow_pickle=1).item()    
        for j in range(2,11): # j temporal noise
            fpath_out = '/home/tnieus/Projects/RESULTS/Lasso/paper/20nrn_16exc_4inh/%s/added_noise/%s/%s' % (nu.snum(i), sFolders, nu.snum(j))
            if not(os.path.isdir(fpath_out)):
                os.makedirs(fpath_out)
            dc = update_with_internal_noise_v2('/home/tnieus/Projects/RESULTS/Lasso/paper/20nrn_16exc_4inh/%s/RSmat_new.npy'% nu.snum(i),ne)
            np.save(os.path.join(fpath_out, 'RSmat.npy'),dc)
            
            
# RUN             
params_roc['reg_vect'] = [ 0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  1. ,  1.2,  1.5, 2. ,  3. ,  5. , 10. ]
plt.ioff()

#sFolder='exc_inh_x2' 
#sFolder='exc_inh_x4'
sFolder='exc_inh_x16'
#for i in [2]: #range(3,11):
for i in range(3,11):
    for j in range(1,11):
        run_all('paper/20nrn_16exc_4inh/%s/added_noise/%s/%s' % (nu.snum(i), sFolder, nu.snum(j)))
        plt.close('all')
        

# network 0002..0010        
noise_factors={}
noise_factors['exc_inh_x2'] = (1, 2)
noise_factors['exc_inh_x4'] = (2, 4)
noise_factors['exc_inh_x16'] = (8, 16)
for i in range(2,11): # network number i
    for knoise, sFolders in enumerate(['exc_inh_x2', 'exc_inh_x4', 'exc_inh_x16']):    
        fname_RS = '/home/tnieus/Projects/RESULTS/Lasso/paper/20nrn_16exc_4inh/%s/RSmat_new.npy' % (nu.snum(i))    
        ne = nu.load_dict(fname_RS)['noise_amount']
        #
        dd = {}
        dd['exc'] = ne * noise_factors[sFolders][0]
        dd['inh'] = ne * noise_factors[sFolders][1]               
        #
        fname_noise = '/home/tnieus/Projects/RESULTS/Lasso/paper/20nrn_16exc_4inh/%s/added_noise/%s/noise_amount.npy' % (nu.snum(i),sFolders)
        np.save(fname_noise, dd)
        for j in range(1,11): # j temporal noise
            fpath_out = '/home/tnieus/Projects/RESULTS/Lasso/paper/20nrn_16exc_4inh/%s/added_noise/%s/%s' % (nu.snum(i), sFolders, nu.snum(j))
            if not(os.path.isdir(fpath_out)):
                os.makedirs(fpath_out)
            dc = update_with_internal_noise_v2('/home/tnieus/Projects/RESULTS/Lasso/paper/20nrn_16exc_4inh/%s/RSmat_new.npy'% nu.snum(i),dd)
            np.save(os.path.join(fpath_out, 'RSmat.npy'),dc)

import numpy as np
np.random.seed(5264)
import os
fpath = '/home/tnieus/Projects/RESULTS/Lasso/paper/20nrn_16exc_4inh'
frac_str={}
frac_str[0.2] = '20perc'
frac_str[0.5] = '50perc'
frac_str[0.9] = '90perc'
for i in range(1,11):
    fn_rs = os.path.join(fpath, nu.snum(i), 'RSmat.npy')
    for frac in [0.2, 0.5, 0.9]:
        for trial in range(1,11):
            data_copy = add_inh_to_exc_switch(fn_rs, frac)
            fpath_out = os.path.join(fpath, nu.snum(i), 'added_wrong_detection/%s/'%frac_str[frac], nu.snum(trial))
            if not(os.path.isdir(fpath_out)):
                os.makedirs(fpath_out)                            
            np.save(os.path.join(fpath_out, 'RSmat.npy'), data_copy)      


# network 0001..0010
# 20perc, 50perc, 90perc
# trials 0001..0010
run_all('paper/20nrn_16exc_4inh/0001/added_wrong_detection/90perc/0001','paper/20nrn_16exc_4inh/0001/')        

            
'''
# /home/tnieus/Projects/RESULTS/Lasso/paper/20nrn_16exc_4inh/0004/added_noise/inhibitory_as_excitatory_in_sim/0001
# poi 
cd ../../../../0010/added_noise/inhibitory_as_excitatory_in_sim/0001


cp  -P  output.npy ../../exc_inh_x2/0001 # copy symbolic link -P, --no-dereference, never follow symbolic links in SOURCE
cp  -P  output.npy ../../exc_inh_x2/0002
cp  -P  output.npy ../../exc_inh_x2/0003
cp  -P  output.npy ../../exc_inh_x2/0004
cp  -P  output.npy ../../exc_inh_x2/0005
cp  -P  output.npy ../../exc_inh_x2/0006
cp  -P  output.npy ../../exc_inh_x2/0007
cp  -P  output.npy ../../exc_inh_x2/0008
cp  -P  output.npy ../../exc_inh_x2/0009
cp  -P  output.npy ../../exc_inh_x2/0010
cp  -P  output.npy ../../exc_inh_x4/0001
cp  -P  output.npy ../../exc_inh_x4/0002
cp  -P  output.npy ../../exc_inh_x4/0003
cp  -P  output.npy ../../exc_inh_x4/0004
cp  -P  output.npy ../../exc_inh_x4/0005
cp  -P  output.npy ../../exc_inh_x4/0006
cp  -P  output.npy ../../exc_inh_x4/0007
cp  -P  output.npy ../../exc_inh_x4/0008
cp  -P  output.npy ../../exc_inh_x4/0009
cp  -P  output.npy ../../exc_inh_x4/0010
cp  -P  output.npy ../../exc_inh_x16/0001
cp  -P  output.npy ../../exc_inh_x16/0002
cp  -P  output.npy ../../exc_inh_x16/0003
cp  -P  output.npy ../../exc_inh_x16/0004
cp  -P  output.npy ../../exc_inh_x16/0005
cp  -P  output.npy ../../exc_inh_x16/0006
cp  -P  output.npy ../../exc_inh_x16/0007
cp  -P  output.npy ../../exc_inh_x16/0008
cp  -P  output.npy ../../exc_inh_x16/0009
cp  -P  output.npy ../../exc_inh_x16/0010





'''            
