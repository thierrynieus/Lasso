import os

# STEP 1
for i in range(2,11):
    d = nu.load_dict('/home/tnieus/Projects/RESULTS/Lasso/paper/20nrn_16exc_4inh/%s/RSmat_new.npy' % nu.snum(i))
    ne = {}
    ne['exc'] = np.zeros(20,'int')
    ne['inh'] = d['noise_amount'].copy()
    fpath_noise = '/home/tnieus/Projects/RESULTS/Lasso/paper/20nrn_16exc_4inh/%s/added_noise/inhibitory_as_excitatory_in_sim' % nu.snum(i)
    if not(os.path.isdir(fpath_noise)):
        os.makedirs(fpath_noise)
    np.save(os.path.join(fpath_noise, 'noise_amount.npy'), ne)

# STEP 2
for i in range(2,11):
    fname_noise = '/home/tnieus/Projects/RESULTS/Lasso/paper/20nrn_16exc_4inh/%s/added_noise/inhibitory_as_excitatory_in_sim/noise_amount.npy' % nu.snum(i)
    ne = np.load(fname_noise, allow_pickle=1).item()
    for j in range(1,11):
        fpath_out = '/home/tnieus/Projects/RESULTS/Lasso/paper/20nrn_16exc_4inh/%s/added_noise/inhibitory_as_excitatory_in_sim/%s' % (nu.snum(i), nu.snum(j))
        if not(os.path.isdir(fpath_out)):
            os.makedirs(fpath_out)
        dc = update_with_internal_noise_v2('/home/tnieus/Projects/RESULTS/Lasso/paper/20nrn_16exc_4inh/%s/RSmat_new.npy'% nu.snum(i),ne)
        np.save(os.path.join(fpath_out, 'RSmat.npy'),dc)

i = 1
np.random.seed(7831)
fname_noise = '/home/tnieus/Projects/RESULTS/Lasso/paper/20nrn_16exc_4inh/%s/added_noise/inhibitory_as_excitatory_in_sim/noise_amount.npy' % nu.snum(i)
ne = np.load(fname_noise, allow_pickle=1).item()
for j in range(6,11):
    fpath_out = '/home/tnieus/Projects/RESULTS/Lasso/paper/20nrn_16exc_4inh/%s/added_noise/inhibitory_as_excitatory_in_sim/%s' % (nu.snum(i), nu.snum(j))
    if not(os.path.isdir(fpath_out)):
        os.makedirs(fpath_out)
    dc = update_with_internal_noise_v2('/home/tnieus/Projects/RESULTS/Lasso/paper/20nrn_16exc_4inh/%s/RSmat_new.npy'% nu.snum(i),ne)
    np.save(os.path.join(fpath_out, 'RSmat.npy'),dc)

# STEP 3
lst = []
for i in range(1,11):
    fpath_do = '/home/tnieus/Projects/RESULTS/Lasso/paper/20nrn_16exc_4inh/%s/added_noise/inhibitory_as_excitatory_in_sim' % nu.snum(i)
    df_tmp = batch_compute_dale(fpath_do,[1],0)
    df_tmp['folder']='20nrn_16exc_4inh/%s/added_noise/inhibitory_as_excitatory_in_sim' % nu.snum(i)
    lst.append(df_tmp)
df = pd.concat(lst)


for j in range(2,11):
    run_all('/home/tnieus/Projects/RESULTS/Lasso/paper/20nrn_16exc_4inh/0002/added_noise/inhibitory_as_excitatory_in_sim/%s/'%nu.snum(j),get_rs=False)
    plt.close('all')

for j in range(2,11):
    run_all('/home/tnieus/Projects/RESULTS/Lasso/paper/20nrn_16exc_4inh/0003/added_noise/inhibitory_as_excitatory_in_sim/%s/'%nu.snum(j),get_rs=False)
    plt.close('all')

for j in range(2,11):
    run_all('/home/tnieus/Projects/RESULTS/Lasso/paper/20nrn_16exc_4inh/0004/added_noise/inhibitory_as_excitatory_in_sim/%s/'%nu.snum(j),get_rs=False)
    plt.close('all')

for j in range(2,11):
    run_all('/home/tnieus/Projects/RESULTS/Lasso/paper/20nrn_16exc_4inh/0005/added_noise/inhibitory_as_excitatory_in_sim/%s/'%nu.snum(j),get_rs=False)
    plt.close('all')

for j in range(2,11):
    run_all('/home/tnieus/Projects/RESULTS/Lasso/paper/20nrn_16exc_4inh/0006/added_noise/inhibitory_as_excitatory_in_sim/%s/'%nu.snum(j),get_rs=False)
    plt.close('all')

for j in range(2,11):
    run_all('/home/tnieus/Projects/RESULTS/Lasso/paper/20nrn_16exc_4inh/0007/added_noise/inhibitory_as_excitatory_in_sim/%s/'%nu.snum(j),get_rs=False)
    plt.close('all')

# qui

for j in range(3,11):
    run_all('/home/tnieus/Projects/RESULTS/Lasso/paper/20nrn_16exc_4inh/0001/added_noise/inhibitory_as_excitatory_in_sim/%s/'%nu.snum(j),get_rs=False)
    plt.close('all')


for j in range(2,11):
    run_all('/home/tnieus/Projects/RESULTS/Lasso/paper/20nrn_16exc_4inh/0008/added_noise/inhibitory_as_excitatory_in_sim/%s/' % nu.snum(j),get_rs=False)
    plt.close('all')

for j in range(2,11):
    run_all('/home/tnieus/Projects/RESULTS/Lasso/paper/20nrn_16exc_4inh/0009/added_noise/inhibitory_as_excitatory_in_sim/%s/' % nu.snum(j),get_rs=False)
    plt.close('all')

for j in range(2,11):
    run_all('/home/tnieus/Projects/RESULTS/Lasso/paper/20nrn_16exc_4inh/0010/added_noise/inhibitory_as_excitatory_in_sim/%s/' % nu.snum(j),get_rs=False)
    plt.close('all')

for j in range(1,11):
    fpath_csv = '/home/tnieus/Projects/RESULTS/Lasso/paper/20nrn_16exc_4inh/%s/added_noise/inhibitory_as_excitatory_in_sim' % nu.snum(j)
    df = batch_compute_dale(fpath_csv)
    df.to_csv(os.path.join(fpath_csv,'dale.csv'))
