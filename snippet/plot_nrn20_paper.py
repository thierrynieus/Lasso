import network_analysis as na
import os
fpath='/home/tnieus/Projects/CODE/Lasso/config/paper/20nrn_16exc_4inh/'
plt.ioff()
for i in range(1,11):
    snum=nu.snum(i,4)
    na.plot_traces('/home/tnieus/Projects/RESULTS/Lasso/paper/20nrn_16exc_4inh/%s/output.npy'%snum,show_noise=False)
    na.plt.savefig('/home/tnieus/Projects/RESULTS/Lasso/paper/20nrn_16exc_4inh/%s/traces_with_noise.png'%snum);
    na.plt.close()
    na.raster_plot('/home/tnieus/Projects/RESULTS/Lasso/paper/20nrn_16exc_4inh/%s/output.npy'%snum,binsz=50)
    na.plt.savefig('/home/tnieus/Projects/RESULTS/Lasso/paper/20nrn_16exc_4inh/%s/raster.png'%snum);
    na.plt.close()


"""
for k in range(6,11):
    params_netw=nu.create_conn_mat(params_neurons)
    snum=nu.snum(k,4)
    fpath='/home/tnieus/Projects/CODE/Lasso/config/paper/20nrn_16exc_4inh/%s/'%snum
    if not(os.path.exists(fpath)):
        os.makedirs(fpath)
        np.save(os.path.join(fpath,'params_netw.npy'),params_netw)


"""

"""
plt.plot(df_out['Matthew_argmax_exc'],df_out['numconn_exc'],'ko')
plt.xlabel('Matthew argmax',fontsize=16)
plt.ylabel('#excitatory connections',fontsize=16)
plt.savefig('/home/tnieus/Projects/RESULTS/Lasso/paper/20nrn_16exc_4inh/numexc_vs_MCCargmax.png')

plt.plot(df_out['Matthew_argmax_all'],df_out['numconn_all'],'ko')
plt.xlabel(r'$\lambda^{-1}_{max}$',fontsize=16)
plt.ylabel('#connections',fontsize=16)
plt.savefig('/home/tnieus/Projects/RESULTS/Lasso/paper/20nrn_16exc_4inh/numall_vs_MCCargmax.png')

"""


"""
plt.plot(1/df_out['Matthew_argmax_all'].to_numpy(),df_out['numconn_all'],'ko')
plt.xlabel(r'$\lambda_{max}$',fontsize=14)
plt.ylabel('#connections',fontsize=14)
plt.savefig('/home/tnieus/Projects/RESULTS/Lasso/paper/20nrn_16exc_4inh/numall_vs_lambda.png')
"""

"""
fs_lab=14
fs_ticks=12
plt.figure(figsize=(4,4))
plt.plot(df_out['numconn_all'],1/df_out['Matthew_argmax_all'].to_numpy(),'ko')
plt.xticks(fontsize=fs_ticks)
plt.yticks(fontsize=fs_ticks)
plt.xlabel('#connections',fontsize=fs_lab)
plt.ylabel(r'$\lambda_{max}$',fontsize=fs_lab)
plt.tight_layout()
plt.savefig('/home/tnieus/Projects/RESULTS/Lasso/paper/20nrn_16exc_4inh/lambda_vs_numall.png')
"""

"""
plt.ioff()
for i in range(1,11):
    snum=nu.snum(i,4)
    run_all('paper/20nrn_16exc_4inh/%s' % snum,get_rs=False)
    plt.close('all')
plt.ion()
"""

"""
plt.plot(df_out['lambda_Mcc_max_all'].to_numpy(),df_out['numconn_all'],'ko')
plt.xlabel(r'$\lambda_{max}$',fontsize=14)
plt.ylabel('#connections',fontsize=14)
plt.savefig('/home/tnieus/Projects/RESULTS/Lasso/paper/20nrn_16exc_4inh/numall_vs_lambda.png')
"""


"""
plt.plot(df_out['lambda_Mcc_max_decay0.05_exc'].to_numpy(),df_out['numconn_all'],'ko')
plt.xlabel(r'$\lambda_{knee}$',fontsize=14)
plt.ylabel('#connections',fontsize=14)
plt.savefig('/home/tnieus/Projects/RESULTS/Lasso/paper/20nrn_16exc_4inh/numall_vs_lambda_knee.png')
"""

"""
mfr_exc,mfr_inh=[],[]
for i in range(1,11):
    snum=nu.snum(i,4)
    out=na.mean_firing_rate('/home/tnieus/Projects/RESULTS/Lasso/paper/20nrn_16exc_4inh/%s/output.npy'%snum)
    mfr_exc.append(np.mean(out[:16]))
    mfr_inh.append(np.mean(out[16:]))
"""

""" # num connections
import network_util as nu
import pandas as pd
num_exc, num_inh, num_all = [], [], []
snum_all = []
for i in range(1, 11):
    snum = nu.snum(i, 4)
    snum_all.append(snum)
    d = np.load('/home/tnieus/Projects/RESULTS/Lasso/paper/20nrn_16exc_4inh/%s/output.npy'%snum, allow_pickle=1).item()
    num_all.append(len(d['params_netw']['conn_mat']))
    nexc, ninh = 0, 0
    for src, dst in d['params_netw']['conn_mat']:
        if src in d['params_netw']['exc']:
            nexc += 1
        if src in d['params_netw']['inh']:
            ninh += 1
    num_exc.append(nexc)
    num_inh.append(ninh)    
dout = dict(num_all=num_all, num_exc=num_exc, num_inh=num_inh, snum_all=snum_all)    
df = pd.DataFrame(dout)    
df.to_csv('/home/tnieus/Projects/RESULTS/Lasso/paper/20nrn_16exc_4inh/num_connections', sep='\t', index = False)
"""

"""
for i in range(1, 11):
    snum = nu.snum(i, 4)
    run_all('paper/20nrn_16exc_4inh/%s' % snum,get_rs=False)
    plt.close('all')
"""

