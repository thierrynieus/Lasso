import numpy as np
import pylab as plt
import os

import network_util as nu

plt.ion()
colormap = 'jet'

fpath = '/home/tnieus/Projects/RESULTS/Lasso/paper/20nrn_16exc_4inh/'
fpath_fig = '/home/tnieus/Projects/RESULTS/Lasso/paper/20nrn_16exc_4inh/noise/added_noise/fig'

noise_types = ['inhibitory_as_excitatory_in_sim', 'exc_inh_x2', 'exc_inh_x4', 'exc_inh_x16']
noise_labels = ['(0,1)', '(1,2)', '(2,4)', '(8,16)']
number = len(noise_types)
figsize = (4,4)
fs_lab = 14
fs_ticks = 12
fs_legend = 12
ms = 10  # markersize
fig_ext = 'png'

    
def plot_MCC(sim_num=1, idx_rng=range(1,11)):
    """Get MCC."""
    # main
    fn_csv = os.path.join(fpath, nu.snum(sim_num), 'confusion_mat.csv')        
    df = pd.read_csv(fn_csv)
    mcc_main = {}
    for syn_type in ['all', 'exc', 'inh']:
        mcc_main[syn_type] = matthew_coeff(select_conf_mat(df, syn_type))[::-1]    
    lamb = 1/df['regularization_strength'].to_numpy()[::-1]    
    # trials
    mat_dict = {}
    for noise_type in noise_types:
        mat_dict[noise_type] = {}
        for syn_type in ['all', 'exc', 'inh']:        
            mat_dict[noise_type][syn_type] = []
    for noise_type in noise_types:            
        for idx in idx_rng:            
            fn_csv = os.path.join(fpath, nu.snum(sim_num), 'added_noise', noise_type, nu.snum(idx), 'confusion_mat.csv')        
            df = pd.read_csv(fn_csv)
            for syn_type in ['all', 'exc', 'inh']:
                mat_dict[noise_type][syn_type].append(matthew_coeff(select_conf_mat(df, syn_type))[::-1])
                        
    # plot
    cmap = plt.get_cmap(colormap)
    colors = [cmap(i) for i in np.linspace(0, 1, number)]
    
    for count, syn_type in enumerate(['all', 'exc', 'inh']):
        plt.figure(count+1, figsize=figsize)
        #plt.title(syn_type)
        plt.plot(lamb, mcc_main[syn_type], 'r-', lw=2, label='default')
        for i, (col, noise_type) in enumerate(zip(colors, noise_types)):
            mat = np.array(mat_dict[noise_type][syn_type])
            m = mat.mean(axis=0)
            s = mat.std(axis=0)
            plt.errorbar(lamb, y=m, yerr=s, ecolor=col, color=col, label=noise_labels[i], 
                         markersize=ms, elinewidth=2, linewidth=2)
        plt.legend(loc=0, fontsize=fs_legend)    
        plt.xlabel(r'$\lambda$', fontsize=fs_lab)
        plt.ylabel('MCC', fontsize=fs_lab)
        plt.xticks(fontsize=fs_ticks)
        plt.yticks(fontsize=fs_ticks)
        plt.tight_layout(pad=1)
        plt.savefig(os.path.join(fpath_fig,'%s_%s.%s' % (nu.snum(sim_num), syn_type, fig_ext)))
        plt.close()


def plot_MCC_peak(sim=range(1,11), idx_rng=range(1,11)):
    """Get MCC."""
    # trials
    mat_dict = {}
    for syn_type in ['all', 'exc', 'inh']:        
        mat_dict[syn_type] = {}
        for noise_type in noise_types:        
            mat_dict[syn_type][noise_type] = []                
    for syn_type in ['all', 'exc', 'inh']:            
        for noise_type in noise_types:            
            for idx in idx_rng:    
                for sim_num in sim:
                    fn_csv = os.path.join(fpath, nu.snum(sim_num), 'added_noise', noise_type, nu.snum(idx), 'confusion_mat.csv')        
                    df = pd.read_csv(fn_csv)                
                    mat_dict[syn_type][noise_type].append(np.max(matthew_coeff(select_conf_mat(df, syn_type))))
                    
    for syn_type in ['all', 'exc', 'inh']:
        plt.figure(figsize=figsize)
        dout = []
        for noise_type in noise_types:
            dout.append(mat_dict[syn_type][noise_type])
        plt.boxplot(dout, whis=[5,95])        
        plt.xticks([1,2,3,4], noise_labels, fontsize=fs_ticks)
        plt.yticks(fontsize=fs_ticks)
        plt.xlabel('instrumental noise', fontsize=fs_lab)
        plt.ylabel('MCC peak', fontsize=fs_lab)
        plt.tight_layout(pad=1)      
        plt.savefig(os.path.join(fpath_fig,'boxplot_%s.%s' % (syn_type, fig_ext)))          
        plt.close()




