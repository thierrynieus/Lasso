# network_analysis
import numpy as np
import pylab as plt
import network_util as nu

'''
> sincronia Kreuz
> MFR, IFR
> graph parameters ? networkx
 '''


def plot_traces(fn_data, num_max=1e10, show_noise=True, delay=0):
    '''
    '''
    data = nu.load_dict(fn_data)
    if not('vm' in list(data[0])):
        print('>>> Error: raw traces are not available!')
        return None
    num_neurons = len([x for x in data if isinstance(x, int)])  # robust?
    num_show = min(num_neurons, num_max)
    idx = np.random.choice(num_neurons, num_show, replace=False)
    #
    plt.figure()
    #
    for k in range(num_show):
        plt.subplot(num_show, 1, k+1)
        i_k = idx[k]
        plt.plot(data[i_k]['vm']['times'], data[i_k]['vm']['V_m'], 'k-', lw=2)
        plt.ylabel('V_%d (mV)' % k)
        plt.ylim([-80, 20])
        plt.grid()
    # show_noise
    if show_noise:
        for k in range(num_show):
            plt.subplot(num_show, 1, k+1)
            i_k = idx[k]
            for nt in data[i_k]['spikes_noise']:
                plt.plot([nt+delay, nt+delay], [-80, 20], 'b-')


def raster_plot(fn_data, binsz=20):
    """Produce a raster plot."""
    data = nu.load_dict(fn_data)
    num_neurons = len([x for x in data if isinstance(x, int)])  # robust?
    spk = []
    # plot
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    tmax = -1
    for k in range(num_neurons):
        tspk = data[k]['spikes_nrn']
        ax1.plot(tspk, np.repeat(k, len(tspk)), 'ko')
        spk.extend(tspk)
        tmax = max(tmax, np.max(spk))
    tmax += binsz
    bins = np.arange(0, tmax, binsz)
    h = np.histogram(spk, bins=bins)[0]/(binsz/1000.*num_neurons)
    ax2.plot(bins[:-1], h, 'r-', lw=2)
    ax1.set_xlabel('time (ms)', fontsize=14)
    ax1.set_ylabel('cell', fontsize=14, color='k')
    ax2.set_ylabel('istantaneous firing rate (Hz)', fontsize=14, color='r')


def mean_firing_rate(fn_data, twindow=5000, figsize=(5, 5)):
    """Report mean firing rate."""
    data = nu.load_dict(fn_data)
    num_neurons = len([x for x in data if isinstance(x, int)])  # robust?
    plt.figure(figsize=figsize)
    spk = []
    for k in range(num_neurons):
        tspk = data[k]['spikes_nrn']
        spk.append(1e3*len(tspk)/twindow)
    plt.plot(spk, 'k-')
    plt.xlabel('cell #', fontsize=16)
    plt.ylabel('mean firing rate (Hz)', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)


def remove_synchronous_spikes(fn_data, binsz=1, mfr_thresh=20, dt_thresh=20):
    """Delete synchronous spikes.

    fn_data: file name
    binsz (ms):
    mfr_threshold (Hz):
    """
    data = nu.load_dict(fn_data)
    # need to add a key with number of neurons
    if 'num_neurons' in data:
        num_neurons = data['num_neurons']
    else:
        # in old fn_data num_neurons was not available!
        num_neurons = len([x for x in data if isinstance(x, int)])  # robust?
    spk = []
    tmax = -1
    for k in range(num_neurons):
        tspk = data[k]['spikes_nrn']
        spk.extend(tspk)
        tmax = max(tmax, np.max(spk))
    tmax += binsz
    bins = np.arange(0, tmax, binsz)
    in_firing = np.histogram(spk, bins=bins)[0]/(binsz/1000.*num_neurons)
    idx = np.where(in_firing > mfr_thresh)[0]
    tsyn = bins[idx]
    tsyn = np.hstack((tsyn, 1e12))
    tsyn_diff = np.diff(tsyn)
    idx2 = np.where(tsyn_diff > dt_thresh)[0]
    n_syn = len(idx2)

    i = 0
    syn_int = []
    for k in range(n_syn):
        syn_int.append((tsyn[i], tsyn[idx2[k]]))
        i = idx2[k] + 1
    for k in range(num_neurons):
        tspk = data[k]['spikes_nrn']
        for ti, te in syn_int:
            idx = np.where((tspk > ti) & (tspk < te))[0]
            tspk[idx] = -1
        data[k]['spikes_nrn'] = tspk[tspk > 0]
    return data
