# network_analysis
import numpy as np
import pylab as plt
import networkx as nx
import network_util as nu

'''
> sincronia Kreuz
> MFR, IFR
> graph parameters ? networkx
 '''

figsize=(5, 5)


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
    # save in the same folder!!!


def mean_firing_rate(fn_data, time_trim=(100, 5000)):
    """Report mean firing rate."""
    data = nu.load_dict(fn_data)
    #
    if 'num_neurons' in data:
        num_neurons = data['num_neurons']
    else:
        # in old fn_data num_neurons was not available!
        num_neurons = len([x for x in data if isinstance(x, int)])  # robust?
    mfr = []
    twindow = time_trim[1] - time_trim[0]
    print('time window = %g ms' % twindow)
    for k in range(num_neurons):
        tspk = data[k]['spikes_nrn']
        tspk = tspk[(tspk > time_trim[0]) & (tspk < time_trim[1])]
        mfr.append(1e3*len(tspk)/twindow)
    return mfr


def remove_synchronous_spikes(fn_data, binsz=1, mfr_thresh=20, dt_thresh=20):
    """Delete synchronous spikes.

    fn_data: file name
    binsz (ms):
    mfr_threshold (Hz):
    """
    data = nu.load_dict(fn_data)
    #
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


def inter_connectivity(fn_data, nrn_patch, verbose=False):
    """Report inter-connectivity of a list of neurons.

    fn_data: filename of results with graph info
    nrn_patch: neuron list to check for
    """
    d = np.load(fn_data, allow_pickle=1).item()
    num_exc = d['params_neurons']['num_exc_neurons']
    num_inh = d['params_neurons']['num_neurons'] - num_exc
    exc_nrn = np.arange(num_exc)
    inh_nrn = np.arange(num_exc, num_inh)
    num_exc_patch = len(np.intersect1d(nrn_patch, exc_nrn))
    num_inh_patch = len(np.intersect1d(nrn_patch, inh_nrn))
    print('# exc neurons in the patch = %d' % num_exc_patch)
    print('# inh neurons in the patch = %d' % num_inh_patch)
    graph = nx.DiGraph(d['params_netw']['conn_mat'])
    len_patch = len(nrn_patch)
    # in_edges - patch
    print('in edges')
    count_in = 0
    for nrn_in in nrn_patch:
        nrn_from_graph = [edge[0]
                          for edge in graph.in_edges(nrn_in)]
        nrn_from_patch = np.intersect1d(nrn_from_graph, nrn_patch)
        lun_nrn_from_patch = len(nrn_from_patch)
        count_in += lun_nrn_from_patch
        if verbose:
            print('%d (%d)' % (lun_nrn_from_patch, len(nrn_from_graph)),
                  end=' ')
    print(' TOTAL %d ' % count_in)
    # out_edges - patch
    print('out edges')
    count_out = 0
    for nrn_out in nrn_patch:
        nrn_out_graph = [edge[1]
                         for edge in graph.out_edges(nrn_out)]
        nrn_out_patch = np.intersect1d(nrn_out_graph, nrn_patch)
        lun_out_from_patch = len(nrn_out_patch)
        count_out += lun_out_from_patch
        if verbose:
            print('%d (%d)' % (lun_nrn_from_patch, len(nrn_out_graph)),
                  end=' ')
    print(' TOTAL %d ' % count_out)
    print(count_in/len_patch, count_out/len_patch)
