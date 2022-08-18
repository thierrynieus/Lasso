# network_analysis
import numpy as np
import matplotlib
#matplotlib.rcParams['svg.fonttype']
matplotlib.rcParams['svg.fonttype'] = 'none'
import pylab as plt
import networkx as nx
import os
import pandas as pd

import network_util as nu
import network_analysis as na


'''
> sincronia Kreuz
> MFR, IFR
> graph parameters ? networkx
 '''

colormap = 'jet'
fs_lab = 14
fs_ticks = 12
fs_legend = 12
ms = 10  # markersize
fig_type = '.png'


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


def plot_one_trace(fn_data, num_neuron=0):
    """Plot one trace well."""
    data = nu.load_dict(fn_data)
    if not('vm' in list(data[0])):
        print('>>> Error: raw traces are not available!')
        return None
    # source neurons
    conn_mat = np.array(data['params_netw']['conn_mat'])
    idx_src = np.where(conn_mat[:, 1] == num_neuron)[0]
    nrn_src = np.unique(conn_mat[idx_src, 0])

    plt.figure(figsize=(20, 20))
    vm = data[num_neuron]['vm']['V_m']
    plt.plot(data[num_neuron]['vm']['times'], vm, 'k-', lw=2)
    for src in nrn_src:
        col = 'r' if src in data['params_netw']['exc'] else 'b'
        time_lst = list(data[src]['vm']['times'])
        tspk_all_input = data[src]['spikes_nrn']
        #  network inputs
        idx = np.array([time_lst.index(tspk_in) for tspk_in in tspk_all_input])
        plt.plot(data[src]['vm']['times'][idx], vm[idx], c=col, marker='^',
                 markersize=8, alpha=0.5, lw=0)
        #  noisy events
        tspk_all_noise = data[src]['spikes_noise']
        idx = np.array([time_lst.index(tspk_in) for tspk_in in tspk_all_noise])
        plt.plot(data[src]['vm']['times'][idx], vm[idx], c='k', marker='o',
                 markersize=4, alpha=0.5, lw=0)
    plt.xlabel('time (ms)', fontsize=fs_lab)
    plt.ylabel('Voltage (mV)', fontsize=fs_lab)
    plt.xticks(fontsize=fs_ticks)
    plt.yticks(fontsize=fs_ticks)


def raster_plot(fn_data, binsz=20, figsize=(4, 4), lw=1, ms=6):
    """Produce a raster plot."""
    data = nu.load_dict(fn_data)
    num_neurons = len([x for x in data if isinstance(x, int)])  # robust?
    spk = []
    # plot
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()
    tmax = -1
    for k in range(num_neurons):
        tspk = data[k]['spikes_nrn']
        ax1.plot(tspk, np.repeat(k, len(tspk)), 'ko', markersize=ms)
        spk.extend(tspk)
        tmax = max(tmax, np.max(spk))
    tmax += binsz
    bins = np.arange(0, tmax, binsz)
    h = np.histogram(spk, bins=bins)[0]/(binsz/1000.*num_neurons)
    ax2.plot(bins[:-1], h, 'r-', lw=lw)
    ax1.set_xlabel('time (ms)', fontsize=fs_lab)
    ax1.set_ylabel('cell', fontsize=fs_lab, color='k')
    ax2.set_ylabel('istantaneous firing rate (Hz)', fontsize=fs_lab, color='r')
    ax1.set_yticklabels([], fontsize=fs_ticks)
    #  ax1.tick_params(axis='x', size=fs_ticks)
    ax1.tick_params(axis='y', size=fs_ticks)
    plt.tight_layout(pad=1)
    # save in the same folder!!!
    fn_fig = os.path.join(os.path.dirname(fn_data), 'raster_plot' + fig_type)
    plt.savefig(fn_fig)


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
    #print('time window = %g ms' % twindow)
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


def stat_distance(fn_out, fn_reg):
    """Report statistics on functional/structural link lenght.

    input:
        fn_out  output.py
        fn_reg  reg_XYZ/RSmat_lasso.py
    output:
        stat_dist
    """
    from scipy.spatial.distance import pdist, squareform

    d = np.load(fn_out, allow_pickle=1).item()
    lasso_mat = np.load(fn_reg, allow_pickle=1).item()['beta']
    #  d['params_netw']) -> ['num_neurons', 'exc', 'inh', 'conn_mat', 'x', 'y']

    num_neurons = d['params_netw']['num_neurons']
    conn_mat = np.array(d['params_netw']['conn_mat'], 'int')
    pos = np.zeros((num_neurons, 2))
    pos[:, 0] = d['params_netw']['x']
    pos[:, 1] = d['params_netw']['y']
    dist_mat = squareform(pdist(pos))

    # structural and functional links
    stat_dist = {}
    for conn_type in ['structural', 'functional']:
        stat_dist[conn_type] = {}
        for syn_type in ['exc', 'inh']:
            stat_dist[conn_type][syn_type] = []

    # structural
    for syn_type in ['exc', 'inh']:
        for nrn in d['params_netw'][syn_type]:
            idx_nrn_src = np.where(conn_mat[:, 0] == nrn)[0]
            idx_nrn_dst = conn_mat[idx_nrn_src, 1]
            stat_dist['structural'][syn_type].extend(dist_mat[nrn,
                                                              idx_nrn_dst])
    # functional
    mixed_detect = {}
    num_detect = {}
    for syn_type in ['exc', 'inh']:
        mixed_detect[syn_type] = []
        num_detect[syn_type] = []
        syn_sign = 1 if syn_type == 'exc' else -1
        for nrn in d['params_netw'][syn_type]:
            idx_nrn_dst = np.where((lasso_mat[nrn, :]) == syn_sign)[0]
            stat_dist['functional'][syn_type].extend(dist_mat[nrn,
                                                     idx_nrn_dst])
            #  determine the amount of mixed connections (E->I, I->E)
            idx_nrn_dst = np.where((lasso_mat[nrn, :]) == -syn_sign)[0]
            mixed_detect[syn_type].append(len(idx_nrn_dst))
            idx_nrn_dst = np.where(lasso_mat[nrn, :])[0]
            num_detect[syn_type].append(len(idx_nrn_dst))
    return stat_dist, mixed_detect, num_detect


def stat_distance_dist(fpath_out, reg_vect=None, bins=50):
    """Extract statistics on link lengths.

    input:
        fpath_out   output.py (output of simulation)
        reg_vect    array of regularization coefficients
    """
    if reg_vect is None:
        reg_vect = nu.get_regularization_factor(fpath_out)

    fn_out = os.path.join(fpath_out, 'output.npy')

    funct = {}
    hist_conn = {}
    mixed_conn = {}
    # functional
    for reg_coeff in reg_vect:
        fn_reg = os.path.join(fpath_out, 'reg_%g/RSmat_lasso.npy' % reg_coeff)
        stat_dist, mixed_detect, _ = stat_distance(fn_out, fn_reg)
        # stat_dist, mixed_detect, num_detect
        #
        funct[reg_coeff] = stat_dist['functional']
        hist_conn[reg_coeff] = {}
        mixed_conn[reg_coeff] = {}
        for syn_type in ['exc', 'inh']:
            f_data = funct[reg_coeff][syn_type]
            hist_conn[reg_coeff][syn_type] = np.histogram(f_data, bins=bins)[0]
            mixed_conn[reg_coeff][syn_type] = mixed_detect[syn_type]
    # structural
    hist_conn['structural'] = {}
    for syn_type in ['exc', 'inh']:
        f_data = stat_dist['structural'][syn_type]
        h = np.histogram(f_data, bins=bins)
        hist_conn['structural'][syn_type] = h[0]
    # center bins
    hist_conn['bins'] = h[1][:-1] + .5 * (h[1][1] - h[1][0])
    # prepare output
    dout = {}
    dout['structural'] = stat_dist['structural']
    dout['functional'] = funct
    dout['hist_conn'] = hist_conn
    dout['mixed_conn'] = mixed_conn
    return dout


def plot_stat_link_stat(dout, fpath_fig, norm=True):
    """Plot link lenght statistics.

    dout generated by stat_distance_dist
    """
    reg_vect = list(dout['functional'])
    bins = dout['hist_conn']['bins']
    number = len(reg_vect)
    cmap = plt.get_cmap(colormap)
    colors = [cmap(i) for i in np.linspace(0, 1, number)]
    #  *** summary plot ***
    plt.figure(1, figsize=(20, 10))
    # functional
    for k, reg in enumerate(reg_vect):
        for j, syn_type in enumerate(['exc', 'inh']):
            h = dout['hist_conn'][reg][syn_type]
            if norm:
                h = h / np.sum(h)
            plt.subplot(1, 2, j+1)
            plt.plot(bins, h, label='reg = %g' % reg, lw=2, c=colors[k])
    # structural
    if norm:
        ylabel = 'frequency'
    else:
        ylabel = 'count'
    for j, syn_type in enumerate(['exc', 'inh']):
        h = dout['hist_conn']['structural'][syn_type]
        if norm:
            h = h / np.sum(h)
        plt.subplot(1, 2, j+1)
        plt.plot(bins, h, label='structural', lw=4, c='k')
        plt.legend(loc=0, fontsize=14)
        # decorate
        plt.title(syn_type, fontsize=20)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('link length', fontsize=18)
        plt.ylabel(ylabel, fontsize=18)
    plt.savefig(os.path.join(fpath_fig, 'link_length_summary%s' % fig_type))

    #  *** single plots ***
    count_fig = 2
    for reg in reg_vect:
        plt.figure(count_fig, figsize=(20, 10))
        count_fig += 1
        for j, syn_type in enumerate(['exc', 'inh']):
            # functional
            h = dout['hist_conn'][reg][syn_type]
            if norm:
                h = h / np.sum(h)
            plt.subplot(1, 2, j+1)
            plt.plot(bins, h, label='reg = %g' % reg, lw=2, c='r')
            # structural
            h = dout['hist_conn']['structural'][syn_type]
            if norm:
                h = h / np.sum(h)
            plt.plot(bins, h, label='structural', lw=4, c='k')
            plt.legend(loc=0, fontsize=14)
            # decorate
            plt.title(syn_type, fontsize=20)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.xlabel('link length', fontsize=18)
            plt.ylabel(ylabel, fontsize=18)
        plt.savefig(os.path.join(fpath_fig,
                                 'link_length_reg=%g%s' % (reg, fig_type)))

    #  *** mixed connections ***
    mix_stat = {}
    for syn_type in ['exc', 'inh']:
        mix_stat[syn_type] = {}
        mix_stat[syn_type]['mean'] = []
        mix_stat[syn_type]['std'] = []

    for reg in reg_vect:
        for syn_type in ['exc', 'inh']:
            data = dout['mixed_conn'][reg][syn_type]
            mix_stat[syn_type]['mean'].append(np.mean(data))
            mix_stat[syn_type]['std'].append(np.std(data))

    plt.figure(count_fig, figsize=(20, 10))
    for j, syn_type in enumerate(['exc', 'inh']):
        plt.subplot(1, 2, j+1)
        y = mix_stat[syn_type]['mean']
        yerr = mix_stat[syn_type]['std']
        plt.errorbar(x=reg_vect, y=y, yerr=yerr, elinewidth=2, ecolor='k',
                     marker='o', color='k', linewidth=2)
        # decorate

        plt.grid()
        plt.title(syn_type, fontsize=20)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('regularization', fontsize=18)
        plt.ylabel('# missmatches / neuron', fontsize=18)
        plt.ylim(ymin=-0.05)
    plt.savefig(os.path.join(fpath_fig, 'missmatches%s' % fig_type))


def inter_group_connections(conn_mat, groups):
    """Compute the amount of inter group connections.

    note: of interest with clusters
    example: groups=[np.arange(50), 50+np.arange(50)]
    """
    count_tot = 0
    count_dct = {}
    for ngr, group in enumerate(groups):
        count = 0
        for conn in conn_mat:
            if (conn[0] in group) & (conn[1] in group):
                count += 1
        count_tot += count
        count_dct[ngr] = count
    count_across = len(conn_mat) - count_tot
    return count_dct, count_across


def plot_graph(fpath_out, reg, nodelist=None, ew_max=8, node_size=800,
               arrsize=20, pos=None):
    """Plot graph.

    useful:
        nx.write_gpickle(G,'graph.gpickle')
        H=nx.read_gpickle('graph.gpickle')
    """
    # global pos
    data = nu.load_dict(os.path.join(fpath_out, 'output.npy'))
    conn_mat_struct = data['params_netw']['conn_mat']
    beta = nu.load_dict(os.path.join(fpath_out, 'reg_%g' % reg,
                                     'RSmat_lasso.npy'))['beta']
    if 'x' in list(data['params_netw']):
        pos = [(x, y) for x, y in zip(data['params_netw']['x'],
                                      data['params_netw']['y'])]
    elif pos is None:
        print('The coordinates are absent. Generating random (x,y)!')
        pos = [tuple(np.random.rand(2))
               for k in range(data['params_netw']['num_neurons'])]
    # functional graph
    r_all, c_all = np.where(beta)

    """
    # for subgraph
    https://stackoverflow.com/questions/29838746/how-to-draw-subgraph-using-networkx

    other softwares
    https://gephi.org/screenshots/
    https://www.graphviz.org/

    """

    """
    if nodelist is not None:
        r_all_tmp, c_all_tmp = [], []
        for r, c in zip(r_all, c_all):
            if (r in nodelist):
                r_all_tmp.append(r)
                c_all_tmp.append(c)
        r_all, c_all = r_all_tmp.copy(), r_all_tmp.copy()
    """

    """
    conn_mat_funct = []
    for r, c in zip(r_all, c_all):
        if (r in nodelist):  # | (c in nodelist):
            conn_mat_funct.append((r, c))
    """
    conn_mat_funct = [(r, c) for r, c in zip(r_all, c_all)]
    sign_syn = beta[r_all, c_all]
    graph = nx.DiGraph(conn_mat_funct)
    color_map = ['red' if node in data['params_netw']['exc'] else 'blue'
                 for node in graph]
    edge_widths = []
    edge_colors = []
    for conn, sign in zip(conn_mat_funct, sign_syn):
        ew = ew_max if conn in conn_mat_struct else 1
        edge_widths.append(ew)
        """
        if ew == 1:
            col = 'k' if sign == 1 else 'green'
        else:
        """
        col = 'r' if sign == 1 else 'b'
        edge_colors.append(col)

    # draw graph
    #  fig = plt.figure(1, figsize=(200, 80), dpi=60)
    nx.draw(graph, with_labels=True, pos=pos, node_color=color_map,
            width=edge_widths, node_size=node_size, edge_color=edge_colors,
            arrowsize=arrsize)  # , arrowstyle='fancy')
    return pos


def zoom_graph(fpath_out, nrn=0, hemi=0.1):
    """Zoom on a specific neuron."""
    data = nu.load_dict(os.path.join(fpath_out, 'output.npy'))
    x = data['params_netw']['x'][nrn]
    y = data['params_netw']['y'][nrn]
    plt.xlim(x-hemi, x+hemi)
    plt.ylim(y-hemi, y+hemi)


def cross_corr_fast(spk_train1, spk_train2, params):
    """Compute spike cross-correlation among two spike trains.

    The algorithm implements the somehow trivial idea spike-cross-correlation
    measures the amount of coincidences for shifted vectors.
    The code is houndreds of times faster than more naive algorithms.

    spk_train1, spk_train2:  spike train list
    params (dictionary):
        dt
        tw
        tmax
        corr
    """
    st1 = (np.array(spk_train1)/params['dt']).astype(int)
    st2 = (np.array(spk_train2)/params['dt']).astype(int)
    l1 = len(st1)
    l2 = len(st2)
    l12 = l1 * l2
    nstep = int(params['tw'] / params['dt'])
    cc_vect = np.zeros(2 * nstep + 1)
    for k in range(-nstep, nstep+1):
        cc_vect[nstep+k] = len(np.intersect1d(st1, st2+k, True))
    if params['corr']:
        cc_vect -= (l12 / params['tmax']) * params['dt']
    cc_vect /= np.sqrt(l12)
    return cc_vect[::-1]
