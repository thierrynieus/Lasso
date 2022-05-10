
fn_out = '/home/tnieus/Projects/RESULTS/Lasso/100nrn_80exc_20inh_spatial/0001/output.npy'

d = nu.load_dict(fn_out)
#dout = nu.load_dict('/home/tnieus/Projects/RESULTS/Lasso/100nrn_80exc_20inh_spatial/0001/subsampled_neurons/m
#  icro_circuit_mpl2/0001/dout.npy')

idx=np.where(dout['paths3'][:,1]==dout['nrn_exclude'])[0]

idx_src = dout['paths3'][idx,0]
idx_dst = dout['paths3'][idx,2]

params={}
params['dt'] = 1
params['tw'] = 20
params['tmax'] = 9500
params['corr'] = True

fpath_cc = '/home/tnieus/Projects/RESULTS/Lasso/100nrn_80exc_20inh_spatial/0001/subsampled_neurons/micro_circuit_mpl2/0001/cross_corr/mpl3'

cc_lst = []
for src, dst in zip(idx_src, idx_dst):
  cc = na.cross_corr_fast(d[src]['spikes_nrn'], d[dst]['spikes_nrn'], params)
  cc_lst.append((src,dst,cc))

x = np.arange(-params['tw'],params['tw']+params['dt'],params['dt'])

out={}
out['x'] = x
out['cc_lst'] = cc_lst
np.save(os.path.join(fpath_cc,'cc_out.npy'), out)

plot_it=True
if plot_it:
  plt.ioff()

  for cc in cc_lst:
    plt.plot(x,cc[2],'k-')
    stit = '(%d, %d) ' % (cc[0], cc[1])
    plt.title(stit, fontsize=18)
    plt.grid()
    plt.savefig(os.path.join(fpath_cc,'%s.png' % stit))
    plt.close()

fpath_cc = '/home/tnieus/Projects/RESULTS/Lasso/100nrn_80exc_20inh_spatial/0001/subsampled_neurons/micro_circuit_mpl2/0001/cross_corr/mpl2'

cc_lst = []
dst = dout['nrn_exclude']
for src in np.unique(idx_src):
  cc = na.cross_corr_fast(d[src]['spikes_nrn'], d[dst]['spikes_nrn'], params)
  cc_lst.append((src,dst,cc))

  plot_it=True
  if plot_it:
    plt.ioff()

    for cc in cc_lst:
      plt.plot(x,cc[2],'k-')
      stit = '(%d, %d) ' % (cc[0], cc[1])
      plt.title(stit, fontsize=18)
      plt.grid()
      plt.savefig(os.path.join(fpath_cc,'%s.png' % stit))
      plt.close()

out={}
out['x'] = x
out['cc_lst'] = cc_lst
np.save(os.path.join(fpath_cc,'cc_out.npy'), out)


plt.ion()
