number = len(reg_sel)
cmap = plt.get_cmap('jet')
colors = [cmap(i) for i in np.linspace(0, 1, number)]

norm = True
syn_type = 'inh'

plt.figure(figsize=(10,10))


reg_sel = [0.075, 1.0e-01, 2.0e-01, 3.0e-01, 5.0e-01, 0.8, 1.2e+00, 1.5e+00]
#reg_sel=[0.3]

for k, reg in enumerate(reg_sel):
    h = np.histogram(funct[reg][syn_type])
    if norm:
      norm_fact = np.sum(h[0])
    else:
      norm_fact = 1
    plt.plot(h[1][:-1], h[0]/norm_fact, label='reg = %g' % reg, lw=2, c=colors[k])

h = np.histogram(struct[syn_type])
if norm:
  norm_fact = np.sum(h[0])
else:
  norm_fact = 1
plt.plot(h[1][:-1], h[0]/norm_fact, label='structural', lw=4, c='k')

plt.legend(loc=0, fontsize=14)
plt.xlabel('link length', fontsize=14)
plt.ylabel('count', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

if norm:
  plt.savefig(os.path.join(fpath_out,'link_length_inh_freq.png'))
else:
  plt.savefig(os.path.join(fpath_out,'link_length_inh_count.png'))
