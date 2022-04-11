number = len(reg_sel)
cmap = plt.get_cmap('jet')
colors = [cmap(i) for i in np.linspace(0, 1, number)]

norm = True

plt.figure(figsize=(10,10))

#jj = 7


#for k, reg in enumerate([reg_sel[jj]]):
for k, reg in enumerate(reg_sel):
    h = np.histogram(funct[reg]['exc'])
    if norm:
      norm_fact = np.sum(h[0])
    else:
      norm_fact = 1
    plt.plot(h[1][:-1], h[0]/norm_fact, label='reg = %g' % reg, lw=2, c=colors[k])

h = np.histogram(struct['exc'])
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
  plt.savefig(os.path.join(fpath_out,'link_length_freq.png'))
else:
  plt.savefig(os.path.join(fpath_out,'link_length_count.png'))
