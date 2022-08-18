"""
fname='/home/tnieus/Projects/RESULTS/Lasso/paper/20nrn_16exc_4inh/0001/output.npy'

In [3]: run -i lasso.py
d=n
In [4]: d=nu.load_dict(fname)

conn_mat=np.array(d['params_netw']['conn_mat'])

a,b=np.unique(conn_mat[:,1],return_counts=True)

In [19]: for ai in a:
    ...:     idx=np.where(conn_mat[:,1]==ai)[0]
    ...:     print(ai,conn_mat[idx,0])
    
    
0 [ 2  9 15 16]
1 [ 2  3  4  8 10 12 14 16 18]
2 [ 0  4  6  8 11 14 16 18 19]
3 [11 15]
4 [ 1  2  3  5  7 10 14 15 16]
5 [ 0  2  6  8 11 15 17 18]
6 [ 1  3  7 13]
7 [ 9 11 12 17 18 19]
8 [ 3  4 11 12 16]
9 [ 2  8 11 14 16 17]
10 [ 1  4  5  7  8 15 16 19]
11 [ 0  2  7  9 10 13 14 15 17]
12 [ 1  3  6  8  9 15 16]
13 [ 0  1  4 10 11 14 15 16]
14 [ 1  4  8 13 15 16 17 19]
15 [ 0  2  9 12 13 16]
16 [ 0  3  5  7 11 14 15 17 19]
17 [ 2  9 14 15]
18 [ 0  4  5  9 11]
19 [ 1  2  4  6  9 13 14]

"""

figsize=(4,4)

delay=0.1
delay_n=1
ms=6
ms_noise=6
alpha=0.5

#lun=len(d[0]['vm']['V_m'])
#t=np.arange(lun)*.1
t=d[0]['vm']['times']

dst=16#9#5
src_e=0#2#0
src_i=19#18

plt.figure(figsize=figsize)

ax1=plt.subplot(311)
ax1.plot(t,d[dst]['vm']['V_m'],'k-')
for tk in d[src_e]['spikes_nrn']:
    idx = np.where(t==tk)[0][0]
    plt.plot(tk+delay,d[dst]['vm']['V_m'][idx],c='r',marker='o',markersize=ms,lw=2,alpha=alpha)
for tk in d[src_i]['spikes_nrn']:
    idx = np.where(t==tk)[0][0]
    plt.plot(tk+delay,d[dst]['vm']['V_m'][idx],c='b',marker='o',markersize=ms,lw=2,alpha=alpha)
for tk in d[dst]['spikes_noise']:
    idx = np.where(t==tk+delay_n)[0][0]
    plt.plot(tk+delay_n,d[dst]['vm']['V_m'][idx],c='gray',marker='o',markersize=ms_noise,lw=2,alpha=alpha)
#plt.xlabel('time (ms)',fontsize=16)
plt.ylabel('V (mV)',fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylim(-60,-48)

ax2=plt.subplot(312, sharex=ax1)
ax2.plot(t,d[src_e]['vm']['V_m'],'r-')
#plt.xlabel('time (ms)',fontsize=16)
plt.ylabel('V (mV)',fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

ax3=plt.subplot(313, sharex=ax1)
ax3.plot(t,d[src_i]['vm']['V_m'],'b-')
plt.xlabel('time (ms)',fontsize=16)
plt.ylabel('V (mV)',fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.tight_layout(pad=1)
plt.xlim(4425,4650)


