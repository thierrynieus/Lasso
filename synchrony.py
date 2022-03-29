def spike_synchrony(fn,exc_ele,tlim=[0,3e5]):
    '''
    '''
    spikes,_=spont.LoadSpontData(fn,exc_ele)
    spk_trains={}
    for k,ch in enumerate(np.unique(spikes[0,:])): spk_trains[k]=pyspike.SpikeTrain(spikes[1,spikes[0,:]==ch],tlim)
    L=k+1 # num chan 
    sync_lst=[]
    for k in range(L):
        for j in range(k+1,L):
            sync_lst.append(pyspike.spike_sync(spk_trains[k],spk_trains[j]))
    return np.array(sync_lst)
