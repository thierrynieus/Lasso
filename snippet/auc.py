def area_under_curve(uncon_val, con_val, nsteps=1000):
    """Compute area under curve.
    
    note:
        add (FPR,TPR)=(1,1)
    """
    num_cons = len(con_val)
    num_uncon = len(uncon_val)
    threshold_rng = np.linspace(0, 1, nsteps)
    tp_rate = np.zeros(nsteps)
    fp_rate = np.zeros(nsteps)
    for i, threshold in enumerate(threshold_rng):
        tp = (con_val > threshold).sum()
        tn = (uncon_val < threshold).sum()
        fp = num_cons - tn
        fn = num_uncon - tp
        tpn = tp + fn
        fpn = fp + tn
        if tpn:
            tp_rate[i] = tp / tpn
        if fpn:
            fp_rate[i] = fp / fpn
    idx_sort = np.argsort(fp_rate)
    auc = np.trapz(tp_rate[idx_sort], fp_rate[idx_sort])
    return auc
