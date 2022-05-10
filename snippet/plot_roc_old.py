def plot_roc_old(params):
    """Plot Receiver Operative Curve."""
    def ratio(a, b):
        s = a + b
        return a/s if s else 1.
    # sensitivity
    sensitivity = []
    sensitivity_exc = []
    sensitivity_inh = []
    # specificity
    specificity = []
    specificity_exc = []
    specificity_inh = []
    #
    for reg in params['reg_vect']:
        fname_lasso = os.path.join(resultsfolder, params['rel_path_results'],
                                   'reg_%g' % reg, params['fname_conf_mat'])
        d = np.load(fname_lasso, allow_pickle=1).item()
        # all
        sensitivity.append(ratio(d['tp'], d['fn']))
        specificity.append(ratio(d['tn'], d['fp']))
        print(reg, 1-specificity[-1], sensitivity[-1])
        # exc
        sensitivity_exc.append(ratio(d['tp_exc'], d['fn_exc']))
        specificity_exc.append(ratio(d['tn_exc'], d['fp_exc']))
        # inh
        sensitivity_inh.append(ratio(d['tp_inh'], d['fn_inh']))
        specificity_inh.append(ratio(d['tn_inh'], d['fp_inh']))

    plt.figure(figsize=(6, 6))
    # all
    oneminus_specificity = 1-np.array(specificity)
    plt.plot(oneminus_specificity, sensitivity, 'ks-', markersize=10,
             label='all', lw=2)
    # exc
    oneminus_specificity_exc = 1-np.array(specificity_exc)
    plt.plot(oneminus_specificity_exc, sensitivity_exc, 'ro--', markersize=10,
             label='excitatory')
    # inh
    oneminus_specificity_inh = 1-np.array(specificity_inh)
    plt.plot(oneminus_specificity_inh, sensitivity_inh, 'bo--', markersize=10,
             label='inhibitory')
    # decorate
    plt.xlabel('false positive rate', fontsize=16)
    plt.ylabel('true positive rate', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.tight_layout(pad=1)

    fname_png = os.path.join(resultsfolder, params['rel_path_results'],
                             'plot_ROC.png')
    plt.savefig(fname_png)
