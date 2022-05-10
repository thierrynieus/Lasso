def plot_matthew_vs_regularization(params_roc, label='', lw=2, fignum=1):
    """Plot Matthew coefficient versus regularization.

    params_roc

    """
    fname_lasso = os.path.join(resultsfolder, params_roc['rel_path_results'],
                               'confusion_mat.csv')
    df = pd.read_csv(fname_lasso)
    # plot
    plt.figure(fignum, figsize=figsize)

    # all
    mc = matthew_coeff(select_conf_mat(df, 'all'))
    plt.plot(df['regularization_strength'], mc, 'ks-', markersize=10,
             label='all %s' % label, lw=lw)
    # exc
    mc = matthew_coeff(select_conf_mat(df, 'exc'))
    plt.plot(df['regularization_strength'], mc, 'ro--', markersize=10,
             label='excitatory %s' % label, lw=lw)
    # inh
    mc = matthew_coeff(select_conf_mat(df, 'inh'))
    plt.plot(df['regularization_strength'], mc, 'bo--', markersize=10,
             label='inhibitory %s' % label, lw=lw)
    plt.legend(loc=0, fontsize=14)
    #  decorate
    plt.xlabel('regularization strength', fontsize=16)
    plt.ylabel('Matthew coefficient', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.tight_layout(pad=1)
    fname_png = os.path.join(resultsfolder, params_roc['rel_path_results'],
                             'plot_Matthew.png')
    plt.savefig(fname_png)
