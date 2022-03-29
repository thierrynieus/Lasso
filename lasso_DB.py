import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
import os
from network_util import load_dict
import pandas as pd

resultsfolder = '/home/tnieus/Projects/RESULTS/Lasso/'
configfolder = '/home/tnieus/Projects/CODE/Lasso/config/'

params_lasso = {'regularization_strength': 1, 'subfolder': 'test',
                'fname_RSmat': 'RSmat_0014.npy',
                'fname_lasso': 'RSmat_0014_lasso.npy',
                'skip_existent': True, 'max_iter': 300}

params_conf_mat = {'regularization_strength': 0.1, 'subfolder': 'test',
                   'fname_lasso': 'RSmat_0015_lasso.npy',
                   'fname_netw': 'params_netw.npy',
                   'foldercfg': '20nrn_16exc_4inh/0000',
                   'fname_conf_mat': 'RSmat_0015_conf_mat.npy'}

params_roc = {'subfolder': 'test', 'fname_conf_mat': 'RSmat_0015_conf_mat.npy',
              'reg_vect': [0.0001, 0.001, 0.005, 0.0075, 0.01, 0.025, 0.05,
                           0.075, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 3, 4, 5, 5.2,
                           5.5, 6, 6.1, 6.2, 6.5, 6.6, 6.7]}


def calc_mat_processes(fn_data, fn_cfg, fn_out, time_trim=(100, 5000), dt_dis=1):
    """Computes the matrices of the internal (S) and external (R) events

    data[k]['spikes_nrn']
    data[k]['spikes_noise']
    conn a list of connections (src,dst)
    time_trim discard outside interval
    dt_dist discretization
    """
    def discretize(spk, tmin, tmax, dt_dis):
        '''
        '''
        spk1 = spk.copy()
        spk1 = spk1[spk1 < tmax]
        spk1 = spk1[spk1 > tmin] - tmin
        return np.round(spk1 / dt_dis).astype(int)

    data = load_dict(os.path.join(resultsfolder, fn_data))
    config = load_dict(os.path.join(configfolder, fn_cfg))

    conn_mat = np.array(config['conn_mat'])
    num_neurons = config['num_neurons']
    tmin, tmax = time_trim
    ntime = int((tmax - tmin) / dt_dis)
    mat_ext = np.zeros((num_neurons, ntime), dtype=np.int8)
    mat_int = np.zeros((num_neurons, ntime), dtype=np.int8)
    # mat_ext_dic = {} # lists, dictionary for sparse representation

    # external (spikes)
    for k in range(num_neurons):
        idx = discretize(data[k]['spikes_nrn'], tmin, tmax, dt_dis)
        mat_ext[k, idx] = 1

    # internal (epsp, ipsp, noise epsp)
    for k in range(num_neurons):
        #  process endogeneous noise epsp events
        idx = discretize(data[k]['spikes_noise'], tmin, tmax, dt_dis)
        mat_int[k, idx] = 1
        #  all inputs to neuron k
        idx_in_conn = np.where(conn_mat[:, 1] == k)[0]
        for idx_in in idx_in_conn:
            nrn_in = conn_mat[idx_in, 0]  # nrn_in to k
            sign = 1 if nrn_in in config['exc'] else -1
            # conn_mat[idx_in, 0] connections to neuron k
            idx_nz = np.where(mat_ext[nrn_in, :])[0]
            ''' possibly do not update non-null mat_int values
            '''
            mat_int[k, idx_nz] = sign

    np.save(os.path.join(resultsfolder, fn_out), {'R': mat_int, 'S': mat_ext})


def lasso(params):
    '''
    load R and S matrices
    '''
    # Inverso del coefficiente di regolarizzazione (lambda^{-1})
    reg_strength = params['regularization_strength']

    # output folder
    fpath_out = os.path.join(resultsfolder, params['subfolder'],
                             'reg_%g' % reg_strength)
    if not(os.path.isdir(fpath_out)):
        os.mkdir(fpath_out)
    fname_lasso = os.path.join(fpath_out, params['fname_lasso'])
    # skip the analysis if skip_existent is TRUE and file already exists
    if os.path.isfile(fname_lasso) & params['skip_existent']:
        return None

    # load data
    fn = os.path.join(resultsfolder, params['subfolder'],
                      params['fname_RSmat'])
    data = np.load(fn, allow_pickle=1).item()
    R = data['R']
    S = data['S']

    #
    n, m = S.shape   # number of neurons (N) and observations (M)
    Y = R.reshape(-1, 1)  # response vector

    # Definiamo la trasposta di S
    ST = np.transpose(S)

    # Definiamo X a partire da ST
    X = ST
    for i in range(n-1):  # Si può evitare questo ciclo for??
        X = block_diag(X, ST)

    '''
    nota1
    X=block_diag([ST for i in range(N)]) # una comprehensive list è più veloce
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.block_diag.html
    nota2
    implementazione per matrici sparse
    https://stackoverflow.com/questions/30895154/scipy-block-diag-of-a-list-of-matrices
    '''

    ############################################################################################################

    # Importiamo libreria per Lasso Logistico
    from sklearn.linear_model import LogisticRegression

    '''
    Definisco pesi \omega_{i,m}=\sum_{i,m} y[i,m] se y[i,m]=0 oppure \omega_{i,m}=\sum_{i,m} (1-y[i,m]) se y[i,m]=1
      con i in {1,...,N} e m in {1,...,M}.
      Inoltre li normalizzo altrimenti devo aumentare i parametri tol o max_iter per far convergere il metodo.
    '''

    nm = n * m
    omega0 = np.count_nonzero(Y)/nm
    omega1 = (nm-np.count_nonzero(Y == 1))/nm
    omegam1 = (nm - np.count_nonzero(Y == -1))/nm
    #print(omega0,omega1,omegam1)

    model = LogisticRegression(
        # equivalente all'uso di l1_ratio=1 significa che usiamo penalizzazione Lasso
        penalty='l1',
        class_weight={0: omega0, 1: omega1, -1: omegam1},
        # se i pesi non sommano a 1, cambia automaticamente il parametro C di penalizz.
        solver='saga',
        #  o 'liblinear' che è meno efficiente per grandi dataset ma più efficiente per piccoli dataset
        #  eventualmente saga va bene anche per Ridge ed ElasticNet
        multi_class='multinomial',
        max_iter=params['max_iter'],
        #  tol=0.0001,
        C=reg_strength)   # regularization_strength è l'inverso di lambda coeff per la penalizzazione l1

    model.fit(X, Y.ravel())

    # Coefficienti del modello
    alpha = model.coef_
    # Coefficienti alpha relativi ai casi -1, 0 e 1 rispettivamente
    alpha_meno_uno = np.transpose(alpha[0].reshape(n, n))
    alpha_zero = np.transpose(alpha[1].reshape(n, n))
    alpha_uno = np.transpose(alpha[2].reshape(n, n))

    # Coefficienti beta
    beta_uno = np.zeros(shape=(n, n), dtype='int8')
    beta_zero = np.zeros(shape=(n, n), dtype='int8')
    beta_meno_uno = np.zeros(shape=(n, n), dtype='int8')

    """ Selezioniamo i valori positivi (in alternativa selezioniamo i valori
    del range 30% più alto) """
    coef_threshold = 0
    beta_meno_uno[alpha_meno_uno > coef_threshold] = 1
    beta_zero[alpha_zero > coef_threshold] = 1
    beta_uno[alpha_uno > coef_threshold] = 1

    '''
    beta dev'essere -1 per beta_meno_uno = 1
        e dev'essere 1 per beta_meno_uno = 0 and beta_uno = 1
    beta = beta_uno - beta_meno_uno
    beta = np.zeros(shape=(n, n))
    '''
    beta = np.zeros(shape=(n, n), dtype='int8')
    beta[beta_meno_uno == 1] = -1
    beta[np.logical_and(beta_meno_uno == 0, beta_uno == 1)] = 1

    dout = {}
    dout['beta'] = beta
    dout['regularization_strength'] = reg_strength
    # save data
    np.save(fname_lasso, dout)

# params for confusion_matrix


def confusion_matrix(params):
    '''
    '''
    # load data

    fname_netw = os.path.join(configfolder, params['foldercfg'], params['fname_netw'])
    folder_reg = 'reg_%g' % params['regularization_strength']
    fname_lasso = os.path.join(resultsfolder, params['subfolder'], folder_reg,
                               params['fname_lasso'])
    data_netw = np.load(fname_netw, allow_pickle=1).item()
    data_lasso = np.load(fname_lasso, allow_pickle=1).item()

    # ADJ = dIN_adj['matriceADJ']
    # build adjancency matrix
    src, dst = np.array(data_netw['conn_mat']).T  # -1, 1, 0
    idx_exc = np.isin(src, data_netw['exc'])
    idx_inh = np.isin(src, data_netw['inh'])
    n = data_netw['num_neurons']
    adj_mat = np.zeros((n, n), dtype='int8')
    adj_mat[src[idx_exc], dst[idx_exc]] = 1
    adj_mat[src[idx_inh], dst[idx_inh]] = -1

    beta = data_lasso['beta']
    reg_strength = data_lasso['regularization_strength']

    # Matrice dei risultati a valori in {-3, -2, -1, 0, 1, 2, 3}
    res_mat = np.zeros((n, n), dtype='int8')
    res_mat[np.logical_and(adj_mat == 1, beta == -1)] = 2
    res_mat[np.logical_and(adj_mat == 1, beta == 0)] = 1
    res_mat[np.logical_and(adj_mat == -1, beta == 1)] = -2
    res_mat[np.logical_and(adj_mat == -1, beta == 0)] = -1
    res_mat[np.logical_and(adj_mat == 0, beta == 1)] = 3
    res_mat[np.logical_and(adj_mat == 0, beta == -1)] = -3

    # Calcolo falsi positivi
    fp_exc = np.count_nonzero(res_mat == 3) + np.count_nonzero(res_mat == -2)
    fp_inh = np.count_nonzero(res_mat == -3) + np.count_nonzero(res_mat == 2)

    # Calcolo falsi negativi
    fn_exc = np.count_nonzero(res_mat == 1) + np.count_nonzero(res_mat == 2)
    fn_inh = np.count_nonzero(res_mat == -1) + np.count_nonzero(res_mat == -2)

    # Calcolo veri negativi
    tn_exc = np.count_nonzero(res_mat == 3)
    tn_inh = np.count_nonzero(res_mat == -3)

    #  Calcolo veri positivi
    tp_exc = np.count_nonzero(adj_mat == 1) - fp_exc - fn_exc
    tp_inh = np.count_nonzero(adj_mat == -1) - fp_inh - fn_inh

    fp = fp_exc + fp_inh
    fn = fn_exc + fn_inh
    tp = tp_exc + tp_inh
    tn = np.count_nonzero(adj_mat == 0) - tn_exc - tn_inh
    #tn = tn_exc + tn_inh

    # save results
    dout = {}
    dout['regularization_strength'] = reg_strength
    # all
    dout['tp'] = tp
    dout['tn'] = tn
    dout['fp'] = fp
    dout['fn'] = fn
    # exc
    dout['tp_exc'] = tp_exc
    dout['tn_exc'] = tn_exc
    dout['fp_exc'] = fp_exc
    dout['fn_exc'] = fn_exc
    # inh
    dout['tp_inh'] = tp_inh
    dout['tn_inh'] = tn_inh
    dout['fp_inh'] = fp_inh
    dout['fn_inh'] = fn_inh

    fname_conf_mat = os.path.join(resultsfolder, params['subfolder'],
                                  folder_reg,
                                  params['fname_conf_mat'])

    np.save(fname_conf_mat, dout)
    return dout


def ratio(a, b):
    '''
    '''
    s = a + b
    return a/s if s else 1.


def plot_roc(params):
    '''
    '''
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
        fname_lasso = os.path.join(resultsfolder, params['subfolder'],
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
    plt.savefig('plot_ROC.png')


def run_all(snum='0013', foldercfg='0002'): #, reg_vect=[0.001, 0.01, 0.1, 0.2, 0.3, 0.4]):
    ''' foldercfg potrebbe essere letto dal file di output ..
    '''
    #subfolder = 'test/%s' % snum
    #subfolder = '%s' % snum
    subfolder = ''

    # lasso
    params_lasso['subfolder'] = subfolder
    params_lasso['fname_RSmat'] = 'RSmat_%s.npy' % snum
    params_lasso['fname_lasso'] = 'RSmat_%s_lasso.npy' % snum

    # confusion mat
    params_conf_mat['subfolder'] = subfolder
    params_conf_mat['fname_lasso'] = 'RSmat_%s_lasso.npy' % snum
    params_conf_mat['foldercfg'] = foldercfg
    params_conf_mat['fname_conf_mat'] = 'RSmat_%s_conf_mat.npy' % snum

    # roc
    params_roc['subfolder'] = subfolder
    params_roc['fname_conf_mat'] = 'RSmat_%s_conf_mat.npy' % snum

    # step 0 R,S mat
    calc_mat_processes('%s/output_%s.npy' % (subfolder, snum),
                       '%s/params_netw.npy' % foldercfg,
                       '%s/RSmat_%s.npy' % (subfolder, snum))

    keys = ['regularization_strength', 'tp', 'tn', 'fp', 'fn', 'tp_exc',
            'tn_exc', 'fp_exc', 'fn_exc', 'tp_inh', 'tn_inh', 'fp_inh',
            'fn_inh']
    dict_out = {}
    for key in keys:
        dict_out[key] = []
    for reg in params_roc['reg_vect']:
        # lasso regularization (step 1)
        print(reg)
        params_lasso['regularization_strength'] = reg
        lasso(params_lasso)
        # confusion matrix (step 2)
        params_conf_mat['regularization_strength'] = reg
        dout = confusion_matrix(params_conf_mat)
        for key in keys:
            dict_out[key].append(dout[key])
    df = pd.DataFrame(dict_out)
    df['tp+tn+fp+fn'] = df['tp'] + df['tn'] + df['fp'] + df['fn']
    df['(tp+tn+fp+fn)_exc'] = df['tp_exc'] + df['tn_exc'] + df['fp_exc'] + \
                              df['fn_exc']
    df['(tp+tn+fp+fn)_inh'] = df['tp_inh'] + df['tn_inh'] + df['fp_inh'] + \
                              df['fn_inh']

    conn = load_dict(os.path.join(configfolder, '%s/params_netw.npy' % foldercfg))
    src = np.array(conn['conn_mat'])[:, 0]
    nexc = 0
    for nrn in conn['exc']:
        nexc += np.count_nonzero(src == nrn)
    ninh = 0
    for nrn in conn['inh']:
        ninh += np.count_nonzero(src == nrn)

    df['total_exc'] = nexc
    df['total_inh'] = ninh
    df.to_csv(os.path.join(resultsfolder, subfolder, 'confusion_mat.csv'))

    plot_roc(params_roc)
