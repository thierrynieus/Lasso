fn_out='/home/tnieus/Projects/RESULTS/Lasso/100nrn_80exc_20inh_spatial/0001/output.npy'
dout = np.load(fn_out, allow_pickle=1).item()
conn_mat=np.array(dout['params_netw']['conn_mat'])

fn_beta='/home/tnieus/Projects/RESULTS/Lasso/100nrn_80exc_20inh_spatial/0001/reg_0.8/RSmat_lasso.npy'
dbeta = np.load(fn_beta, allow_pickle=1).item()
beta=dbeta['beta']

for nrn in np.arange(0,10):
  print("neuron %d" % nrn)
  idx = np.where(conn_mat[:, 0] == nrn)[0]
  x = conn_mat[idx, 1]
  y = np.where(beta[nrn, :])[0]

  #print("structural", x)
  #print("functional", y)
  if len(np.setdiff1d(x, y)) | len(np.setdiff1d(y, x)):
    print(x)
    print(y)
    print(beta[nrn, y])
