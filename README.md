

The code was realized using python 3.10.1 and it depends on the standard
modules: nest (2.20.2), numpy (1.21.5), sklearn (1.0.2), scipy (1.7.3),
json (2.0.9), pandas (1.3.5).

Before running the code you need to adjust the variables:

> fpath_results (default '/home/tnieus/Projects/RESULTS/Lasso/') in network_sim.py
> fpath_cfg (default '/home/tnieus/Projects/CODE/Lasso/config') in network_util.py

A typical run consists in:

1. simulating network dynamics (with network_sim.py)
2. inferring the network connectivity (with lasso.py)

1) A configuration file is needed to run a simulation (e.g. Lasso/config).
The configuration files used in this work have been provided.
To use configuration number 5 of the 20_neurons simulation part the variables
are set as:
    > config_num = 5
    > fpath_cfg = '/home/USER/Projects/CODE/Lasso/config/20_neurons/'
    > fpath_results = (default '/home/tnieus/Projects/RESULTS/Lasso/')

The simulation generates an output file (output.npy) in fpath_results.

2) At first we need to set a sequence of regularization coefficients (inverse of lambda) with params_roc['reg_vect']=[....] (a default is provided).
Then, with the command:

    > run_all(outputfolder)

text files (csv) and plots (png) are generated about ROC, Youden and MCC.
