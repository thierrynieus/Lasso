import numpy as np
import os
import nest
import neuron_defs as ndef
import network_util as nu

nest.ResetKernel()
nest.ResetNetwork()
seed = np.random.randint(10000)
nest.SetStatus([0], [{'rng_seeds': [seed]}])

receptors = {'AMPA': 1, 'GABA_A': 3, 'GABA_B': 4, 'NMDA': 2}

if not('config_num' in globals()):
    config_num = 0
if not('fpath_cfg' in globals()):
    fpath_cfg = nu.fpath_cfg
else:
    nu.fpath_cfg = fpath_cfg
if not('fpath_results' in globals()):
    fpath_results = '/home/tnieus/Projects/RESULTS/Lasso/'


def shape_output(rec):
    '''
    '''
    n = len(rec['noise'])
    dout = {}
    for k in range(n):
        dout[k] = {}
        dout[k]['spikes_nrn'] = nest.GetStatus(rec['spikes'])[k]['events']['times']
        dout[k]['spikes_noise'] = nest.GetStatus(rec['noise'])[k]['events']['times']
    # record also voltage if required
    if 'vm' in list(rec):
        for k in range(n):
            dout[k]['vm'] = nest.GetStatus(rec['vm'])[k]['events']
    return dout


params_syn, params_noise, params_neurons, params_netw = nu.load_config(config_num)

nest.CopyModel('ht_synapse', 'AMPA_syn_ee', params_syn['ee'])
nest.CopyModel('ht_synapse', 'AMPA_syn_ei', params_syn['ei'])
nest.CopyModel('ht_synapse', 'GABA_syn_ii', params_syn['ii'])
nest.CopyModel('ht_synapse', 'GABA_syn_ie', params_syn['ie'])

# params_netw = create_conn_mat(params_neurons)  # some global var ...

# create neurons
neurons = nest.Create('ht_neuron', params_netw['num_neurons'])
dc_gen = nest.Create('dc_generator', params_netw['num_neurons'],
                     params=params_neurons['cc'])
parrots = nest.Create('parrot_neuron', params_netw['num_neurons'])
noise = nest.Create('sinusoidal_poisson_generator', params_netw['num_neurons'],
                    params=params_noise['stim'])

rec = {}
rec['spikes'] = nest.Create('spike_detector', params_netw['num_neurons'])
rec['noise'] = nest.Create('spike_detector', params_netw['num_neurons'])

if params_neurons['rec_Vm']:
    nest.CopyModel('multimeter', 'RecordingNode',
                   params={'interval': .1, 'record_from': ['V_m'],
                           'record_to': ['memory'], 'withgid': True,
                           'withtime': True})
    rec['vm'] = nest.Create('RecordingNode', params_netw['num_neurons'])
    nest.Connect(rec['vm'], neurons, 'one_to_one')

nest.Connect(neurons, rec['spikes'], 'one_to_one')
nest.Connect(dc_gen, neurons, 'one_to_one')
nest.Connect(noise, parrots, 'one_to_one')
nest.Connect(parrots, neurons, 'one_to_one', params_noise['syn'])
nest.Connect(parrots, rec['noise'], 'one_to_one')

for i_exc in params_netw['exc']:
    nest.SetStatus([neurons[i_exc]], ndef.ExNeuronC())
for i_inh in params_netw['inh']:
    nest.SetStatus([neurons[i_inh]], ndef.ExNeuronC())

# connections
for conn in params_netw['conn_mat']:
    src, dst = conn
    # ee
    if (src in params_netw['exc']) & (dst in params_netw['exc']):
        nest.Connect([neurons[src]], [neurons[dst]], syn_spec='AMPA_syn_ee')
    # ei
    if (src in params_netw['exc']) & (dst in params_netw['inh']):
        nest.Connect([neurons[src]], [neurons[dst]], syn_spec='AMPA_syn_ei')
    # ii
    if (src in params_netw['inh']) & (dst in params_netw['inh']):
        nest.Connect([neurons[src]], [neurons[dst]], syn_spec='GABA_syn_ii')
    # ie
    if (src in params_netw['inh']) & (dst in params_netw['exc']):
        nest.Connect([neurons[src]], [neurons[dst]], syn_spec='GABA_syn_ie')


''' run simulation
'''

nest.Simulate(params_neurons['sim_duration'])

dout = shape_output(rec)
dout['cfg_num'] = config_num
dout['fpath_cfg'] = fpath_cfg
dout['params_syn'] = params_syn
dout['params_noise'] = params_noise
dout['params_neurons'] = params_neurons
dout['params_netw'] = params_netw

#  nu.save_npy(dout, fpath_results, 'output_%s.npy')
nu.save_npy(dout, fpath_results, 'output.npy')
