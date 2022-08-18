import copy


def updateDicts(dict1, dict2):
    temp = copy.deepcopy(dict1)
    temp.update(dict2)
    return temp


def CommonNeuron():
    param = {
            'E_rev_AMPA': 0.0,
            'g_peak_AMPA': 0.1,
            'tau_decay_AMPA': 2.4,
            'tau_rise_AMPA': 0.5,
            'g_peak_GABA_A': 0.33,
            'tau_decay_GABA_A': 7.0,
            'tau_rise_GABA_A': 1.0,
            'E_rev_GABA_B': -90.0,
            'g_peak_GABA_B': 0.0132,
            'tau_decay_GABA_B': 200.0,
            'tau_rise_GABA_B': 60.0,
            'E_rev_NMDA': 0.0,
            'g_peak_NMDA': 0.075,
            'tau_decay_NMDA': 40.0,
            'tau_rise_NMDA': 4.0,
            'S_act_NMDA': 0.081,
            'V_act_NMDA': -25.57,
            'tau_Mg_fast_NMDA': 0.68,
            'tau_Mg_slow_NMDA': 22.7,
            'instant_unblock_NMDA': True,
            'g_NaL': 0.2,
            'g_KL': 1.0,
            'E_Na': 30.0,
            'E_K': -90.0,
            'Ca': 0.0,
            'tau_D_KNa': 1400.0,
            'E_rev_KNa': -90.0,
            'E_rev_NaP': 30.0,
            'E_rev_T': 0.0,
            'E_rev_h': -40.0,
            'V_m': -70.0,
            'beta_Ca': 0.001,
            'supports_precise_spikes': False,
            'tau_Ca': 10000.0,
            'tau_minus': 20.0,
            'tau_minus_triplet': 110.0,
            'theta': -51.0,
            'voltage_clamp': False,
            'frozen': False,
            'global_id': 0,
            'local': True,
            'node_uses_wfr': False,
            'archiver_length': 0,
            'receptor_types': {'AMPA': 1, 'GABA_A': 3, 'GABA_B': 4, 'NMDA': 2},
            'synaptic_elements': {},

            'g_peak_NaP': 0.,
            'g_peak_KNa': 0.,
            'g_peak_T': 0.,
            'g_peak_h': 0.,
            'tau_m': 8.0,
            'theta_eq': -53.0,
            'tau_theta': 1.0,
            't_ref': 2.0,
            'tau_spike': 0.5,
            'E_rev_GABA_A': -70.0,
            }
    return param


def InNeuronC():
    param = {
        'tau_m':   8.0,
        'theta_eq': -53.0,
        'tau_theta': 1.0,
        't_ref': 1.0,
        'tau_spike': 0.5,
        'E_rev_GABA_A': -70.0,
        'g_KL': 1.0,
        'g_peak_NaP': 0.5,
        'g_peak_h': 0.00,
        'g_peak_KNa': 0.5,
    }
    return updateDicts(CommonNeuron(), param)


def ExNeuronC():
    param = {
        'tau_m':   16.0,
        'theta_eq': -51.0,
        'tau_theta': 2.0,
        't_ref': 2.0,
        'tau_spike': 1.75,
        'E_rev_GABA_A': -70.0,
        'g_KL': 1.0,
        'g_peak_NaP': 0.5,
        'g_peak_h': 0.00,
        'g_peak_KNa': 0.5,
    }
    return updateDicts(CommonNeuron(), param)
