# -*- coding: utf-8 -*-

import copy
#Nice little auxilary function
def updateDicts(dict1,dict2):
    temp=copy.deepcopy(dict1)
    temp.update(dict2)
    return temp

#**********!!** denotes a parameter that is conflicting with either original code or the original HT paper.

#Common Neuronal parameters/all parameters/change globals here (some might be overwritten later)
def CommonNeuron():
    param = {
            #General parameters (mostly standard accross neuron types), missing GABA_a_tc
            'E_rev_AMPA':0.0, #Reversal potential
            'g_peak_AMPA':0.1,####0.05#####, #Peak conductance
            'tau_decay_AMPA': 2.4, #Time constant for exponential decay
            'tau_rise_AMPA': 0.5, #Time constant for exponential rise
            'g_peak_GABA_A':0.33,#####0.66########,
            'tau_decay_GABA_A': 7.0,
            'tau_rise_GABA_A': 1.0,
            'E_rev_GABA_B':-90.0,
            'g_peak_GABA_B':0.0132,####0.0264#######, #This parameter is a promising tuning candidate
            'tau_decay_GABA_B': 200.0,
            'tau_rise_GABA_B': 60.0,
            'E_rev_NMDA':0.0,
            'g_peak_NMDA':0.075,#####0.075########, #This parameter is a promising tuning candidate
            'tau_decay_NMDA': 40.0,
            'tau_rise_NMDA': 4.0,
            'S_act_NMDA': 0.081,
            'V_act_NMDA': -25.57,
            'tau_Mg_fast_NMDA': 0.68,  #Time constant for blocking magnesium Ion (fast)
            'tau_Mg_slow_NMDA': 22.7, #Time constant for blocking magnesium Ion (slow)
            'instant_unblock_NMDA': True, #####False########
            'g_NaL': 0.2, #Sodium leak (conductance constant)
            'g_KL': 1.0, #Potassium leak (conductance constant) ** changed
            'E_Na': 30.0, #Sodium reversal potential
            'E_K': -90.0, #Potassium reversal potential
            'Ca': 0.0, #Calcium, I'll have none of it
            "tau_D_KNa":1400.0, #Sodium dependent potassium time constant
            'E_rev_KNa': -90.0, #Sodium dependent potassium reversal potential
            'E_rev_NaP': 30.0, #Persistent sodium reversal potential
            'E_rev_T': 0.0, #T-type calcium channel, reversal potential
            'E_rev_h': -40.0, #H-current reversal potential
            'V_m': -70.0, #Membrane potential, resting
            'beta_Ca': 0.001, #Equilibrium sodium&calcium concentration (KNa?), depolarization
            'supports_precise_spikes': False, #Interpolation of spikes (might lead to artificial synchronization)
            'tau_Ca': 10000.0, #Experiment with*** #Structural plasticity parameter
            'tau_minus': 20.0, #?
            'tau_minus_triplet': 110.0, #?
            'theta': -51.0, #Cortical exitatory threshold
            'voltage_clamp': False, #Hold voltage constant
            
            #System parameters (unwritable in this context)
            #'vp': -1, #System stuff            
            #'thread': 0, #System stuff
            #'thread_local_id': -1, #System stuff  
            #'capacity': (1000,), #System stuff
            #'elementsize': 968, #System stuff
            'frozen': False, #System stuff
            'global_id': 0, #System stuff
            #'instantiations': (100,), #System stuff
            'local': True, #System stuff
            'node_uses_wfr': False, #System stuff
            'archiver_length': 0, #System stuff
            #'available': (900,), #System stuff
            'receptor_types': {'AMPA': 1, 'GABA_A': 3, 'GABA_B': 4, 'NMDA': 2}, #Synapse ID's
            'synaptic_elements': {}, #Continaer for elements
            
            #These will be updated by neuron class specific (and some general parameters)
            'g_peak_NaP': 0., #Peak conductance Sodium persistant current
            'g_peak_KNa': 0., #Peak conductance Sodium dependent potassium current
            'g_peak_T': 0., #T-type calsium channel conductance
            'g_peak_h': 0., #H-current conductance
            'tau_m':8.0,
            'theta_eq':-53.0,
            'tau_theta':1.0,
            't_ref': 2.0, #Time of refractory period
            'tau_spike':0.5,
            'E_rev_GABA_A':-70.0,
            }
    return param
    
#Inhibitory Cortical Neuron
def InNeuronC():
     param = {
        #"INaP_tresh": -55.0,
        'tau_m'  :   8.0,
        'theta_eq' : -53.0,
        'tau_theta': 1.0,
        't_ref':1.0,
        'tau_spike': 0.5,
        'E_rev_GABA_A':-70.0,
        "g_KL":1.0,#2.0,
        "g_peak_NaP":0.5,#1.0, #*** added for all neurons 0.5
        "g_peak_h":0.00,
        "g_peak_KNa":0.5,
     }
     return updateDicts(CommonNeuron(),param)
    
#Exicatory Cortical Neuron
def ExNeuronC():
    param= {
        #"INaP_tresh": -55.0,
        'tau_m'  :   16.0,
        'theta_eq' : -51.0,
        'tau_theta': 2.0,
        't_ref': 2.0,
        'tau_spike': 1.75, #Dont change this one without also changing the value in the 50% AMPA syn function in the main script
        'E_rev_GABA_A':-70.0,
        "g_KL":1.0, #2.0,
        "g_peak_NaP":0.5, #1.0, #0.5
        "g_peak_h":0.00, #1.0
        "g_peak_KNa":0.5,
        }
    return updateDicts(CommonNeuron(),param)

#Exicatory Thalamic Neuron
def ExThaNeuron():
      param = {
        #"INaP_tresh": -45.0,
        'tau_m'  :   8.0,
        'theta_eq' : -53.0,
        'tau_theta': 1.0,#0.75,
        't_ref': 1.0,
        'tau_spike': 0.75,
        'E_rev_GABA_A':-80.0,
        "g_KL":1.0,#2.0,
        "g_peak_NaP":0.5,#1.0, #0.5
        "g_peak_h":1.0,
        "g_peak_T":1.0,
        }
      return updateDicts(CommonNeuron(),param)

#Inhibitory Thalamic Neuron
def InThaNeuron():
      param = {
        "INaP_tresh": -45.0,
        'tau_m'  :   8.0,
        'theta_eq' : -53.0,
        'tau_theta': 1.0,#0.75,
        't_ref': 1.0,
        'tau_spike': 0.75,
        'E_rev_GABA_A':-80.0,#-70.0,
        "g_KL":1.0,#2.0,
        "g_peak_NaP":0.0,#1.0, #0.5
        "g_peak_h":0.0,#1.0, 
        "g_peak_T":0.0,#1.0
        }
      return updateDicts(CommonNeuron(),param)

#Reticular neuron
def RetNeuron():
     param = {
        "INaP_tresh": -45.0,
        'tau_m'  :   8.0,
        'theta_eq' : -53.0,
        'tau_theta': 1.0,#0.75,
        't_ref': 1.0,
        'tau_spike': 0.75,
        'E_rev_GABA_A':-70.0,
        "g_KL":1.0,#2.0,
        "g_peak_NaP":0.5,#1.0, #0.5
        "g_peak_h":0.00,
        "g_peak_T":1.0,
        }
     return updateDicts(CommonNeuron(),param)
