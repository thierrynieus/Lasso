# for file search 
import numpy as np
import os 
import fnmatch

import pandas as pd

sTIME='08:00:00' 

home_laptop = '/home/tnieus/Projects/CODE/Lasso/'         # path to the scripts of laptop 
home_cluster = '/gpfs/home/users/thierry.nieus/lasso/'    # path to the home of cluster 

fpath_rel_config = '100nrn_80exc_20inh_spatial_2clusters/0000'
fpath_rel_results = '100nrn_80exc_20inh_spatial_2clusters/0000'

fpathPYSLURM_local = '100nrn_80exc_20inh_spatial_2clusters/0000'

account='cerebellum'
ram_memory=8000 #30000

def write_out(reg, count, fpathPYSLURM):
    '''
        fpathPYSLURM    subfolder where to write/find PY & SLURM files 
        count           file number to process 
    '''
    # python file to run 
    fnPY = os.path.join(home_laptop, fpathPYSLURM, 'run_%d.py'%count)     
    # slurm file to run 
    fnSLURM = fnPY.replace('.py','.slurm')                


    # PY file to execute 
    f = open(fnPY, 'w')
    f.write('import sys,os\n')
    f.write('sys.path.append(os.getcwd())\n')
    f.write('exec(open(\'lasso.py\').read())\n')      
    f.write('resultsfolder = \'/gpfs/home/users/thierry.nieus/lasso/results/\' \n')
    f.write('configfolder = \'/gpfs/home/users/thierry.nieus/lasso/config/\' \n')          
    f.write('params_roc[\'reg_vect\'] = [%g]\n' % reg)       
    f.write('run_all(\'%s\', \'%s\') \n' % (fpath_rel_config, fpath_rel_results))    
    f.close()
 
    # SLURM file to execute
    
    fname_out = os.path.join(home_cluster,fpathPYSLURM,'out')

    f=open(fnSLURM, 'w')
    f.write('#!/bin/bash \n')
    f.write('#SBATCH -o %s/%s.%s.out  \n' % (fname_out, '%j', '%N'))     
    f.write('#SBATCH -D %s/%s  \n' % (home_cluster, fpathPYSLURM))
    f.write('#SBATCH -J PCIev_V2  \n')
    f.write('#SBATCH --get-user-env  \n')
    f.write('#SBATCH -p light\n')
    f.write('#SBATCH --nodes=1\n')
    f.write('#SBATCH -c 1\n')  
    f.write('#SBATCH --mem-per-cpu %d\n'%ram_memory)
    f.write('#SBATCH --account=%s\n'%account)
    f.write('#SBATCH --time=%s  \n'%sTIME)
    f.write('module load python3/intel/2019  \n') 
    f.write('cd %s \n'%home_cluster)
    f.write('python3 %s \n'%(fnPY.replace(home_laptop, home_cluster)))
    f.write('seff $SLURM_JOBID   \n')   
    f.close()

    return fnPY, fnSLURM

def batch(reg_seq=[],fnout='execall.sh'):
    '''
    '''
    fpathPYSLURM = 'slurm/%s' % fpathPYSLURM_local
    if not(os.path.exists(fpathPYSLURM_local)): 
        os.makedirs(fpathPYSLURM_local)

    # put execall.sh in the same folder of the PY+SLURM files 
    g = open(os.path.join(home_laptop, fpathPYSLURM, fnout), 'w')      
    fpath_out = os.path.join(home_cluster, fpathPYSLURM, 'out')
    g.write('mkdir out\n')
    
    count=0
    for count, reg in enumerate(reg_seq):  
        fnSLURM = write_out(reg, count, fpathPYSLURM)[1]
        idx = fnSLURM.rfind('/') + 1
        g.write('sbatch %s\n' % fnSLURM[idx:])
    g.close()

