import numpy as np

import matplotlib.pyplot as plt
from timeit import default_timer as timer
import time, os
import pandas as pd
import shutil

from aux.gen_settings import gen_list_settings, gen_list_optparams, gen_data_sem
from icid import utils
from icid.icid import run_icid, AMA_independece_decomp


def test_save_res_icid(W_true, X, k=25, \
                       optp1=2e-1, optp2=1e-1, \
                       tid=None, fdir='outputs/exp0_'):
    if not os.path.exists(fdir):
        os.makedirs(fdir)
    """ ICID """
    print('-------Ready to run ICID ----------')
    W_icd, ith_icd  = run_icid(X, sigma_0=1.0, k=k,\
                              lambda_1=optp1, \
                              idec_lambda1=optp2, \
                              beta_2 = 0.7, gamma_2=1.0, \
                              maxit_prox_inner=500, W_true=W_true)
    # """ Evaluations and record results """
    # acc = utils.count_accuracy(W_true!=0, W_icd!=0)
    """ SAVE RESULTS """
    pd.DataFrame(ith_icd).to_csv('%s/%s_ith_icid.csv'\
                                    %(fdir, tid))
    return W_icd, ith_icd

if __name__ == '__main__':
    timestr = time.strftime("%H%M%S%m%d")
    FDIR = 'outputs/exp1_%s' % timestr
    if not os.path.exists(FDIR):
        os.makedirs(FDIR)
    # Generate input parameters
    ds            = np.array([100, 200, 400, 600, 800, 1000, 2000, 3000])
    degs          = [0.5, 1.0] # 0.5
    graph_types   = ['ER']
    sem_types     = ['gauss']
    # n = 5*ds  # to be updated later (depending on d) on the fly
    #
    pbs_l, pbs = gen_list_settings(d=ds, degs=degs, \
                                 graph_types=graph_types, \
                                 sem_types=sem_types)

    # Parameters of ICID algorithm
    opts={'k':              [25], \
          'lambda_1':       [4e-1], \
          'idec_lambda1':   [1e-1]}
    l_o, df_o = gen_list_optparams(opts)
    print('List of opt parameters to tets are:')
    print(df_o)

    # Iterate through all problem settings ('pbs')
    res = []
    for i in range(len(pbs)):
        pbs.at[i,'n'] = 5 * pbs.iloc[i]['d']  # number of samples be 5xd
        print(pbs.iloc[i])
        print('Above is input info to be used.')
        Wtrue, X = gen_data_sem(d           = pbs['d'][i], \
                                deg         = pbs['deg'][i], \
                                n           = pbs['n'][i], \
                                graph_type  = pbs['graph_type'][i],\
                                sem_type    = pbs['sem_type'][i], \
                                seed = 1)
        print('Graph and SCM data generated.\n')
        # Iterate through all optimization parameter configs
        for j in range(len(df_o)):
            W_icd, ith = test_save_res_icid(Wtrue, X, \
                                k    =df_o['k'][j], \
                                optp1=df_o['lambda_1'][j] , \
                                optp2=df_o['idec_lambda1'][j] , \
                                tid='pb%dopt%d' %(i+1,j+1), \
                                fdir=FDIR)
            acc = utils.count_accuracy(Wtrue!=0, W_icd!=0)
            runtime = ith.iloc[-1]['time']
            # acc = {'shd':np.nan, 'tpr':np.nan, 'fdr':np.nan,\
            #         'fpr':np.nan, 'nnz':np.nan}
            # runtime = np.nan
            # Record results
            res.append({'alg':'ICID',
                        'd'           : pbs['d'][i], \
                        'deg'         : pbs['deg'][i], \
                        'n'           : pbs['n'][i], \
                        'graph_type'  : pbs['graph_type'][i],\
                        'sem_type'    : pbs['sem_type'][i], \
                        'k'           : df_o['k'][j], \
                        'lambda_1'    : df_o['lambda_1'][j], \
                    'idec_lambda1'    : df_o['idec_lambda1'][j], \
                    'shd':acc['shd'], 'tpr':acc['tpr'], \
                    'fdr':acc['fdr'], 'fpr':acc['fpr'], \
                    'nnz':acc['nnz'], 'time': runtime}
                )
    # Save results to file
    pd.DataFrame(res, columns=res[0].keys()).to_csv('%s/res_all.csv' %FDIR)
    endstr = time.strftime("%H%M%S%m%d")
    pd.DataFrame(res, columns=res[0].keys()).to_csv('%s/resall_%s.csv' %(FDIR,endstr))



