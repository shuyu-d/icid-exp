import numpy as np

from timeit import default_timer as timer
import time, os, sys
import pandas as pd
import shutil

from aux.gen_settings import gen_list_settings, gen_list_optparams, gen_data_sem
from icid import utils
from icid.icid import run_icid, AMA_independece_decomp
from external.test_linear import notears_linear
import ges


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
    """ Evaluations and record results """
    acc = utils.count_accuracy(W_true!=0, W_icd!=0)
    """ SAVE RESULTS """
    pd.DataFrame(ith_icd).to_csv('%s/%s_ith_icid.csv'\
                                    %(fdir, tid))
    return acc, ith_icd

if __name__ == '__main__':
    timestr = time.strftime("%H%M%S%m%d")
    FDIR = 'outputs/exp2_%s' % timestr
    if not os.path.exists(FDIR):
        os.makedirs(FDIR)
    # Generate input parameters
    ds            = np.array([25,50,75,100])
    degs          = np.linspace(0.2,2.0,10)
    graph_types   = ['ER']      # ['ER', 'SF']
    sem_types     = ['gauss']   # ['gauss', 'gumbel', 'exp']
    R_N2D         = 10          # ratio of n/d, used below to define n
    #
    pbs_l, pbs = gen_list_settings(d=ds, degs=degs, \
                                 graph_types=graph_types, \
                                 sem_types=sem_types)

    # Methods to run
    # NOTE: Modify here to control which methods to run
    ms = {'icid': False, \
            'ges':  False}
    ni = len(sys.argv)
    for i in range(ni):
        ms[sys.argv[i]] = True
    # Iterate through all problem settings ('pbs')
    res = []
    for i in range(len(pbs)):
        n = R_N2D * pbs.iloc[i]['d']  # actual number of samples
        pbs.at[i,'n'] = n
        print(pbs.iloc[i])
        print('Above is input info to be used\n')
        Wtrue, X = gen_data_sem(d           = pbs['d'][i], \
                                deg         = pbs['deg'][i], \
                                n           = pbs['n'][i], \
                                graph_type  = pbs['graph_type'][i],\
                                sem_type    = pbs['sem_type'][i], \
                                seed = 1)
        #----------- ICID ------------
        if ms['icid']:
            # Parameters of ICID algorithm
            opts={'k':              [25], \
                  'lambda_1':       [1e-1, 4e-1], \
                  'idec_lambda1':   [1e-1]}
            l_o, df_o = gen_list_optparams(opts)
            print('List of opt parameters to tets are:')
            print(df_o)
            # Iterate through all optimization parameter configs
            for j in range(len(df_o)):
                acc, ith = test_save_res_icid(Wtrue, X, \
                                    k    =df_o['k'][j], \
                                    optp1=df_o['lambda_1'][j] , \
                                    optp2=df_o['idec_lambda1'][j] , \
                                    tid='pb%dopt%d' %(i+1,j+1), \
                                    fdir=FDIR)
                print(acc)
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
                        'nnz':acc['nnz'], 'time': ith.iloc[-1]['time']}
                    )
                pd.DataFrame(res, columns=res[0].keys()).to_csv('%s/res_all.csv' %FDIR)
        #-------- GES ----------------
        if ms['ges']:
            print('-------Ready to run GES ----------')
            t0 = timer()
            w_ges, _ = ges.fit_bic(X)
            t_ges = timer() - t0
            acc_ges = utils.count_accuracy(Wtrue!=0, w_ges!=0)
            print(acc_ges)
            res.append({'alg':'GES',
                        'd'           : pbs['d'][i], \
                        'deg'         : pbs['deg'][i], \
                        'n'           : pbs['n'][i], \
                        'graph_type'  : pbs['graph_type'][i],\
                        'sem_type'    : pbs['sem_type'][i], \
                        'shd':acc_ges['shd'], 'tpr':acc_ges['tpr'], \
                        'fdr':acc_ges['fdr'], 'fpr':acc_ges['fpr'], \
                        'nnz':acc_ges['nnz'], 'time': t_ges}
                    )
            pd.DataFrame(res, columns=res[0].keys()).to_csv('%s/res_all.csv' %FDIR)
    if len(res) > 0:
        endstr = time.strftime("%H%M%S%m%d")
        pd.DataFrame(res, columns=res[0].keys()).to_csv('%s/resall_%s.csv' %(FDIR,endstr))


