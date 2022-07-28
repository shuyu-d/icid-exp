import numpy as np

import matplotlib.pyplot as plt
from timeit import default_timer as timer
import time, os
import pandas as pd
import shutil

from icid import utils
from icid.icid import run_icid, AMA_independece_decomp
from external.test_linear import notears_linear

from sklearn.covariance import GraphLassoCV, GraphLasso
from sklearn.metrics import  confusion_matrix, classification_report

def test_multisettings(fdir,  ntests=8,  fin=None, fname=None,\
                        graphtype='ER', semtype='gauss'):
    utils.set_random_seed(1)
    inp0 = {'n' : 600, \
            'd' : 0, \
            'k' : 25, \
            'deg' : 0.3, \
            'graph_type':graphtype, 'sem_type' : semtype,\
            'lambda_1': 2e-1}
    if fin is not None:
        list_    = pd.read_csv(fin)
        # columns of list_ are d, n, deg, lambda_1
        inps = []
        for i in range(len(list_)):
            inp_ = inp0.copy()
            inp_['d'] = list_['d'][i]
            inp_['n'] = list_['n'][i]
            inp_['deg'] = list_['deg'][i]
            # inp_['lambda_1'] = list_['lambda1'][i]
            inps.append(inp_.copy())
    else:
        # Only d varies
        d0 = 100
        inps = []
        for i in range(ntests):
            inp0['d'] += d0
            inp0['n'] = 4*inp0['d']
            inps.append(inp0.copy())
    df_inps = pd.DataFrame(inps).to_csv('%s/inputs_all.csv' %fdir)
    shutil.copy2('./test_main.py', '%s/test_main.py.txt' % fdir)
    res = []
    for i in range(len(inps)):
        print('-------Test %d/%d------' %(i+1,len(inps)))
        n,d,k,deg,graph_type,sem_type,lambda_1 = inps[i]['n'], \
                                        inps[i]['d'],\
                                        inps[i]['k'],\
                                        inps[i]['deg'],\
                                        inps[i]['graph_type'],\
                                        inps[i]['sem_type'],\
                                        inps[i]['lambda_1']
        spr = deg*d / d**2
        s0 = int(np.ceil(spr*d**2))
        print(d)
        print(s0)
        print(graph_type)
        print(lambda_1)
        B_true = utils.simulate_dag(d, s0, graph_type)
        B_true = B_true.T # In the case of scale-free (SF) graphs, hubs are mostly causes, rather than effects, of its neighbors
        W_true = utils.simulate_parameter(B_true)
        X = utils.simulate_linear_sem(W_true, n, sem_type)

        """ Alg 3-----ICD-Loram-Altmin """
        print('-------Ready to run ICD-LoRAM-Altmin---')
        W_icd, ith_icd  = run_icid(X, lambda_1=lambda_1,
                                    sigma_0=1.0, k=k,
                                    beta_2 = 0.7, gamma_2=1.0, \
                                    maxit_prox_inner=500, W_true=W_true)

        """ Evaluations and record results """
        # Cmax = max(abs(W_icd.ravel()))
        # W_icd[abs(W_icd) <= 9e-2 ] = 0
        acc = utils.count_accuracy(B_true, W_icd != 0)
        print(acc)
        res.append({'d':d, 'alg':'ICD (ours)',
                'deg': deg, 'n':n, 'graph_type': graph_type,\
            'shd':acc['shd'],
            'tpr':acc['tpr'], 'fdr':acc['fdr'], 'fpr':acc['fpr'],
            'nnz':acc['nnz'], 'time': ith_icd.iloc[-1]['time']})
        """ SAVE RESULTS """
        pd.DataFrame(ith_icd).to_csv('%s/tid%d_ith_icid.csv'\
                                      %(fdir, i))
    df_res = pd.DataFrame(res).to_csv('%s/res_all.csv' %fdir)

if __name__ == '__main__':
    if True:
        # test 5b (ER)
        timestr = time.strftime("%H%M%S%m%d")
        # fin = 'aux/list_d_n_deg.csv'
        fin = 'aux/list_dndeg_2.csv'
        fout = '../outputs/linsem_oursvnotears_d_%s' % timestr
        if not os.path.exists(fout):
            os.makedirs(fout)
        test_multisettings(fout, fin=fin, \
                graphtype='ER')

    if False:
        # test 5b (SF)
        timestr = time.strftime("%H%M%S%m%d")
        fin = 'aux/list_dndeg_3.csv'
        fout = '../outputs/linsem_oursvnotears_d_%s' % timestr
        if not os.path.exists(fout):
            os.makedirs(fout)
        test_multisettings(fout, fin=fin, \
                graphtype='SF')


