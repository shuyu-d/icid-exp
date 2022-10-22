import numpy as np

from timeit import default_timer as timer
import time, os, sys
import pandas as pd
import shutil

from aux.gen_settings import gen_list_settings, gen_list_optparams, gen_data_sem
from aux.dag_utils import get_data_sachs
from icid import utils
from icid.icid import run_icid, AMA_independece_decomp
from external.test_linear import notears_linear
import ges


## Results of dev-ic for sachs data
# 1,Sparse Empirical 2,129.20421332570325,0.4236880958709285,-8.091694286558296,32,43,35,0.0003571510314941406,0.0035789473684210526
# 2,Sparse Empirical 3,129.1433468759635,2.1572468264227607,-7.970138986668255,26,33,35,0.00034809112548828125,0.006157894736842105



if __name__ == '__main__':
    timestr = time.strftime("%H%M%S%m%d")
    FDIR = 'outputs/exp-sachs_%s' % timestr

    # Methods to run
    ms = {'icid':   False, \
          'ges':    False,\
          'notears':False,\
          'ideal':  False}
    ni = len(sys.argv)
    for i in range(ni):
        ms[sys.argv[i]] = True
    if ni > 1:
        FDIR = 'outputs/exp2f_%s_%s' % (timestr, sys.argv[1])
    if not os.path.exists(FDIR):
        os.makedirs(FDIR)
    Nt = 10
    res = []
    for i in range(Nt):
        # Get data
        X, Wtrue = get_data_sachs(normalize=True, ndata=-1)
        print(X.shape)
        print(X)
        print('Above is the size of the Sachs data\n')
        n, d = X.shape
        #----------- ICID ------------
        if ms['icid'] or ms['ideal']:
            if ms['ideal']:
                IC = 'ideal'
            else:
                IC = 'emp'
                # IC = 'emp_gs'
            if IC == 'emp_gs':
                lambda1s = np.linspace(1e-3,5e-2,20)
                # lambda1s = [0.0035789473684210526, 0.006157894736842105]
            # Parameters of ICID algorithm
            opts={'opt_ic':         [IC], \
                 'idec_solver':     ['FISTA'], \
                  'k':              [9], \
                  'lambda_1':       [1.2e-3,1.3e-3, 0.0015], \
                  'idec_lambda1':   [5.5e-2, 5.8e-2, 6.1e-2,6.5e-2, 7e-2]}  #
            # Recoreds:
            # 1. combo of (1e-3, 4e-2) got SHD=16, // icid older commit
            # 2. combo (0.001, 0.07) got (shd, tpr,fdr)=(16, 0.11764705882352941,0.6, 0.07894736842) // icid this commit
            # -- (0.002      0.065) shd=16 with tpr = 0.23, fdr=0.55
            # --  (0.0015, 0.065),  (0.0015, 0.07)
            l_o, df_o = gen_list_optparams(opts)
            print('List of opt parameters to tets are:')
            print(df_o)
            # Iterate through all optimization parameter configs
            for j in range(len(df_o)):
                print('Opt parameter to run now:')
                print(df_o.iloc[j])
                W_icd, ith  = run_icid(X, sigma_0=1.0, \
                                          k=df_o['k'][j], \
                                          lambda_1= df_o['lambda_1'], \
                                          # lambda_1= lambda1s, \
                                          idec_lambda1=df_o['idec_lambda1'][j],\
                                          beta_2 = 0.7, gamma_2=5.0, \
                                          opt_ic=df_o['opt_ic'][j], \
                                          idec_solver = df_o['idec_solver'][j], \
                                          maxit_prox_inner=500, W_true=Wtrue)
                """ Evaluations and record results """
                acc = utils.count_accuracy(Wtrue!=0, W_icd!=0)
                """ SAVE RESULTS """
                pd.DataFrame(ith).to_csv('%s/pb%dopt%d_ith_icid.csv' %(FDIR, i+1,j+1))
                print(acc)
                res.append({'alg':'ICID',
                            'd'           : X.shape[0], \
                            'deg'         : (Wtrue!=0).sum()/d, \
                            'n'           : n, \
                            'graph_type'  : 'sachs',\
                            'sem_type'    : 'sachs', \
                            'k'           :  df_o['k'][j], \
                            'ic'           : df_o['opt_ic'][j], \
                            'id'           : df_o['idec_solver'][j], \
                            'lambda_1'    :  df_o['lambda_1'][j], \
                        'idec_lambda1'    :  df_o['idec_lambda1'][j], \
                        'shd':acc['shd'], 'tpr':acc['tpr'], \
                        'fdr':acc['fdr'], 'fpr':acc['fpr'], \
                        'nnz':acc['nnz'], 'time': ith.iloc[-1]['time']}
                    )
                pd.DataFrame(res, columns=res[0].keys()).to_csv('%s/res_all.csv' %FDIR)
    #-------- NOTEARS ----------------
    if ms['notears']:
        # Parameters
        opts={'lambda_1':       [1e-1, 4e-1]}
        l_o, df_o = gen_list_optparams(opts)
        print('List of opt parameters to tets are:')
        print(df_o)
        # Iterate through all optimization parameter configs
        for j in range(len(df_o)):
            print('-------Ready to run NOTEARS ----------')
            t0 = timer()
            #
            W_no, ith_no = notears_linear(Xn, \
                                    lambda1=df_o['lambda_1'][j], \
                                    loss_type='l2', Wtrue=Wtrue)
            t_no = timer() - t0
            acc_no = utils.count_accuracy(Wtrue!=0, W_no!=0)
            print(acc_no)
            res.append({'alg':'NOTEARS',
                        'd'           : d, \
                        'deg'         : (Wtrue!=0).sum()/d, \
                        'n'           : n, \
                        'graph_type'  : 'sachs',\
                        'sem_type'    : 'sachs', \
                        'shd':acc_no['shd'], 'tpr':acc_no['tpr'], \
                        'fdr':acc_no['fdr'], 'fpr':acc_no['fpr'], \
                        'nnz':acc_no['nnz'], \
                        'time': ith_no[-1]['time']}
                    )
            pd.DataFrame(ith_no).to_csv('%s/pb%dopt%d_ith_notears.csv' %(FDIR, i+1,j+1))
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
                    'd'           : d, \
                    'deg'         : (Wtrue!=0).sum()/d, \
                    'n'           : n, \
                    'graph_type'  : 'sachs',\
                    'sem_type'    : 'sachs', \
                    'shd':acc_ges['shd'], 'tpr':acc_ges['tpr'], \
                    'fdr':acc_ges['fdr'], 'fpr':acc_ges['fpr'], \
                    'nnz':acc_ges['nnz'], 'time': t_ges}
                )
        pd.DataFrame(res, columns=res[0].keys()).to_csv('%s/res_all.csv' %FDIR)
    if len(res) > 0:
        endstr = time.strftime("%H%M%S%m%d")
        pd.DataFrame(res, columns=res[0].keys()).to_csv('%s/resall_%s.csv' %(FDIR,endstr))


