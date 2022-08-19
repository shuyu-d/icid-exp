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


if __name__ == '__main__':
    timestr = time.strftime("%H%M%S%m%d")
    FDIR = 'outputs/exp2f_%s' % timestr
    # Generate input parameters
    ds            = np.array([50,200]) #x5  #,300,400,500])
    degs          = [1] #[4, 2, 1] #x3 # 0.5
    graph_types   = ['ER'] #'SF']
    sem_types     = ['gauss'] #['gauss','exp'] # x2
    R_N2D         = 32
    #
    pbs_l, pbs = gen_list_settings(d=ds, degs=degs, \
                                 graph_types=graph_types, \
                                 sem_types=sem_types)
    # Methods to run
    #
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
        if ms['icid'] or ms['ideal']:
            if ms['ideal']:
                IC = 'ideal'
            else:
                IC = 'emp_gs'
            # Parameters of ICID algorithm
            opts={'opt_ic':         [IC], \
                'idec_solver':     ['FISTA'], \
                  'k':              [25], \
                  'lambda_1':       [5e-2], \
                  'idec_lambda1':   [1e-2, 4e-2, 5e-2]}  #
            l_o, df_o = gen_list_optparams(opts)
            print('List of opt parameters to tets are:')
            print(df_o)
            # Iterate through all optimization parameter configs
            for j in range(len(df_o)):
                print('Opt parameter to run now:')
                print(df_o.iloc[j])
                if pbs['deg'][i] > 2:
                    lambda1s = np.linspace(1e-2,1e-1,10)
                else:
                    lambda1s = df_o['lambda_1'][j]
                W_icd, ith, idh  = run_icid(X, sigma_0=1.0, \
                                          k=df_o['k'][j], \
                                          lambda_1= lambda1s, \
                                          idec_lambda1=df_o['idec_lambda1'][j],\
                                          beta_2 = 0.7, gamma_2=5.0, \
                                          opt_ic=df_o['opt_ic'][j], \
                                          idec_solver = df_o['idec_solver'][j], \
                                          maxit_prox_inner=500, W_true=Wtrue)
                """ Evaluations and record results """
                acc = utils.count_accuracy(Wtrue!=0, W_icd!=0)
                """ SAVE RESULTS """
                pd.DataFrame(ith).to_csv('%s/pb%dopt%d_ith_icid.csv' %(FDIR, i+1,j+1))
                pd.DataFrame(idh).to_csv('%s/pb%dopt%d_ith_idec.csv' %(FDIR, i+1,j+1))
                print(acc)
                lambda_1 = ith[ith['niter']==-1].iloc[0]['subpb_name']
                res.append({'alg':'ICID',
                            'd'           : pbs['d'][i], \
                            'deg'         : pbs['deg'][i], \
                            'n'           : pbs['n'][i], \
                            'graph_type'  : pbs['graph_type'][i],\
                            'sem_type'    : pbs['sem_type'][i], \
                            'k'           : df_o['k'][j], \
                            'ic'           : df_o['opt_ic'][j], \
                            'id'           : df_o['idec_solver'][j], \
                            'lambda_1'    : lambda_1, \
                        'idec_lambda1'    : df_o['idec_lambda1'][j], \
                        'shd':acc['shd'], 'tpr':acc['tpr'], \
                        'fdr':acc['fdr'], 'fpr':acc['fpr'], \
                        'nnz':acc['nnz'], 'time': ith.iloc[-1]['time']}
                    )
                pd.DataFrame(res, columns=res[0].keys()).to_csv('%s/res_all.csv' %FDIR)
        #-------- NOTEARS ----------------
        if ms['notears']:
            Xn = X[:10*pbs['d'][i], :]
            print('For Notears: n = 10d (%d)' %Xn.shape[0])
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
                            'd'           : pbs['d'][i], \
                            'deg'         : pbs['deg'][i], \
                            'n'           : pbs['n'][i], \
                            'graph_type'  : pbs['graph_type'][i],\
                            'sem_type'    : pbs['sem_type'][i], \
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



