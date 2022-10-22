from matplotlib import use
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

import lingam ## DirectLiNGAM
import sumu ## Gadget

## Ghoshal
from sklearn.covariance import GraphicalLassoCV
from typing import cast
## NOTE To use jax.numpy, see https://github.com/google/jax
use_jax = False
if use_jax:
    import jax.numpy as jnp
else:
    jnp = np ## Use numpy instead.
##

def test_save_res_icid(W_true, X, k=25, \
                       optp1=2e-1, optp2=1e-1, \
                       opt_ic='sk', \
                       tid=None, fdir='outputs/exp0_'):
    if not os.path.exists(fdir):
        os.makedirs(fdir)
    """ ICID """
    print('-------Ready to run ICID ----------')
    # 'opt_ic' controls which IC method to use (ideal, sklearn, QUIC)
    W_icd, ith_icd  = run_icid(X, sigma_0=1.0, k=k,\
                              lambda_1=optp1, \
                              idec_lambda1=optp2, \
                              beta_2 = 0.7, gamma_2=1.0, \
                              opt_ic=opt_ic, \
                              maxit_prox_inner=500, W_true=W_true)
    """ Evaluations and record results """
    acc = utils.count_accuracy(W_true!=0, W_icd!=0)
    """ SAVE RESULTS """
    pd.DataFrame(ith_icd).to_csv('%s/%s_ith_icid.csv'\
                                    %(fdir, tid))
    return acc, ith_icd


## The following implementation of Ghoshal is taken from 
## https://github.com/ermongroup/BCD-Nets/blob/main/baselines.py
graphical_lasso_iters = 10_000  # default is 100
def ghoshal(X, ground_truth_W):
    # No code available to our knowledge, have to re-implement this method
    _, d = X.shape
    try:
        clf = GraphicalLassoCV(max_iter=graphical_lasso_iters)
        clf.fit(X)
        O = clf.get_precision()
        O = cast(jnp.ndarray, O)
        O_empirical = np.linalg.pinv(jnp.cov(X.T))
        ground_truth_O = (jnp.eye(d) - ground_truth_W) @ (jnp.eye(d) - ground_truth_W).T
        O_dist = jnp.sqrt(jnp.mean((O - ground_truth_O) ** 2))
        empirical_dist = jnp.sqrt(jnp.mean((O_empirical - ground_truth_O) ** 2))
        if empirical_dist < O_dist:
            O = O_empirical
    except:
        graphical_lasso_success = False
        O_empirical = np.linalg.pinv(jnp.cov(X.T))
        O = O_empirical

    # A minor hack here: sometimes GraphicalLassoCV completely fails
    # to find a good precision, since we're pretty underdetermined. In
    # this case we're better off just using the unregularized
    # empirical estimate. In the interest of giving a stronger
    # baseline, we'll use the real W to choose when to do this or not,
    # since it might be that a better precision estimator would do
    # better here

    B = np.zeros((d, d))
    D = np.eye(d)
    for d in range(d):
        i = np.argmin(np.diag(O * D))
        B[i, :] = -O[i, :] / O[i, i]
        B[i, i] = 0
        O = O - (O[:, i][:, None] @ O[i, :][None, :]).T / O[i, i]
        O[i, i] = np.inf
    return O, B


if __name__ == '__main__':
    timestr = time.strftime("%H%M%S%m%d")
    FDIR = 'outputs/exp3_%s' % timestr
    if not os.path.exists(FDIR):
        os.makedirs(FDIR)
    # Generate input parameters
    # NOTE: the following 5 parameters determine the experimental setting.
    # This script 'exp_3' is meant to test with
    # varying values of 'n' (ns below).
    # ds            = np.array([100])   # d
    # degs          = [0.5, 1.0, 2]     # degree
    # graph_types   = ['ER', 'SF']      # graph type
    # sem_types     = ['gauss'] # TODO: enrich this list with 'exponential'and 'gumbel'
    # ns            = np.ceil([2, 4, 8, 16, 32]*ds).astype(np.uint)

    ## test
    ds            = np.array([10])   # d
    degs          = [1.0, 2.0]     # degree
    graph_types   = ['ER',]      # graph type
    sem_types     = ['gauss', 'exp', 'gumbel'] # TODO: enrich this list with 'exponential'and 'gumbel'
    ns            = np.ceil([10, 100]*ds).astype(np.uint)

    #
    pbs_l, pbs = gen_list_settings(d=ds, degs=degs, \
                                 n=ns, \
                                 graph_types=graph_types, \
                                 sem_types=sem_types)

    # Methods to run
    # NOTE: Modify the booleans below or use shell input args such as
    #       python exp_3.py icid        # run icid separately
    #       python exp_3.py notears     # run icid separately
    #       python exp_3.py icid ges    # run icid and ges
    #
    # TODO: Prioritize the following ones that need to be
    #       added to this script:
    #               LINGAM, GHOSHAL and GADGET.
    #
    #       Skip 'ges' (it is very slow even for d=100 here) or let
    #       'ges' run for 'ds=25' (line 46) exceptionally.
    #
    # NOTE: Good news is all results are recorded and saved to csv
    #       after each individual run (i, method, j). It is still
    #       recommended to run each method separately.

    ms = {'icid':   False, \
          'ges':    False,\
          'notears':False,\
          'directlingam':False,\
          'ghoshal':False,\
          'gadget':False}
    ni = len(sys.argv)
    for i in range(ni):
        ms[sys.argv[i]] = True
    # Iterate through all problem settings ('pbs')
    res = []
    for i in range(len(pbs)):
        print('='*50)
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
            opts={'opt_ic':         ['ideal','sk'], \
                  'k':              [25], \
                  'lambda_1':       [1e-1, 4e-1], \
                  'idec_lambda1':   [1e-1]}
            l_o, df_o = gen_list_optparams(opts)
            print('List of opt parameters to tets are:')
            print(df_o)
            # Iterate through all optimization parameter configs
            for j in range(len(df_o)):
                print('Opt parameter to run now:')
                print(df_o.iloc[j])
                acc, ith = test_save_res_icid(Wtrue, X, \
                                    k    =df_o['k'][j], \
                                    optp1=df_o['lambda_1'][j] , \
                                    optp2=df_o['idec_lambda1'][j] , \
                                    opt_ic=df_o['opt_ic'][j],\
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
                            'ic'           : df_o['opt_ic'][j], \
                            'lambda_1'    : df_o['lambda_1'][j], \
                        'idec_lambda1'    : df_o['idec_lambda1'][j], \
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
                # w_notears, _ = NOTEARS
                W_no, ith_no = notears_linear(X, \
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
        #-------- DirectLiNGAM ----------------
        if ms['directlingam']:
            print('-------Ready to run DirectLiNGAM ----------')
            model = lingam.DirectLiNGAM()

            t0 = timer()
            model.fit(X)
            w_dlingam = model.adjacency_matrix_.T ## Need to transpose!
            t_dlingam = timer() - t0
            
            acc_dlingam = utils.count_accuracy(Wtrue!=0, w_dlingam!=0)
            print(acc_dlingam)
            res.append({'alg':'DirectLiNGAM',
                        'd'           : pbs['d'][i], \
                        'deg'         : pbs['deg'][i], \
                        'n'           : pbs['n'][i], \
                        'graph_type'  : pbs['graph_type'][i],\
                        'sem_type'    : pbs['sem_type'][i], \
                        'shd':acc_dlingam['shd'], 'tpr':acc_dlingam['tpr'], \
                        'fdr':acc_dlingam['fdr'], 'fpr':acc_dlingam['fpr'], \
                        'nnz':acc_dlingam['nnz'], 'time': t_dlingam}
                    )
            pd.DataFrame(res, columns=res[0].keys()).to_csv('%s/res_all.csv' %FDIR)
        #-------- Gadget ----------------
        if ms['gadget']:
            ## The following settings are taken from
            ## https://github.com/ermongroup/BCD-Nets/blob/main/baselines.py#L39
            print('-------Ready to run Gadget ----------')
            _,d = X.shape
            K = d - 1 if d < 16 else 15
            n_sumu_iters = 50_000
            subset_size = 50 ## for eval.

            data = sumu.Data(X, discrete=False)
            params = {
                "data": data,
                "scoref": "bge",  # Or "bdeu" for discrete data.
                "ess": 10,  # If using BDeu.
                "max_id": -1,  # Max indegree, -1 for none.
                "K": K,  # Number of candidate parents per variable (< n).
                "d": 3,  # Max size for parent sets not constrained to candidates.
                "cp_algo": "greedy-lite",  # Algorithm for finding the candidate parents.
                "mc3_chains": 48,  # Number of parallel Metropolis coupled Markov chains.
                "burn_in": int(
                    0.5 * n_sumu_iters
                ),  # Number of burn-in iterations in the chain.
                "iterations": n_sumu_iters,  # Number of iterations after burn-in.
                "thinning": 10,
            }  # Sample a DAG at every nth iteration.

            t0 = timer()
            g = sumu.Gadget(**params)
            dags,_ = g.sample()
            t_gadget = timer() - t0

            ## NOTE
            ## According to the above implementation, evaluate the results by 
            ## averaging the stats of subsampled adjacency matrices.

            # Conversion to binary adj. matrices.
            ws_gadget = [sumu.bnet.family_sequence_to_adj_mat(dag).T for dag in dags]

            res_gadget = []
            for w_gadget in ws_gadget:
                res_gadget.append(utils.count_accuracy(Wtrue!=0, w_gadget!=0))
            acc_gadget = {}
            for key in res_gadget[0].keys():
                acc_gadget[key] = np.mean([_[key] for _ in res_gadget])
            
            print(acc_gadget)
            res.append({'alg':'Gadget',
                        'd'           : pbs['d'][i], \
                        'deg'         : pbs['deg'][i], \
                        'n'           : pbs['n'][i], \
                        'graph_type'  : pbs['graph_type'][i],\
                        'sem_type'    : pbs['sem_type'][i], \
                        'shd':acc_gadget['shd'], 'tpr':acc_gadget['tpr'], \
                        'fdr':acc_gadget['fdr'], 'fpr':acc_gadget['fpr'], \
                        'nnz':acc_gadget['nnz'], 'time': t_gadget}
                    )
            pd.DataFrame(res, columns=res[0].keys()).to_csv('%s/res_all.csv' %FDIR)
        #-------- Ghoshal ----------------
        if ms['ghoshal']:
            print('-------Ready to run Ghoshal ----------')
            t0 = timer()
            _, w_ghoshal = ghoshal(X, Wtrue)
            t_ghoshal = timer() - t0
            
            w_ghoshal = w_ghoshal.T ## TODO Check
            acc_ghoshal = utils.count_accuracy(Wtrue!=0, w_ghoshal!=0)
            print(acc_ghoshal)
            res.append({'alg':'Ghoshal',
                        'd'           : pbs['d'][i], \
                        'deg'         : pbs['deg'][i], \
                        'n'           : pbs['n'][i], \
                        'graph_type'  : pbs['graph_type'][i],\
                        'sem_type'    : pbs['sem_type'][i], \
                        'shd':acc_ghoshal['shd'], 'tpr':acc_ghoshal['tpr'], \
                        'fdr':acc_ghoshal['fdr'], 'fpr':acc_ghoshal['fpr'], \
                        'nnz':acc_ghoshal['nnz'], 'time': t_ghoshal}
                    )
            pd.DataFrame(res, columns=res[0].keys()).to_csv('%s/res_all.csv' %FDIR)
    if len(res) > 0:
        endstr = time.strftime("%H%M%S%m%d")
        pd.DataFrame(res, columns=res[0].keys()).to_csv('%s/resall_%s.csv' %(FDIR,endstr))



