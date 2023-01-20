# Test on baseline and algorithm O-ICID (v2)
import sys
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import time, os
import pandas as pd
import random
from itertools import product

from icid import utils
from icid.icid import run_icid
from icid.oicid import oicid_solver_alm, oicid_solver_alm_exactfit
from aux.gen_settings import gen_list_settings, gen_list_optparams, gen_data_sem

# --------------------- Baseline methods
## ---------- if any other
#
## ---------- Goshal
# The following implementation of Ghoshal is taken from
# https://github.com/ermongroup/BCD-Nets/blob/main/baselines.py
from sklearn.covariance import GraphicalLassoCV
graphical_lasso_iters = 10_000  # default is 100

def ghoshal_fit(X, O):
    """Structure estimation by Ghoshal's algorithm.

    Parameters
    ----------
    X : array-like
        Samples, shape=(#samples, #variables).
    O : array-like
        Precision matrix.

    Returns
    -------
    array-like
        Estimated adjacency matrix.
    """
    d = X.shape[1]
    B = np.zeros((d, d))
    D = np.eye(d)
    for _ in range(d):
        i = np.argmin(np.diag(O * D))
        B[i, :] = -O[i, :] / O[i, i]
        B[i, i] = 0
        O = O - (O[:, i][:, None] @ O[i, :][None, :]).T / O[i, i]
        O[i, i] = np.inf
    return B

# def ghoshal_oracle(X, ground_truth_W):
#     """ Ghoshal using true precision matrix. """
#     O = (np.eye(d) - ground_truth_W) @ (np.eye(d) - ground_truth_W).T # TODO check
#     return ghoshal_fit(X, O)

def ghoshal(X):
    """ Ghoshal using estimated precision matrix. """
    graphical_lasso_success = True
    try:
        clf = GraphicalLassoCV(max_iter=graphical_lasso_iters)
        clf.fit(X)
        O = clf.get_precision()
    except:
        graphical_lasso_success = False
        O = np.linalg.pinv(np.cov(X.T))
    return ghoshal_fit(X, O), graphical_lasso_success
#

if __name__ == '__main__':
    # Settings that determine the tests of this script:
    #
    #   - range of 'ds', 'degs', 'seeds'
    #   - range of 'graph_type' is kept as constant in this script
    #
    # Other parts of the script are to be checked

    timestr = time.strftime("%H%M%S%m%d")
    fdir = '../outputs/oicid_alm_%s' % timestr
    ##
    # ------- To re-adjust nb of seeds if needed--------
    degs= [2.0, 3.0, 4.0]                        # average node degrees
    ds = [50, 200, 800]                          # number of nodes
    seeds = list(range(10))                       # random seeds to run
    # degs= [0.5, 1.0, 2.0]                        # average node degrees
    # ds = [50, 200, 800]                          # number of nodes
    # seeds = list(range(10))                      # random seeds to run
    ##
    # ---------
    graph_type, sem_type = 'SF', 'gauss'         # to change if the three combinations above can be tested in reasonable time
    sigma_0 = 1.0
    NOISE_EV = 1                                 # true, equivariance case

    # methods
    m = {
        "ghoshal": False, "o-ghoshal": False
    }
    # output csv file name
    fname = "res.csv"
    #---- modify parameters
    for a in sys.argv:
        if "=" in a:
            p_k, p_v = a.split("=")
            p_v = p_v.split(",")
            if p_k == "ds":
                ds = [int(_) for _ in p_v]
            elif p_k == "degs":
                degs = [float(_) for _ in p_v]
            elif p_k == "seeds":
                seeds = [int(_) for _ in p_v]
            elif p_k == "graph_type":
                graph_type = p_v[0]
            elif p_k == "sem_type":
                sem_type = p_v[0]
            elif p_k == "dirpath":
                fdir = p_v[0]
            elif p_k == "filename":
                fname = f"{p_v[0]}.csv"
            else:
                print("Unknown parameter:", p_k)
                pass
        else:
            m[a] = True

    if not os.path.exists(fdir):
        os.makedirs(fdir)

    #---- oicid-v1 parameters (to ignore)-----
    LAM=2e-1
    opts_v1={'opt_ic':       ['oicid-v1'], \
                'idec_solver':    ['FISTA'], \
                'k':              [25], \
                'lambda_1':       [5e-2], \
                'idec_lambda1':   [LAM]}  #
    l_o, df_o = gen_list_optparams(opts_v1)
    #--------
    tid=0
    TTS = list(product(ds, degs, seeds))
    res = None
    for tt in TTS:
        tid += 1
        d = tt[0]
        deg = tt[1]
        SEED = tt[2]
        # ----- optional time budget for exceptional cases
        # if deg > 2 and d > 200:
        #     # Force stop any method after <N> hours
        #     MAXTIME = N*3600 # to be imposed for each method to run
        # ---------
        # Generate random graph
        n = 10*d #
        spr = deg*d / d**2
        W_true, X = gen_data_sem(d=d, \
                                deg=deg, \
                                n=n,\
                                graph_type  = graph_type, \
                                sem_type    = sem_type, \
                                seed = SEED)
        # (same as before) size nxd, obeying the SEM "X.T = W_true.T  X.T + Noise"

        print('Experiment setting %d /%d (d=%d, deg=%.2f): ' %(tid, len(TTS),d,deg))

        if m["ghoshal"]:
            print('\n========== Ghoshal =============\n')
            t0 = timer()
            w_ghoshal, gl_suc = ghoshal(X)
            t_ghoshal = timer() - t0

            w_ghoshal = w_ghoshal.T
            acc_ghoshal = utils.count_accuracy(W_true != 0, w_ghoshal != 0)

            res_ghoshal = {
                # settings
                "tid": tid,
                "graph_type": graph_type,
                "sem_type": sem_type,
                "seed": SEED,
                "noise_ev": NOISE_EV,
                "d": d,
                "deg": deg,
                "n": n,
                "model": "Ghoshal",
                "method": "Ghoshal",
                # results
                "shd": acc_ghoshal["shd"],
                "tpr": acc_ghoshal["tpr"],
                "fdr": acc_ghoshal["fdr"],
                "fpr": acc_ghoshal["fpr"],
                "nnz": acc_ghoshal["nnz"],
                "time": t_ghoshal,
                "lasso_success": gl_suc,
            }
            df_a = pd.DataFrame({1:res_ghoshal}).T

            # Append and save
            if res is None:
                res = df_a
            else:
                res = pd.concat([res, df_a.tail(1)])
            # res.to_csv('%s/res_ghoshal.csv' %fdir)
            res.to_csv(f"{fdir}/{fname}")

        if m["o-ghoshal"]:
            print('\n========== O-Ghoshal =============\n')

            O_true = (np.eye(d)-W_true) @ (np.eye(d)-W_true).T  / sigma_0
            t0 = timer()
            w_oghoshal = ghoshal_fit(X, O_true)
            t_oghoshal = timer() - t0

            w_oghoshal = w_oghoshal.T
            acc_oghoshal = utils.count_accuracy(W_true != 0, w_oghoshal != 0)

            res_oghoshal = {
                # settings
                "tid": tid,
                "graph_type": graph_type,
                "sem_type": sem_type,
                "seed": SEED,
                "noise_ev": NOISE_EV,
                "d": d,
                "deg": deg,
                "n": n,
                "model": "O-Ghoshal",
                "method": "O-Ghoshal",
                # results
                "shd": acc_oghoshal["shd"],
                "tpr": acc_oghoshal["tpr"],
                "fdr": acc_oghoshal["fdr"],
                "fpr": acc_oghoshal["fpr"],
                "nnz": acc_oghoshal["nnz"],
                "time": t_oghoshal,
            }
            df_a = pd.DataFrame({1:res_oghoshal}).T

            # Append and save
            if res is None:
                res = df_a
            else:
                res = pd.concat([res, df_a.tail(1)])
            # res.to_csv('%s/res_ghoshal.csv' %fdir)
            res.to_csv(f"{fdir}/{fname}")

        if False:
            print('Experiment setting %d /%d: ' %(tid, len(TTS)))
            print('\n========== O-ICID v2=============\n')
            # True Theta
            Theta_true = (np.eye(d)-W_true) @ (np.eye(d)-W_true).T  / sigma_0
            MAXITER_ALM = 20
            MAXITER_PRIMAL = 1e3
            EPSILON = 1e-6
            if deg < 1:
                TAUX_RHO = 1.1
                LAM = 1e0
            else:
                LAM=2e-1
                TAUX_RHO = 1.5
            A_glo, ithg, optinfo = oicid_solver_alm_exactfit(Theta_true, lambda1=LAM, maxiter=MAXITER_ALM, \
                                                    maxiter_primal=MAXITER_PRIMAL, \
                                                    epsilon=EPSILON, taux_rho=TAUX_RHO, \
                                                    solver_primal = 'FISTA', \
                                                    Wtrue=W_true, verbo=1)
            acc = utils.count_accuracy(W_true!=0, A_glo!=0)
            print(acc)
            # Add test settings and Alg info
            ithg['tid'] = tid
            ithg['graph_type'] = graph_type
            ithg['sem_type'] = sem_type
            ithg['seed'] = SEED
            ithg['noise_ev'] = NOISE_EV
            ithg['d'] = d
            ithg['deg'] = deg
            ithg['n'] = n
            ithg['model'] = 'OICID'
            ithg['method'] = 'ExactFit-ALM-FISTA'
            ithg['par_lambda1'] = LAM
            ithg['par_maxit_alm'] = MAXITER_ALM
            ithg['par_maxit_primal'] = MAXITER_PRIMAL
            ithg['par_epsilon'] = EPSILON
            ithg['par_rho0'] = 1.0
            ithg['par_tau_rho'] = TAUX_RHO
            ithg['code_conver'] = optinfo['code_conver']
            ithg['tid'] = tid   # Redundant but useful for previewing results in the csv
            # Append and save
            if tid ==1:
                # cont = ithg
                resb = pd.DataFrame
                resb = ithg.tail(1)
            else:
                # cont = pd.concat([cont, ithg])
                resb = pd.concat([resb, ithg.tail(1)])
            # cont.to_csv('%s/iterh_alm-fista.csv' %fdir)
            resb.to_csv('%s/res_oicidv2.csv' %fdir)

        if False:
            print('Experiment setting %d /%d: ' %(tid, len(TTS)))
            print('\n========== O-ICID v1=============\n')
            W_glo, ithg1, idh, W0  = run_icid(X, sigma_0=1.0, \
                                    k=df_o['k'][0], \
                                    lambda_1=df_o['lambda_1'][0] , \
                                    idec_lambda1=df_o['idec_lambda1'][0],\
                                    beta_2 = 0.7, gamma_2=1.0, \
                                    opt_ic='ideal', \
                                    idec_solver = df_o['idec_solver'][0], \
                                    maxit_ama=2, \
                                    maxit_prox_inner=500, W_true=W_true)
            acc = utils.count_accuracy(W_true!=0, W_glo!=0)
            print(acc)
            # Add test settings and Alg info
            ithg1['graph_type'] = graph_type
            ithg1['sem_type'] = sem_type
            ithg1['seed'] = SEED
            ithg1['noise_ev'] = NOISE_EV
            ithg1['d'] = d
            ithg1['deg'] = deg
            ithg1['n'] = n
            ithg1['model'] = 'OICID'
            ithg1['method'] = 'FISTA-LoRAM'
            ithg1['par_lambda1'] = df_o['idec_lambda1'][0]
            ithg1['par_maxit_ama'] = 2
            ithg1['par_maxit_loram_prox'] = 500
            ithg1['par_loram_k'] = df_o['k'][0]
            ithg1['tid'] = tid   # Redundant but useful for previewing results in the csv
            # Append and save
            if tid ==1:
                # cont1 = ithg1      # skip iteration
                res1 = pd.DataFrame
                res1 = ithg1.tail(1) # Ensure that the last row (if there is more than one row in ithg1) is the final result
            else:
                # cont1 = pd.concat([cont1, ithg1])
                res1 = pd.concat([res1, ithg1.tail(1)])  #
            # cont1.to_csv('%s/iterh_fista-loram.csv' %fdir)
            res1.to_csv('%s/res_oicidv1.csv' %fdir)


