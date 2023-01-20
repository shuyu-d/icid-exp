# Test on baseline and algorithm O-ICID (v2): with an option to standardization of the variable X
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import time, os
import pandas as pd
import random
from itertools import product

from icid import utils
from icid.icid import run_icid, run_oicid_loc2
from icid.oicid import oicid_solver_alm, oicid_solver_alm_exactfit
from aux.gen_settings import gen_list_settings, gen_list_optparams, gen_data_sem


## ---- Standardization -----
def standardize(B, X, varN=None):
    # B:        Weighted adjacency matrix of the causal structure
    # varN:     Diagonal of noise variances
    d = B.shape[0]
    if varN == None:
        varN = np.diag(np.eye(d)) # EV case
    CovX = (np.linalg.inv(np.eye(d)-B) @ np.diag(varN) ) @ np.linalg.inv(np.eye(d)-B)
    ThetaX = ((np.eye(d)-B) @ np.diag(1/varN)) @ (np.eye(d)-B).T
    D = np.sqrt(np.diag(CovX))  # standard deviations of Xi's
    # Standardization of X is equivalent to the following transformation on InvCov:
    ThetaX_st =  np.diag(D)@ ThetaX @ np.diag(D)
    # Standardization of X
    #   made sure that X is of size nxd
    X_st = X @ np.diag(1/D)
    return ThetaX_st, X_st, ThetaX

# ----- Baseline methods
## if any other
## //---- Ghoshal and O-Ghoshal (to verify in case of change)
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
## ---- Ghoshal and O-Ghoshal //

if __name__ == '__main__':
    # Settings that determine the tests of this script:
    #
    #   - Function 'standardize' for the input data/matrix of this var-sortability test
    #   - range of 'ds', 'degs', 'seeds'
    #   - range of 'graph_type' is kept as constant in this script, but see 'expjan_2SF.py' for same experiment on 'SF' graphs
    #
    # Other parts of the script are to be checked
    timestr = time.strftime("%H%M%S%m%d")
    fdir = '../outputs/oicid_alm_%s' % timestr
    if not os.path.exists(fdir):
        os.makedirs(fdir)
    # graph type, size, seeds
    degs= [2.0, 3.0, 4.0]                        # average node degrees
    ds = [50, 200, 800]                          # number of nodes
    seeds = list(range(10))                      # random seeds to run

    # kept as constant in this script
    graph_type, sem_type = 'SF', 'gauss'         # to change if the three combinations above can be tested in reasonable time
    sigma_0 = 1.0
    NOISE_EV = 1                                 # true, equivariance case

    tid=0
    TTS = list(product(ds, degs, seeds))
    for tt in TTS:
        tid += 1
        d = tt[0]
        deg = tt[1]
        SEED = tt[2]
        # ----- optional time budget for exceptional cases
        # if deg > 3 and d > 200:
        #     # Force stop any method after <N> hours
        #     # MAXTIME = N*3600 # to be imposed for each method to run
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

        ## ----- Standardization of X and Theta (precision matrix of X)
        Theta_true, X, _  = standardize(W_true, X)  # last output is the original inv covariance
        ## -----

        print('Experiment setting %d /%d (d=%d, deg=%.2f): ' %(tid, len(TTS),d,deg))
        print('\n========== Ghoshal =============\n')
        print('\n========== O-Ghoshal =============\n')
        # if False:
        #     # to re-add from src/

        if False:
            print('Experiment setting %d /%d: ' %(tid, len(TTS)))
            print('\n========== O-ICID v2=============\n')
            # True Theta
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


