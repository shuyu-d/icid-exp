# Test on Algorithm O-ICID (v2): test variants of FISTA for O-ICID global
import numpy as np
from timeit import default_timer as timer
import time, os, sys
import pandas as pd
import random
from itertools import product

from icid import utils
from icid.oicid import oicid_solver_alm_exactfit, oicid_alm_exactfit_alliters
from aux.gen_settings import gen_graph_dag_tril, gen_graph_dag, gen_data_sem

from cvxopt import spmatrix, amd
import chompack as cp
from scipy.sparse import coo_matrix, csc_matrix
from external.test_linear import notears_linear
# import golem - todo
#

# /// Optional individual timeout by MAXTIME
import signal
from contextlib import contextmanager
class TimeoutException(Exception): pass
@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
# ///

def ghoshal_fit(O, Btrue):
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
    d = O.shape[1]
    B = np.zeros((d, d))
    D = np.eye(d)
    for t in range(d):
        i = np.argmin(np.diag(O * D))
        B[i, :] = -O[i, :] / O[i, i]
        B[i, i] = 0
        O = O - (O[:, i][:, None] @ O[i, :][None, :]).T / O[i, i]
        O[i, i] = np.inf
        if False: #
            Bb=B.copy()
            wdi = np.linalg.norm(Bb.T-Btrue)
            acc_ = utils.count_accuracy(Btrue != 0, Bb.T!= 0)
    return B

## ---- Standardization -----
def standardize(B, X=None, varN=None):
    # B:        Weighted adjacency matrix of the causal structure
    # varN:     Diagonal of noise variances
    d = B.shape[0]
    if varN == None:
        varN = np.diag(np.eye(d)) # EV case
    elif varN == 'const':
        varN = abs(np.random.normal(scale=1.2))*np.diag(np.eye(d))
    else:
        varN = abs(np.random.normal(scale=1.1, size=[d]))
    CovX = ( (np.linalg.inv(np.eye(d)-B)).T @ np.diag(varN) ) @ np.linalg.inv(np.eye(d)-B)
    ThetaX = ((np.eye(d)-B) @ np.diag(1/varN)) @ (np.eye(d)-B).T
    D = np.sqrt(np.diag(CovX))  # standard deviations of Xi's

    # Standardization of X is equivalent to the following transformation on InvCov:
    ThetaX_st =  (np.diag(D)@ ThetaX) @ np.diag(D)
    # Standardization of X
    if X != None:
        #   made sure that X is of size nxd
        X_st = X @ np.diag(1/D)
    else:
        X_st = None
    return ThetaX_st, X_st, ThetaX


if __name__ == '__main__':
    timestr = time.strftime("%H%M%S%m%d")
    # Methods to run
    ms = {'notears': False,\    # todo
           'golem': False, \    # todo
          'o-ghoshal': False, \        # done
          'chol1': False, \            # done
          'chol2': False, \            # done
          'v2po': False \              # oicid-v2 running, almost done
          }

    # SYS INUPT args
    ni = len(sys.argv)
    if ni > 4:
        # Require 4 sys input args
        #   python expjan_2_others.py {ALG} {graph_type} {SEED} {iset}
        #   ALG:            notears, golem
        #   graph_type:     ER, SF
        #   SEED:           0-9
        #   iset:           (optional) id of this test to help identify output folders/files
        iset, ALG, graph_type, SEED = int(sys.argv[4]), sys.argv[1], \
                                      sys.argv[2], int(sys.argv[3])
        print('Random seed:%d' %SEED)
        # make sure ALG is in the list 'ms'
        ms[ALG] = True
    else:
        print('Nb of expected input args not matched. Stopping.' %len(TTS))
        TTS = []

    # For each combination of (ALG, graph_type, SEED), test the settings
    #       determined by (d, deg, DO_STD) in the following ranges
    ds = [50, 100, 200, 400]
    if graph_type == 'ER':
        degs= [1.0, 2.0] #
    else:
        degs= [2.0, 4.0] #
    do_standardize = [False, True]  # when 'False': test on original X and InvCov
    # ds = [50 ]
    # degs= [2.0]
    # seeds = list(range(1))
    # do_standardize = [True]
    TTS = list(product(ds, degs, do_standardize ))

    # Constants
    sigma_0 = 1.0
    MAXTIME = 3600 # Only active for d>6400, to be updated for d<=6400 later
    #----
    tid=0
    res = []
    if sum(ms.values()) < 1:
        print('No method to run (%d tests)' %len(TTS))
        TTS =[]
    else:
        mlist = {k: v for k, v in ms.items() if v}
        mlist = list(mlist.keys())  # in parallel runs 'ms' has only one method (True) to run
        fdir = '../outputs/oicid_expjan2%s_%s_%s' % (graph_type, mlist[0],timestr)
        if not os.path.exists(fdir):
            os.makedirs(fdir)
    for tt in TTS:
        tid += 1
        d, deg, DO_STD = tt[0], tt[1], tt[2]
        n = 10 * d
        # Generate random graph
        tg=timer()
        spr = deg*d / d**2
        W_true, X = gen_data_sem(d=d, \
                                deg=deg, \
                                n=n,\
                                graph_type  = graph_type, \
                                sem_type    = sem_type, \
                                seed = SEED)
        # Standardization of X and Theta (precision matrix of X)
        Theta_Xstd, X_std, Theta_X  = standardize(W_true, X=X)  # last output is the original inv covariance
        if DO_STD:
            Theta_true = Theta_Xstd  # InvCov after standardization
            X          = X_std       # Data samples afte standardization
        else:
            Theta_true = Theta_X     # original InvCov
        #-----
        tg = timer() - tg
        print('Graph and input matrix generated after %.2e sec.' %tg)

        print('\n||--- Experiment setting %d /%d (d=%d, deg=%.2f): ' %(tid, len(TTS),d, deg))
        if ms['notears']:
            # --------To verify-------
            print('------- NOTEARS ----------')
            t0 = timer()
            W_no, ith_no = notears_linear(X, \
                                   lambda1=0.2, \
                                    h_tol = 1e-5, \
                                    loss_type='l2', Wtrue=Wtrue)
            t_no = timer() - t0
            acc = utils.count_accuracy(Wtrue!=0, W_no!=0)
            print(acc)
            # Append and save
            acc['d'] = d
            acc['deg'] = deg
            acc['do_std'] = DO_STD
            acc['seed'] = SEED
            acc['n'] = n
            #
            acc['time'] = t_no
            acc['wdist'] = np.linalg.norm(W_no-W_true)
            acc['alg'] = 'notears'
            acc['tid'] = tid
            acc = pd.DataFrame([acc])
            if len(res) < 1:
                res = pd.DataFrame
                res = acc.tail(1)
            else:
                res = pd.concat([res, acc])
            res.to_csv('%s/res_notears.csv' %fdir)

        print('\n||--- Experiment setting %d /%d (d=%d, deg=%.2f): ' %(tid, len(TTS),d, deg))
        if ms['golem']:
            print('------- GOLEM ----------')
            # To add

            # Append and save
            acc['d'] = d
            acc['deg'] = deg
            acc['do_std'] = DO_STD
            acc['seed'] = SEED
            acc['n'] = n
            #
            acc['time'] = t_no
            acc['wdist'] = np.linalg.norm(W_no-W_true)
            acc['alg'] = 'notears'
            acc['tid'] = tid
            acc = pd.DataFrame([acc])
            if len(res) < 1:
                res = pd.DataFrame
                res = acc.tail(1)
            else:
                res = pd.concat([res, acc])
            res.to_csv('%s/res_golem.csv' %fdir)

        # print('\n||--- Experiment setting %d /%d (d=%d, deg=%.2f): ' %(tid, len(TTS),d, deg))
        # if ms['chol2']:
        #     print('\n========== chol 2 =============')
        #     tpre = timer()
        #     Smat = Theta_true.copy()
        #     # Mask must not contain diagonal
        #     Smat = np.tril(Smat, k=0)
        #     ind_2d = np.nonzero(Smat)
        #     I = (ind_2d[0]).tolist()  # converts numpy dtype to native python types
        #     J = (ind_2d[1]).tolist()
        #     V = (Smat[ind_2d]).tolist()
        #     print('spmatrix format prepared in %.2e sec.'%(timer()-tpre))
        #     #
        #     t0=timer()
        #     Smatt = spmatrix(V, I, J, (d,d))
        #     #----ordering by AMD ----
        #     # pp = amd.order
        #     #----ordering by variance-----
        #     pp =  (np.argsort(np.diag(Theta_true))).tolist()
        #     # pp =  [int(pp[ii]) for ii in range(len(pp))]
        #     # Compute symbolic factorization using AMD ordering
        #     symb = cp.symbolic(Smatt, p=pp)
        #     PL = cp.cspmatrix(symb)
        #     PL += Smatt # L is the lower-triangular rep of P'AP, where P is from the reordering of p
        #     #
        #     cp.cholesky(PL)
        #     tcp = timer() - t0
        #     mm = PL.spmatrix()
        #     mm = mm[symb.ip,symb.ip]
        #     # convert mm back to np array
        #     mmI = np.array(mm.I).ravel() # ravel is correct for 'matrix' type of size px1
        #     mmJ = np.array(mm.J).ravel()
        #     mmV = np.array(mm.V).ravel()
        #     mnp = np.array(csc_matrix((mmV, (mmI, mmJ)), \
        #                     shape=(d, d)).todense())
        #     mnp += -np.diag(np.diag(mnp))
        #     mnp[abs(mnp)<1e-1] = 0
        #     acc = utils.count_accuracy(W_true!=0, mnp!=0)
        #     print('csp chol time: %.2e' %tcp)
        #     print(acc)
        #     print('RMSE(B)=%.2e' %(np.linalg.norm(mnp-W_true)/d))
        #     # Append and save
        #     acc['d'] = d
        #     acc['deg'] = deg
        #     acc['do_std'] = DO_STD
        #     acc['seed'] = SEED
        #     #
        #     acc['time'] = tcp
        #     acc['wdist'] = np.linalg.norm(mnp-W_true)
        #     acc['alg'] = 'chol2'
        #     acc['tid'] = tid
        #     acc = pd.DataFrame([acc])
        #     if len(res) < 1:
        #         res = pd.DataFrame
        #         res = acc.tail(1)
        #     else:
        #         res = pd.concat([res, acc])
        #     res.to_csv('%s/res_chol2.csv' %fdir)


        # if ms["o-ghoshal"]:
        #     print('\n========== O-Ghoshal =============\n')
        #     if d <= 6400:
        #         t0 = timer()
        #         w_og = ghoshal_fit(Theta_true.copy(), W_true)
        #         w_og[abs(w_og)<1e-1] = 0
        #         t_og = timer() - t0
        #         w_og = w_og.T
        #         MAXTIME = t_og
        #     else:
        #         try:
        #             with time_limit(MAXTIME):
        #                 t0 = timer()
        #                 w_og = ghoshal_fit(Theta_true.copy(), W_true)
        #                 w_og[abs(w_og)<1e-1] = 0
        #                 t_og = timer() - t0
        #                 w_og = w_og.T
        #         except TimeoutException as e:
        #             print("Timed out!")
        #             w_og = np.zeros(W_true.shape)
        #             t_og = 0
        #     acc = utils.count_accuracy(W_true != 0, w_og != 0)
        #     print('Elapsed time: %.2e'%t_og)
        #     print(acc)
        #     print('RMSE(B)=%.2e' %(np.linalg.norm(w_og-W_true)/d))
        #     # Append and save
        #     acc['d'] = d
        #     acc['deg'] = deg
        #     acc['seed'] = SEED
        #     acc['do_std'] = DO_STD
        #     #
        #     acc['time'] = t_og
        #     acc['wdist'] = np.linalg.norm(w_og-W_true)
        #     acc['alg'] = 'o-ghoshal'
        #     acc['tid'] = tid
        #     acc = pd.DataFrame([acc])
        #     if len(res) < 1:
        #         res = pd.DataFrame
        #         res = acc.tail(1)
        #     else:
        #         res = pd.concat([res, acc])
        #     res.to_csv('%s/res_oghoshal.csv' %fdir)
    if len(res) >=1 :
        res['graph_type'] = graph_type
        res['sem_type'] = sem_type
        res.to_csv('%s/res_.csv' %fdir)


