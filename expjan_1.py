# Test on baseline and algorithm O-ICID (v2)
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import time, os
import pandas as pd
import random
from itertools import product

from icid import utils
from icid.icid import run_icid, run_oicid_loc2
from icid.oicid import oicid_solver_alm, oicid_solverA_alm, oicid_solver_alm_exactfit
from aux.gen_settings import gen_list_settings, gen_list_optparams, gen_data_sem

# --------------------- Baseline methods
## ---------- if any other
#
## ---------- Goshal
from sklearn.covariance import GraphicalLassoCV
from typing import cast
## NOTE To use jax.numpy, see https://github.com/google/jax
use_jax = False
if use_jax:
    import jax.numpy as jnp
else:
    jnp = np ## Use numpy instead.
##

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
#

if __name__ == '__main__':
   timestr = time.strftime("%H%M%S%m%d")
   fdir = '../outputs/oicid_alm_%s' % timestr
   if not os.path.exists(fdir):
       os.makedirs(fdir)
   degs= [0.5, 1.0, 2.0]                        # average node degrees
   ds = [50, 200, 800]                          # number of nodes
   seeds = list(range(10))                      # random seeds to run
   graph_type, sem_type = 'ER', 'gauss'         # to change if the three combinations above can be tested in reasonable time
   sigma_0 = 1.0
   NOISE_EV = 1                                 # true, equivariance case
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
   for tt in TTS:
       tid += 1
       d = tt[0]
       deg = tt[1]
       SEED = tt[2]
       # ----- optional time budget for exceptional cases
       if deg > 2 and d > 200:
           # Force stop any method after <N> hours
           MAXTIME = N*3600 # to be imposed for each method to run
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
       print('\n========== Ghoshal =============\n')
       if False:
           df_a = pd.DataFrame
           # df_a  <- pandas DataFrame containing at least one row of scores {'shd': , 'tpr': , 'fdr': , 'fpr': , 'nnz': } of the solution vs W_true
           # test settings
           df_a['tid'] = tid
           df_a['graph_type'] = graph_type
           df_a['sem_type'] = sem_type
           df_a['seed'] = SEED
           df_a['noise_ev'] = NOISE_EV
           df_a['d'] = d
           df_a['deg'] = deg
           df_a['n'] = n
           df_a['model'] = 'Ghoshal'
           df_a['method'] = 'Ghoshal'
           # df_a['options1'] = #
           # df_a['options2'] = #

           # Append and save
           if tid ==1:
               res = pd.DataFrame
               res = df_a.tail(1)
           else:
               res = pd.concat([res, df_a.tail(1)])
           res.to_csv('%s/res_ghoshal.csv' %fdir)

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


