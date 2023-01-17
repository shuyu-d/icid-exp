# Test on Algorithm O-ICID v2: Case NV
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import time, os
import pandas as pd
import random
from itertools import product

from icid import utils
from icid.icid import run_icid, run_oicid_loc2
from icid.oicid import oicid_solver_alm, oicid_solverA_alm, oicid_solver_alm_exactfit, oicidNV_solver_alm_exactfit
from aux.gen_settings import gen_list_settings, gen_list_optparams, gen_data_sem, gen_graph_dag_with_markovblanket



if __name__ == '__main__':
   d = 50
   timestr = time.strftime("%H%M%S%m%d")
   fdir = '../outputs/oicid_alm_%s' % timestr
   if not os.path.exists(fdir):
       os.makedirs(fdir)
   # degs= [0.5, 1.0, 2.0, 3.0] #
   degs= [1.0]
   seeds = list(range(10))
   graph_type, sem_type = 'ER', 'gauss'
   sigma_0 = 1.0
   NOISE_EV = 1 # true
   n = 100 #
   LAM=2e-1
   #
   opts={'opt_ic':       ['oicid-new2'], \
       'idec_solver':    ['FISTA'], \
       'k':              [25], \
       'lambda_1':       [5e-2], \
       'idec_lambda1':   [LAM]}  #
   l_o, df_o = gen_list_optparams(opts)
   cont = []
   cont1 = []
   tid=0
   TTS = list(product(degs, seeds))
   for tt in TTS:
       tid += 1
       deg = tt[0]
       SEED = tt[1]
       # Generate random graph
       spr = deg*d / d**2
       W_true, MBs = gen_graph_dag_with_markovblanket(
                                   d=d, deg=deg,
                                   graph_type=graph_type,
                                   seed=SEED) # seed=1 very good, 2 good
       X = utils.simulate_linear_sem(W_true, n, sem_type)
       # True Theta
       # Theta_true = (np.eye(d)-W_true) @ (np.eye(d)-W_true).T  / sigma_0
       Dtrue = 1 + np.random.normal(scale=0.2, size=d)
       Dtrue[Dtrue < 1] = 1
       Theta_true = (np.diag(Dtrue) -W_true@np.diag(Dtrue)) @ (np.eye(d)-W_true).T

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
           cont1 = ithg1
           res1 = pd.DataFrame
           res1 = ithg1.tail(1)
       else:
           cont1 = pd.concat([cont1, ithg1])
           res1 = pd.concat([res1, ithg1.tail(1)])
       cont1.to_csv('%s/iterh_fista-loram.csv' %fdir)
       res1.to_csv('%s/res_fista-loram.csv' %fdir)

       print('\n========== ALM FISTA=============\n')
       # Observation: ALM below is slower than run_icid (single fista) by 2~3 times
       MAXITER_ALM = 20
       MAXITER_PRIMAL = 1e3
       EPSILON = 1e-6
       if deg < 1:
           TAUX_RHO = 1.1
           LAM = 1e0
       else:
           LAM=8e-1
           TAUX_RHO = 1.5
       # Use ALM-exact fit
       A_glo, ithg, optinfo = oicid_solver_alm_exactfit(Theta_true, lambda1=LAM, maxiter=MAXITER_ALM, \
                                            maxiter_primal=MAXITER_PRIMAL, \
                                            epsilon=EPSILON, taux_rho=TAUX_RHO, \
                                            solver_primal = 'FISTA', \
                                            Wtrue=W_true, verbo=1)
       acc = utils.count_accuracy(W_true!=0, A_glo!=0)

       # # Use Nv
       # Bd, ithg, optinfo = oicidNV_solver_alm_exactfit(Theta_true, lambda1=LAM, maxiter=MAXITER_ALM, \
       #                                      maxiter_primal=MAXITER_PRIMAL, \
       #                                      epsilon=EPSILON, taux_rho=TAUX_RHO, \
       #                                      solver_primal = 'FISTA', \
       #                                      Wtrue=W_true, verbo=1)
       # acc = utils.count_accuracy(W_true!=0, Bd[0]!=0)

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
           cont = ithg
           res = pd.DataFrame
           res = ithg.tail(1)
       else:
           cont = pd.concat([cont, ithg])
           res = pd.concat([res, ithg.tail(1)])
       cont.to_csv('%s/iterh_alm-fista.csv' %fdir)
       res.to_csv('%s/res_alm-fista.csv' %fdir)

       if False:
           print('\n======= ALM BFGS ======\n')
           A2, ithg2, optinfo2 = oicid_solver_alm_exactfit(Theta_true, lambda1=LAM, maxiter=10, \
                                                maxiter_primal=1e3, \
                                                epsilon=1e-6,\
                                                solver_primal = 'BFGS', \
                                                Wtrue=W_true)
           acc2 = utils.count_accuracy(W_true!=0, A_glo!=0)
           print(acc2)

       # # Use A-presentation
       # LAM2 = 2e-2
       # A, ithg, optinfo = oicid_solverA_alm(Theta_true, lambda1=LAM, lambdad2=LAM2, maxiter=MAXITER_ALM, \
       #                                      maxiter_primal=MAXITER_PRIMAL, \
       #                                      epsilon=EPSILON, taux_rho=TAUX_RHO, \
       #                                      solver_primal = 'FISTA', \
       #                                      Wtrue=W_true, verbo=1)
       # B = (np.diag(np.diag(A)) - A ) @ np.diag(1/np.diag(A))
       # acc = utils.count_accuracy(W_true!=0, B!=0)

