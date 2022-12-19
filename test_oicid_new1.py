# Test on Algorithm O-ICID (new1)
import numpy as np

from timeit import default_timer as timer
import time, os
import pandas as pd

from icid import utils
from icid.icid import run_icid, AMA_independece_decomp
from aux.gen_settings import gen_list_settings, gen_list_optparams, gen_data_sem, gen_graph_dag_with_markovblanket

if __name__ == '__main__':
   d = 50
   timestr = time.strftime("%H%M%S%m%d")
   fdir = '../outputs/oicid_on_mb_%s' % timestr
   if not os.path.exists(fdir):
       os.makedirs(fdir)
   degs= [2.0] #
   tid=0
   graph_type, sem_type = 'ER', 'gauss'
   sigma_0 = 1.0
   n = 100 #
   #
   opts={'opt_ic':       ['ideal'], \
       'idec_solver':    ['FISTA'], \
       'k':              [25], \
       'lambda_1':       [5e-2], \
       'idec_lambda1':   [5e-2 ]}  #
   l_o, df_o = gen_list_optparams(opts)
   for deg in degs:
       tid += 1
       spr = deg*d / d**2
       W_true, MBs = gen_graph_dag_with_markovblanket(
                                   d=d, deg=deg,
                                   graph_type=graph_type,
                                   seed=1)
       X = utils.simulate_linear_sem(W_true, n, sem_type)
       # True Theta
       Theta_true = (np.eye(d)-W_true) @ (np.eye(d)-W_true).T  / sigma_0
       # O-ICID global
       W_glo, ithg, idh  = run_icid(X, sigma_0=1.0, \
                             k=df_o['k'][0], \
                             lambda_1=df_o['lambda_1'][0] , \
                             idec_lambda1=df_o['idec_lambda1'][0],\
                             beta_2 = 0.7, gamma_2=1.0, \
                             opt_ic=df_o['opt_ic'][0], \
                             idec_solver = df_o['idec_solver'][0], \
                             maxit_prox_inner=500, W_true=W_true)
       # Initialization for incremental aggregation of local O-ICID solutions
       Wglo_refine = np.double(W_glo > np.Inf) # all zeros
       nvisits_refine = np.double(W_glo > np.Inf) # initial visit count is 0
       fthres = 0.6
       # Solve O-ICID on each Markov blanket
       lens = []
       res = []
       for i in range(d):
           ll =  sum(MBs[i])
           lens.append(ll)
           if ll > 3:
               iarg = i
               MB = list(np.where(MBs[iarg])[0])
               print(MB)
               W_locb = W_true.copy()  # true graph on local MB
               W_locb[~MBs[iarg],:] = 0
               W_locb[:, ~MBs[iarg]] = 0

               #
               Theta_locb = Theta_true.copy()
               Theta_locb[~MBs[iarg],:] = 0
               Theta_locb[:, ~MBs[iarg]] = 0

               # Apply O-ICID on local information (Theta restricted on MB)
               j = 0
               W_icd, ith, idh  = run_icid(X, sigma_0=1.0, \
                                     k=df_o['k'][j], \
                                     lambda_1=df_o['lambda_1'][j] , \
                                     idec_lambda1=df_o['idec_lambda1'][j],\
                                     beta_2 = 0.7, gamma_2=1.0, \
                                     opt_ic=df_o['opt_ic'][j], \
                                     idec_solver = df_o['idec_solver'][j], \
                                     maxit_prox_inner=500, W_true=W_locb)
               Wglo_refine += np.double(W_icd!=0)
               # update submatrix
               subs = np.ix_(MB, MB)
               nvisits_refine[subs] += 1

               # Compute accuracy
               nnz_mb = sum(W_locb.ravel()!=0)
               deg_mb = nnz_mb / ll
               degin_mb = max(np.sum(W_locb!=0, axis=0))
               degout_mb = max(np.sum(W_locb!=0, axis=1))
               degmax_mb = max(degin_mb, degout_mb)

               Wsol_loc = W_glo.copy()
               Wsol_loc[~MBs[iarg],:] = 0
               Wsol_loc[:, ~MBs[iarg]] = 0  # global OICID restricted to local MB
               acc_glo = utils.count_accuracy(W_locb!=0, Wsol_loc!=0)
               acc = utils.count_accuracy(W_locb!=0, W_icd!=0)
               # Record to res
               res.append({'alg':'O-ICID (loc)',
                           'd'           : d, \
                           'deg'         : deg, \
                           'deg_mb'      : deg_mb, \
                           'nnz_mb'      : nnz_mb, \
                           'degin_mb'    : degin_mb, \
                           'degout_mb'   : degout_mb, \
                           'degmax_mb'   : degmax_mb, \
                           'n'           : np.nan, \
                           'graph_type'  : graph_type,\
                           'sem_type'    : sem_type, \
                           'k'           : df_o['k'][0], \
                           'ic'          : df_o['opt_ic'][0], \
                           'id'          : df_o['idec_solver'][0], \
                           'lambda_1'    : df_o['lambda_1'][0], \
                       'idec_lambda1'    : df_o['idec_lambda1'][0], \
                       'shd':acc['shd'], 'tpr':acc['tpr'], \
                       'fdr':acc['fdr'], 'fpr':acc['fpr'], \
                       'nnz':acc['nnz'], 'time': ith.iloc[-1]['time']}
                   )
               res.append({'alg':'O-ICID (glo)',
                           'd'           : d, \
                           'deg'         : deg, \
                           'deg_mb'      : deg_mb, \
                           'nnz_mb'      : nnz_mb, \
                           'degin_mb'    : degin_mb, \
                           'degout_mb'   : degout_mb, \
                           'degmax_mb'   : degmax_mb, \
                           'n'           : np.nan, \
                           'graph_type'  : graph_type,\
                           'sem_type'    : sem_type, \
                           'k'           : df_o['k'][0], \
                           'ic'          : df_o['opt_ic'][0], \
                           'id'          : df_o['idec_solver'][0], \
                           'lambda_1'    : df_o['lambda_1'][0], \
                       'idec_lambda1'    : df_o['idec_lambda1'][0], \
                       'shd':acc_glo['shd'], 'tpr':acc_glo['tpr'], \
                       'fdr':acc_glo['fdr'], 'fpr':acc_glo['fpr'], \
                       'nnz':acc_glo['nnz'], 'time': ithg.iloc[-1]['time']}
                   )
               # edge determination by apperance frequency during refinement
               accg_ref = utils.count_accuracy(W_true!=0, Wglo_refine> fthres*nvisits_refine)
               #
               res.append({'alg':'O-ICID (glo refined)',
                           'd'           : d, \
                           'deg'         : deg, \
                           'deg_mb'      : np.nan, \
                           'nnz_mb'      : -2, \
                           'degin_mb'    : np.nan, \
                           'degout_mb'   : np.nan, \
                           'degmax_mb'   : np.nan, \
                           'n'           : np.nan, \
                           'graph_type'  : graph_type,\
                           'sem_type'    : sem_type, \
                           'k'           : df_o['k'][0], \
                           'ic'          : df_o['opt_ic'][0], \
                           'id'          : df_o['idec_solver'][0], \
                           'lambda_1'    : df_o['lambda_1'][0], \
                       'idec_lambda1'    : df_o['idec_lambda1'][0], \
                       'shd':accg_ref['shd'], 'tpr':accg_ref['tpr'], \
                       'fdr':accg_ref['fdr'], 'fpr':accg_ref['fpr'], \
                       'nnz':accg_ref['nnz'], 'time': ithg.iloc[-1]['time']}
                   )
               pd.DataFrame(res, columns=res[0].keys()).to_csv('%s/res_all.csv' %fdir)
       accg = utils.count_accuracy(W_true!=0, W_glo!=0)
       res.append({'alg':'O-ICID (glo)',
                   'd'           : d, \
                   'deg'         : deg, \
                   'deg_mb'      : np.nan, \
                   'nnz_mb'      : -2, \
                   'degin_mb'    : np.nan, \
                   'degout_mb'   : np.nan, \
                   'degmax_mb'   : np.nan, \
                   'n'           : np.nan, \
                   'graph_type'  : graph_type,\
                   'sem_type'    : sem_type, \
                   'k'           : df_o['k'][0], \
                   'ic'          : df_o['opt_ic'][0], \
                   'id'          : df_o['idec_solver'][0], \
                   'lambda_1'    : df_o['lambda_1'][0], \
               'idec_lambda1'    : df_o['idec_lambda1'][0], \
               'shd':accg['shd'], 'tpr':accg['tpr'], \
               'fdr':accg['fdr'], 'fpr':accg['fpr'], \
               'nnz':accg['nnz'], 'time': ithg.iloc[-1]['time']}
           )
       # edge determination
       accg_ref = utils.count_accuracy(W_true!=0, Wglo_refine> fthres*nvisits_refine)
       res.append({'alg':'O-ICID (glo refined)',
                   'd'           : d, \
                   'deg'         : deg, \
                   'deg_mb'      : np.nan, \
                   'nnz_mb'      : -2, \
                   'degin_mb'    : np.nan, \
                   'degout_mb'   : np.nan, \
                   'degmax_mb'   : np.nan, \
                   'n'           : np.nan, \
                   'graph_type'  : graph_type,\
                   'sem_type'    : sem_type, \
                   'k'           : df_o['k'][0], \
                   'ic'          : df_o['opt_ic'][0], \
                   'id'          : df_o['idec_solver'][0], \
                   'lambda_1'    : df_o['lambda_1'][0], \
               'idec_lambda1'    : df_o['idec_lambda1'][0], \
               'shd':accg_ref['shd'], 'tpr':accg_ref['tpr'], \
               'fdr':accg_ref['fdr'], 'fpr':accg_ref['fpr'], \
               'nnz':accg_ref['nnz'], 'time': ithg.iloc[-1]['time']}
           )
       pd.DataFrame(res, columns=res[0].keys()).to_csv('%s/res_all.csv' %fdir)
       # ----

