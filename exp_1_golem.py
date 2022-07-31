import numpy as np

from timeit import default_timer as timer
import time, os
import pandas as pd

from aux.gen_settings import gen_list_settings, gen_list_optparams, gen_data_sem
from icid import utils


if __name__ == '__main__':
    timestr = time.strftime("%H%M%S%m%d")
    FDIR = 'outputs/exp1_golem_%s' % timestr
    if not os.path.exists(FDIR):
        os.makedirs(FDIR)
    # Generate input parameters
    ds            = np.array([100, 200, 400, 600, 800, 1000, 2000, 3000])
    degs          = [0.5, 1.0, 2.0] # 0.5
    graph_types   = ['ER']
    sem_types     = ['gauss']
    #
    pbs_l, pbs = gen_list_settings(d=ds, degs=degs, \
                                 graph_types=graph_types, \
                                 sem_types=sem_types)

    # Parameters of GOLEM
    #   Adjust the values below if needed
    opts={'lambda_1':       [2e-2], \
          'lambda_2':       [5e0]}
    # Return a pandas table of all combinations of (lambda_1, lambda_2)
    # ----------------------------------------------------
    # NOTE: the total number of combinations is
    #   len(opts['lambda_1']) * len(opts['lambda_2']),
    # which is kept to be 1x1 since we will have lengthy tests with
    # large d such as d=3000 .
    # ----------------------------------------------------
    l_o, opts_df = gen_list_optparams(opts)
    print('List of opt parameters to tets are:')
    print(opts_df)

    # Iterate through all problem settings ('pbs')
    res = []
    for i in range(len(pbs)):
        pbs.at[i,'n'] = 5 * pbs.iloc[i]['d']  # number of samples be 5xd
        print(pbs.iloc[i])
        print('Above is input info to be used.')
        Wtrue, X = gen_data_sem(d           = pbs['d'][i], \
                                deg         = pbs['deg'][i], \
                                n           = pbs['n'][i], \
                                graph_type  = pbs['graph_type'][i],\
                                sem_type    = pbs['sem_type'][i], \
                                seed = 1)
        print('Graph and SCM data generated.\n')
        # NOTE: the size of X is n x d, such that the SCM/SEM says:
        #               X^T = Wtrue^T  X^T + Noise.
        # In other words, (X^T, Wtrue) is the pair of matrices that
        # satisfies the SCM/SEM in the literature.

        # Iterate through all optimization parameter configs
        for j in range(len(opts_df)):
            W_SOLUTION, OPTIONAL_ITERHIST, RUNTIME = \
                                    None, None, np.nan
            ### -----------------------------------
            ### TODO: uncomment the following to let GOLEM run
            #
            # W_SOLUTION, OPTIONAL_ITERHIST, RUNTIME = \
            #     <GOLEM>(X, Wtrue,\
            #             param1=opts_df['lambda_1'][j], \
            #             param2=opts_df['lambda_2'][j])
            ### -----------------------------------

            # Save iteration history if there is
            tid='pb%dopt%d' %(i+1,j+1)
            if OPTIONAL_ITERHIST is not None:
                pd.DataFrame(OPTIONAL_ITERHIST).to_csv('%s/%s_ith_golem.csv' %(FDIR, tid))
            # Compute accuracy of the solution
            # -------------------------------
            # NOTE: check that W_solution should have the same
            # column-row order as Wtrue.
            # -------------------------------
            if W_SOLUTION is None:
                acc = {'shd':np.nan, 'tpr':np.nan, 'fdr':np.nan,\
                       'fpr':np.nan, 'nnz':np.nan}
            else:
                acc = utils.count_accuracy(Wtrue!=0, W_SOLUTION !=0)
            # Record results
            res.append({'alg':'GOLEM',
                        'd'           : pbs['d'][i], \
                        'deg'         : pbs['deg'][i], \
                        'n'           : pbs['n'][i], \
                        'graph_type'  : pbs['graph_type'][i],\
                        'sem_type'    : pbs['sem_type'][i], \
                        'lambda_1'    : opts_df['lambda_1'][j], \
                        'lambda_2'    : opts_df['lambda_2'][j], \
                    'shd':acc['shd'], 'tpr':acc['tpr'], \
                    'fdr':acc['fdr'], 'fpr':acc['fpr'], \
                    'nnz':acc['nnz'], 'time': RUNTIME}
                )
    # Save results to file
    pd.DataFrame(res, columns=res[0].keys()).to_csv('%s/res_all.csv' %FDIR)



