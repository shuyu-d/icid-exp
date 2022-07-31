import sys
sys.path.append("..")
import pandas as pd
import itertools
from icid import utils

def gen_list_settings(graph_types=None, sem_types=None, degs=[1.0], d=[100], n=[200]):
    if graph_types is None:
        graph_types = ['ER']
    if sem_types is None:
        sem_types = ['gauss']

    l_p = list(itertools.product(graph_types, \
                             sem_types, \
                             degs, d, n))
    df_p = pd.DataFrame(l_p, columns=['graph_type',\
                                        'sem_type', \
                                        'deg', 'd','n'])
    return l_p, df_p

def gen_list_optparams(opts=None):
    if opts is None:
        opts={'k': [25], 'lambda_1': [2e-1,4e-1], 'idec_lambda1':[2e-1]}
    nkeys = len(opts.items())
    ll = []
    cols = []
    for key, li in opts.items():
        ll.append(li)
        cols.append(key)
    l_p = list(itertools.product(*ll))
    df_p = pd.DataFrame(l_p, columns=cols)
    return l_p, df_p

def gen_data_sem(d=100,deg=1.0,n=200,graph_type='ER',sem_type='gauss', seed=1):
    utils.set_random_seed(seed)
    s0 = int(deg*d)
    B_true = utils.simulate_dag(d, s0, graph_type)
    if graph_type is 'SF':
        # In the case of scale-free (SF) graphs, hubs are mostly
        # causes, rather than effects, of its neighbors
        B_true = B_true.T
    W_true = utils.simulate_parameter(B_true)
    X = utils.simulate_linear_sem(W_true, n, sem_type)
    return W_true, X

def gen_graph_dag(d=100,deg=1.0,graph_type='ER', seed=1):
    utils.set_random_seed(seed)
    s0 = int(deg*d)
    B_true = utils.simulate_dag(d, s0, graph_type)
    if graph_type is 'SF':
        # In the case of scale-free (SF) graphs, hubs are mostly
        # causes, rather than effects, of its neighbors
        B_true = B_true.T
    W_true = utils.simulate_parameter(B_true)
    return W_true

if __name__ == '__main__':
    l_p, df_p = gen_list_settings(n=[200,400,600,800])

    for v in l_p:
        print(v)
    print(df_p)

