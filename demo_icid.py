import numpy as np
from icid import utils
from icid.icid import run_icid


# Generate graph and data
d            = 200
k            = 20
deg          = 1.0 # 0.5
graph_type   = 'ER'
sem_type     = 'gauss'
#
n = 8*d
s0 = int(deg*d)
print('d=%d | #nonzeros=%d | graph type is: %s' \
            %(d,s0,graph_type))

B_true = utils.simulate_dag(d, s0, graph_type)
if graph_type is 'SF':
    # In the case of scale-free (SF) graphs, hubs are mostly
    # causes, rather than effects, of its neighbors
    B_true = B_true.T
W_true = utils.simulate_parameter(B_true)
X = utils.simulate_linear_sem(W_true, n, sem_type)

# Parameters of icid algorithm
lambda_1, idec_lambda1 = 6e-1, 1e-1

# Run ICID
print('-------Ready to run ICD-LoRAM-Altmin---')
print('lambda_1=%.2e | idec-lambda_1=%.2e ' \
            %(lambda_1, idec_lambda1))

W_icd, ith_icd  = run_icid(X, lambda_1=lambda_1,
                          idec_lambda1 = idec_lambda1, \
                          sigma_0=1.0, k=k,
                          beta_2 = 0.7, gamma_2=1.0, \
                          maxit_prox_inner=500, W_true=W_true)

acc = utils.count_accuracy(B_true, W_icd != 0)
print(acc)

# --------------------
# Example settings for deg and lambda_1's:
#
#  (deg: 0.5, lambda_1: 2e-1, idec-lambda_1: 1e-1)
#  (deg: 1.0, lambda_1: 6e-1, idec-lambda_1: 1e-1)
#  (deg: 2, lambda_1: 5e-1, idec-lambda_1: 1e-1)
# --------------------

