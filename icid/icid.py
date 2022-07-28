"""High-dimensional causal discovery from Inverse Covariance matrices by
Independence-based Decomposition (ICID).

This function implements the algorithm of ICID, which consists of the
following two consecutive steps:

    (IC) S* = argmin loss(S) - logdet(S) + lambda_1 |S|_1,
                subject to S > 0 (symmetric positive definite),

    (ID) B* = argmin |S* - phi(B)|^2 + lambda_1' |B|_1,
                subject to B in DAG and supp(B) \subset supp(S).

For the ID problem, phi(B) = (1/s)(I-B)(I-B)' is a quadratic matrix function of B.

Contact: shuyu.dong@inria.fr
"""


import numpy as np
from timeit import default_timer as timer
import pandas as pd
from scipy.sparse.linalg import expm

from icid import utils
from icid.Loram import LoramProxDAG
from icid.SparseMatDecomp import SparseMatDecomp
from sklearn.covariance import GraphLassoCV, GraphLasso



def AMA_independece_decomp(S, k=25, sigma_0=1.0, \
                          lambda_1=1e-1, c0=500, \
                          gamma_2=1.0, beta_2=0.7, \
                          maxiter=20, maxit_2=20, \
                          maxit_prox_inner=500, \
                          tol_h = 1e-12, epsilon=1e-12, \
                          W_true=None, fdir=None, \
                          tid=None, fname=None):
    """
    Input parameters:
    -------------
     S                 : input precision matrix (inverse covariance)
     k (int)           : number of latent dimensions
     sigma0 (2darray)  :

    Returns:
    ---------------
    z :     Point in the product input space of LoRAM
    Zsol :  DAG candidate matrix via the mapping LoRAM(z)
    """
    def _comp_h_plain(B):
        return np.trace(expm(abs(B))) - d
    def _average_explore(W, A, tau=0.7):
        # Average of W and A
        M = (1-tau)* W + tau*A
        # # Sparsify M and renormalize it
        # frobn = np.linalg.norm(M)
        # M = _threshold_hard(M, tol=5e-2)
        # frobt = np.linalg.norm(M)
        # M *= frobn / frobt
        return M
    def _threshold_hard(B, tol=5e-2):
        c = max(abs(B.ravel()))
        B[abs(B)< tol*c] = 0
        return B
    def _stopp_proxdag(A, h_old):
        # B = _threshold_hard(A, tol=5e-2)
        ht = _comp_h_plain(A) #
        # return ((ht <= beta_2 * h_old) or (ht <= tol_h)), ht
        return ((ht <= beta_2 * h_old) ), ht
    def _stopp_criteria(B, stat=None, time=0, niter=0):
        val = False
        if stat is not None:
            if (stat['hval'] < tol_h) or (stat['gap'] < epsilon):
                print('-----AltMin Stopping criteria attained!')
                val = True
        return val
    def _comp_grad(B):
        phib = np.eye(d) - B - B.T + B @ B.T #
        gradf = (2/sigma_0) * \
                (S - phib/sigma_0) @ (np.eye(d)-B)
        return gradf
    def _comp_loss_grad(B):
        phib = np.eye(d) - B - B.T + B @ B.T #
        res_vec = (S - phib/sigma_0).ravel()
        fval = 0.5 * (res_vec **2).sum()
        gradf = (2/sigma_0) * \
                (S - phib/sigma_0) @ (np.eye(d)-B)
        return fval, gradf
    def _comp_iterhist(W, A, stat=None, time=0, Binfo=None, \
                    subpb_name=None, niter=0, \
                    print_period=50, verbo=2):
        if stat is None:
            stat = {'niter': 0,
                    'time': 0,
                    'subpb_name': 'init',
                    'fval': np.nan,\
                    'hval': np.nan,\
                    'F': np.nan,\
                    'gradnorm_F':  np.nan,\
                    'gradnorm_h':  np.nan,\
                    'gradnorm_f':  np.nan,\
                    'optimality': np.nan,\
                    'gap': np.nan, \
                    'nnz': np.nan,\
                    'shd': np.nan, \
                    'tpr': np.nan, \
                    'fdr': np.nan, \
                    'fpr': np.nan \
                    }
        stat['niter'] = niter
        stat['time'] += time
        stat['subpb_name'] = subpb_name
        if subpb_name is 'proxDAG':
            B = A
        else:
            B = W
        # info of f
        fval, gradf = _comp_loss_grad(B)
        stat['fval'], stat['gradnorm_f'] = fval, np.linalg.norm(gradf)
        # info of h
        if verbo > 0:
            stat['hval'] = _comp_h_plain(B) #
        # optimality and gap
        # stat['optimality'] = TODO
        if niter < 1:
            stat['gap'] = np.nan
        else:
            stat['gap'] = np.linalg.norm(W-A)
        # shd
        acc = utils.count_accuracy(W_true!=0, B !=0)
        stat['nnz'], stat['shd'],stat['tpr'], \
        stat['fdr'],stat['fpr'] = \
            acc['nnz'], acc['shd'],acc['tpr'], \
            acc['fdr'],acc['fpr']
        if verbo > 1:
            # print iter information every T iters
            # if niter % print_period == 0:
            print('Iter: %i | f: %.4e | h: %.3e | gap: %.3e | t(sec): %.2e' \
                 % (niter, stat['fval'], stat['hval'], stat['gap'], stat['time']))
            print(acc)
        return stat
    d = S.shape[0]
    t0 = timer()
    # pb = Proximal_GD(Prec=S, Wtrue=W_true, truegraph_istr=0)
    # pb._update_info(lambda1=lambda_1, lambda2=1/sigma_0)
    # TODO: replace 2 lines above by the following
    pb = SparseMatDecomp(Prec=S, Wtrue=W_true, \
                           lambda1=lambda_1,\
                           invsigma_0=1/sigma_0)
    w0 = pb.initialize_w() #
    tini = timer() - t0
    # FISTA solver for ic-decomp
    Wt, iterh = pb.solver_fista_linesearch(w0, verbo=1)
    At = Wt
    stat = _comp_iterhist(Wt, At, time=tini+iterh.iloc[-1]['time'],\
                          subpb_name='ICD_init')
    iterh = []
    iterh.append(stat.copy())
    h_old = np.Inf
    for t in range(maxiter):
        # -- averaging and explore
        t0 = timer()
        Wt = _average_explore(Wt, At)
        ti = timer() - t0
        # ---- measure optimality of Wt, gather stats
        stat = _comp_iterhist(Wt, At, stat=stat, time=ti, niter=t+1,\
                                subpb_name='Interp')
        iterh.append(stat.copy())
        # -- proximal mapping
        h_old = iterh[-1]['hval']
        t2 = 0
        for j in range(maxit_2):
            print('---Fit proximal DAG matrix using LoRAM-AGD (trial %i/%i)...'\
                    % (j+1, maxit_2))
            pba = LoramProxDAG(Wt/c0, k)
            ith2, x_sol = pba.run_projdag(alpha=gamma_2, maxiter=maxit_prox_inner)
            A, Sca = pba.get_adjmatrix_dense(x_sol) # a dense np matrix
            At = A * c0
            t2 += ith2.iloc[-1]['time']
            Att = _threshold_hard(At, tol=1e-2)
            dostop_h, ht = _stopp_proxdag(Att, h_old)
            if dostop_h:
                # h_old = ht
                break
            else:
                gamma_2 = gamma_2 * 5
        stat = _comp_iterhist(Wt, At, stat=stat, time=t2, niter=t+1,\
                        subpb_name='proxDAG')
        iterh.append(stat.copy())
        dostop_all = _stopp_criteria(At, stat=stat, niter=t+1, time=ti)
        if dostop_all:
            At = _threshold_hard(At, tol=3e-2)
            break
    return At, pd.DataFrame(iterh, columns=iterh[0].keys())


def run_icid(X, lambda_1=1e-1, idec_lambda1=1e-1, \
                sigma_0=1.0, k=25, \
                beta_2 = 0.7, tol_prec=1e-1, \
                gamma_2=1.0, maxit_prox_inner=500, \
                W_true=None):
    def sp_ice_naive(X):
        n_samples = X.shape[0]
        # Estimate the covariance and sparse inverse covariance
        X = X - np.mean(X, axis=0, keepdims=True)
        print("IC using naive matrix inversion.. ")
        emp_cov = np.dot(X.T, X) / n_samples
        cov_ = emp_cov
        prec_ = np.linalg.inv(emp_cov)
        prec_off = prec_.copy()
        prec_off = prec_off - np.diag(np.diag(prec_off))
        cmax = max(abs(prec_off.ravel()))
        prec_off[abs(prec_off) < lambda_1 * cmax] = 0
        prec_sp = prec_off + np.diag(np.diag(prec_))
        return prec_sp, cov_
    def sp_ice_sklearn(X):
        n_samples = X.shape[0]
        # Estimate the covariance and sparse inverse covariance
        X = X - np.mean(X, axis=0, keepdims=True)
        model = GraphLasso(alpha=lambda_1)
        try:
            model.fit(X)
            cov_ = model.covariance_
            prec_ = model.precision_
        except FloatingPointError:
            print("Oops!  Alpha value not working for glasso (sklearn).. ")
            model = []
            prec_, cov_ = sp_ice_naive(X)
        return prec_, cov_, model
    res = []
    # Inverse covariance estimation
    t0 = timer()
    prec_est, cov_est, model = sp_ice_sklearn(X)
    # prec_est, cov_est = sp_ice_naive(X, tau=tol_prec)
    tg = timer() - t0
    # -----
    d = X.shape[1]
    P_true  = (np.eye(d)-W_true) @ (np.eye(d)-W_true).T  / sigma_0
    Prec_input = P_true
    # Prec_input = prec_est
    acc = utils.count_accuracy((W_true)!=0, Prec_input !=0)
    print(acc)
    stats = {'niter': -1,
            'time': tg,
            'subpb_name': 'Prec',
            'fval': np.nan,\
            'hval': np.nan,\
            'F': np.nan,\
            'gradnorm_F':  np.nan,\
            'gradnorm_h':  np.nan,\
            'gradnorm_f':  np.nan,\
            'optimality': np.nan,\
            'gap': np.nan, \
            'nnz': acc['nnz'],\
            'shd': acc['shd'], \
            'tpr': acc['tpr'], \
            'fdr': acc['fdr'], \
            'fpr': acc['fpr']}
    res.append(stats.copy())
    res = pd.DataFrame(res, columns=res[0].keys())
    # ICDecomp and Loram-AltMin
    # TODO: lambda_1 below should be 0 instead of 1e-1
    wnew, iterh = AMA_independece_decomp(Prec_input, k=k, \
                                    W_true = W_true, sigma_0=sigma_0,\
                                    lambda_1=idec_lambda1, \
                                    beta_2=beta_2, gamma_2=gamma_2, \
                                    maxit_prox_inner=maxit_prox_inner,\
                                    epsilon=1e-2)
    iterh['time'] += tg
    res = pd.concat([res, iterh])
    return wnew, res

