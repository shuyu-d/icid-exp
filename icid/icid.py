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

import sys
sys.path.append("..")

import numpy as np
from timeit import default_timer as timer
import pandas as pd
from scipy.sparse.linalg import expm

from icid import utils
from icid.Loram import LoramProxDAG
from icid.SparseMatDecomp import SparseMatDecomp
from sklearn.covariance import GraphicalLassoCV, GraphicalLasso
from external.pyquic.py_quic_w import quic



def AMA_independece_decomp(S, k=25, sigma_0=1.0, \
                          lambda_1=1e-1, c0=500, \
                          gamma_2=1.0, beta_2=0.7, \
                          maxiter=20, maxit_2=20, \
                          maxit_prox_inner=500, \
                          tol_h = 1e-12, epsilon=1e-12, \
                          idec_solver = 'fista', \
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
        return M
    def _threshold_hard_rel(B, tol=5e-2):
        c = max(abs(B.ravel()))
        B[abs(B)< tol*c] = 0
        return B
    def _threshold_hard(B, tol=5e-2):
        B[abs(B)< tol] = 0
        return B
    def _stopp_proxdag(A, h_old):
        ht = _comp_h_plain(A) #
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
            print('Iter: %i | f: %.4e | h: %.3e | gap: %.3e | t(sec): %.2e' \
                 % (niter, stat['fval'], stat['hval'], stat['gap'], stat['time']))
            print(acc)
        return stat
    d = S.shape[0]
    iterh = []
    t0 = timer()
    # ---------------------------------------------------------
    pb = SparseMatDecomp(Prec=S, Wtrue=W_true, \
                           lambda1=lambda_1,\
                           invsigma_0=1/sigma_0,\
                           maxiter=30000)
    w0 = pb.initialize_w(option='zeros') #
    tini = timer() - t0
    if idec_solver == 'FISTA':
        # FISTA solver for ic-decomp
        Wt, idh = pb.solver_fista_linesearch(w0, verbo=1)
        At = Wt
        stat = _comp_iterhist(Wt, At, time=tini+idh.iloc[-1]['time'],\
                              subpb_name='ICD_init-FISTA')
        iterh.append(stat.copy())
    elif idec_solver == 'BFGS':
        # -------BFGS alternative
        Wt, idh = pb.solver_bfgs(w0, verbo=1)
        At = Wt
        stat = _comp_iterhist(Wt, At, time=tini+idh.iloc[-1]['time'],\
                              subpb_name='ICD_init-BFGS')
        iterh.append(stat.copy())
    else:
        raise ValueError('solver %s not available' %idec_solver)
    #---------------------------------------------------------
    h_old = np.Inf
    At = Wt
    DOSTOP_EXCEPTION = False
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
            try:
                ith2, x_sol = pba.run_projdag(alpha=gamma_2,\
                                      maxiter=maxit_prox_inner)
                A, Sca = pba.get_adjmatrix_dense(x_sol) # a dense np matrix
                At = A * c0
                t2 += ith2.iloc[-1]['time']

                Att = _threshold_hard(At, tol=2e-1)
                dostop_h, ht = _stopp_proxdag(Att, h_old)
                if dostop_h:
                    break
                else:
                    gamma_2 = gamma_2 * 3
            except (ZeroDivisionError, ValueError, OverflowError):
                print('///// AMA of ICID terminate with exception from LoRAM\n')
                DOSTOP_EXCEPTION = True
                break
        stat = _comp_iterhist(Wt, At, stat=stat, time=t2, niter=t+1,\
                        subpb_name='proxDAG')
        iterh.append(stat.copy())
        dostop_all = _stopp_criteria(At, stat=stat, niter=t+1, time=ti)
        if dostop_all or DOSTOP_EXCEPTION:
            At = _threshold_hard(At, tol=1e-2)
            break
    return At, pd.DataFrame(iterh, columns=iterh[0].keys()), idh


def run_icid(X, lambda_1=1e-1, idec_lambda1=1e-1, \
                sigma_0=1.0, k=25, \
                beta_2 = 0.7, \
                gamma_2=1.0, maxit_prox_inner=500, \
                W_true=None, opt_ic='sk',\
                idec_solver='fista' \
            ):
    def sp_ice_quic(X):
        # --------------- QUIC
        n_samples, d = X.shape
        X = X - np.mean(X, axis=0, keepdims=True)
        print("IC using QUIC.. ")
        emp_cov = np.dot(X.T, X) / n_samples
        XP, WP, optP, cputimeP, iterP, dGapP = quic(S=emp_cov, \
                L=float(lambda_1), mode="default", tol=1e-16, max_iter=100, msg=1)
        return XP # [0,:].reshape([d, d])
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
        print('lam is %.3e' %lambda_1)
        prec_off[abs(prec_off) < lambda_1 * cmax] = 0
        prec_sp = prec_off + np.diag(np.diag(prec_))
        return prec_sp, cov_
    def sp_ic_ideal(X):
        d = X.shape[1]
        return (np.eye(d)-W_true) @ (np.eye(d)-W_true).T  / sigma_0
    def sp_ice_sklearn(X):
        n_samples = X.shape[0]
        # Estimate the covariance and sparse inverse covariance
        X = X - np.mean(X, axis=0, keepdims=True)
        model = GraphicalLasso(alpha=lambda_1)
        try:
            model.fit(X)
            cov_ = model.covariance_
            prec_ = model.precision_
        except FloatingPointError:
            print("Alpha value not working for glasso (sklearn).. ")
            model = []
            prec_, cov_ = sp_ice_naive(X)
        return prec_, cov_, model
    res = []
    if (opt_ic != 'emp_gs') and (not np.isscalar(lambda_1)):
        lambda_1 = lambda_1[0]
    # Inverse covariance estimation
    if opt_ic is 'sk':
        t0 = timer()
        prec_est, cov_est, model = sp_ice_sklearn(X)
        tg = timer() - t0
    elif opt_ic is 'quic':
        # ----QUIC
        t0 = timer()
        prec_est = sp_ice_quic(X)
        tg = timer() - t0
    elif opt_ic is 'emp':
        print('lambda_1 being used is %.2f' %lambda_1)
        t0 = timer()
        prec_est, cov_est = sp_ice_naive(X)
        tg = timer() - t0
    elif opt_ic is 'emp_gs':
        if np.isscalar(lambda_1):
            prec_est, lambda_1, _, _, tg = ice_sparse_empirical(X)
        else:
            prec_est, lambda_1, _, _, tg = ice_sparse_empirical(X, lams=lambda_1)
        print('lambda_1 chosen by emp-gs is %.3e'% lambda_1)
    else:
        # ----ideal
        t0 = timer()
        prec_est = sp_ic_ideal(X)
        tg = timer() - t0
    # ----------
    Prec_input = prec_est
    acc = utils.count_accuracy((W_true)!=0, Prec_input !=0)
    print(acc)
    stats = {'niter': -1,
            'time': tg,
            'subpb_name': 'IDecomp',
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
    wnew, iterh, idh = AMA_independece_decomp(Prec_input, k=k, \
                                    W_true = W_true, sigma_0=sigma_0,\
                                    lambda_1=idec_lambda1, \
                                    beta_2=beta_2, gamma_2=gamma_2, \
                                    maxit_prox_inner=maxit_prox_inner,\
                                    idec_solver=idec_solver,\
                                    epsilon=1e-2)
    iterh['time'] += tg
    res = pd.concat([res, iterh])
    return wnew, res, idh

# Auxilary functions

# Sparse empirical
def ice_sparse_empirical(X, \
        lams = np.linspace(4e-2,1e-1,10)):
    def _comp_ice_stats(prec, cov):
        error_emp = np.linalg.norm(emp_cov - cov, ord="fro")
        nnz = (prec!=0).sum()
        return error_emp, nnz
    def _criterion_trace(prec):
        prec_ = prec + np.diag(9e-1*np.diag(prec))
        f = (prec * emp_cov).sum() - np.log(np.linalg.det(prec_))
        # f = (prec * emp_cov).sum()
        return f
    def _selection(err_emps, fs, nnzs):
        N = len(err_emps) //2
        c1 = np.argsort(err_emps.values)[:N]
        c2 = np.argsort(fs.values)[:N]
        # Find the most sparse one among c2
        nnzvals = nnzs.values[c2]
        isel = np.argsort(nnzs.values[c2])[N//2]
        sel = c2[isel]
        return c1,c2, nnzvals, sel

    def _comp_sparse_ic(X, lambda_1):
        # Sparsify empirical precision matrix
        prec_ = np.linalg.inv(emp_cov)
        prec_off = prec_.copy()
        prec_off = prec_off - np.diag(np.diag(prec_off))
        cmax = max(abs(prec_off.ravel()))
        #
        prec_off[abs(prec_off) < lambda_1 * cmax] = 0
        #
        prec_sp = prec_off + np.diag(np.diag(prec_))
        cov_o = np.linalg.inv(prec_sp)
        return prec_sp, cov_o
    n_samples, d = X.shape
    X = X - np.mean(X, axis=0, keepdims=True)
    emp_cov = np.dot(X.T, X) / n_samples
    j = 0
    results = []
    precs =[]
    for lam in lams:
        j +=1
        name = "Sparse Empirical %d" % j
        start_time = timer()
        prec, cov = _comp_sparse_ic(emp_cov, lam)
        ctime = timer() - start_time
        err_emp, nnz = _comp_ice_stats(prec, cov)
        f = _criterion_trace(prec)
        precs.append(prec.ravel())
        results.append([name, err_emp, f, nnz, ctime, lam])

    _res=pd.DataFrame(results,
                 columns=[
                    "Estimator",
                    "Error (w Emp Cov)",
                    "Fit function",
                    "nnz (Prec)",
                    "Time",
                    "Lambda"]
                 )
    c1,c2,c3,sel = _selection(
            _res['Error (w Emp Cov)'],\
            _res['Fit function'], \
            _res['nnz (Prec)']
            )
    pd.set_option('display.max_columns', None)
    print("=======Lambda selected vs Argmin SuppDiff(wrt true Prec)=====")
    print(_res.iloc[sel])
    print("============\n")
    return precs[sel].reshape([d,d]), _res['Lambda'][sel], _res, sel, _res['Time'].values.sum()

