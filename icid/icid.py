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
from sklearn.covariance import GraphicalLassoCV, GraphicalLasso
from external.pyquic.py_quic_w import quic

from icid import utils
from icid.Loram import LoramProxDAG
from icid.SparseMatDecomp import SparseMatDecomp,  SpMatDecomp_local
from aux.gen_settings import get_markovblanket



def oicid_solver_alm(Prec, lambda1=0, maxiter=1e5, \
                     solver_primal='fista',\
                     epsilon=1e-10,
                     Wtrue=None):
    """
        Augmented Lagrangian Method (ALM)
    """
    # initialization
    rho = 0
    Delta = np.zeros(Prec.shape) # zeros
    augl2 = np.Inf
    max_rho = 1e10
    for _ in range(maxiter):
        # Primal descent
        while rho < max_rho:
            pb = spMatDecomp(Prec=Prec, Delta=Delta, rho=rho, Wtrue=Wtrue, maxiter=3e5)
            w0 = pb.initialize_w('zeros') #
            tini = timer() - t0
            if solver_primal == 'FISTA':
                # todo: add 3rd output for fista solver
                Wt, idh, aux = pb.solver_fista_linesearch(w0, verbo=1)
                At = Wt
                stat = _comp_iterhist(Wt, At, time=tini+idh.iloc[-1]['time'],\
                                      subpb_name='ICD_init-FISTA')
                iterh.append(stat.copy())
            elif solver_primal == 'BFGS':
                # todo: add 3rd output for solver
                Wt, idh, aux = pb.solver_bfgs(w0, verbo=1)
                At = Wt
                stat = _comp_iterhist(Wt, At, time=tini+idh.iloc[-1]['time'],\
                                      subpb_name='ICD_init-BFGS')
                iterh.append(stat.copy())
            else:
                raise ValueError('solver %s not available' %idec_solver)
            if idh['augl2'] > 0.25 * augl2:
                rho *= 5
            else:
                break
        # Dual ascent
        res_c = aux['res_c']
        Delta += rho * res_c
        # Stopping critera
        augl2 = aux['augl2']
        if (augl2 < epsilon and aux['gradnorm'] < 1e-4) or rho >= max_rho:
            break
    return (Wt, pd.DataFrame(iterh, columns=iterh[0].keys()),
            pd.DataFrame(idh, columns=idh[0].keys()) )

def AMA_independece_decomp(S, k=25, sigma_0=1.0, \
                          lambda_1=1e-1, c0=500, \
                          gamma_2=1.0, beta_2=0.7, \
                          maxiter=20, maxit_2=20, \
                          maxit_prox_inner=500, \
                          tol_h = 1e-12, epsilon=1e-12, \
                          idec_solver = 'fista', \
                          fista_initop = 'zeros', \
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
    def _threshold_hard_rel(B, tol=5e-2):
        c = max(abs(B.ravel()))
        B[abs(B)< tol*c] = 0
        return B
    def _threshold_hard(B, tol=5e-2):
        B[abs(B)< tol] = 0
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
    iterh = []
    t0 = timer()
    # ---------------------------------------------------------
    pb = SparseMatDecomp(Prec=S, Wtrue=W_true, \
                           lambda1=lambda_1,\
                           invsigma_0=1/sigma_0,\
                           maxiter=30000)
    # w0 = pb.initialize_w(option='zeros') #
    w0 = pb.initialize_w(option=fista_initop) #
    tini = timer() - t0
    if idec_solver == 'FISTA':
        # FISTA solver for ic-decomp
        Wt, idh = pb.solver_fista_linesearch(w0, verbo=1)
        At = Wt
        stat = _comp_iterhist(Wt, At, time=tini+idh.iloc[-1]['time'],\
                              subpb_name='ICD_init-FISTA')
        iterh.append(stat.copy())
    # Wt = _threshold_hard(Wt, tol=5e-2)
    # stat = _comp_iterhist(Wt, At, time=tini+idh.iloc[-1]['time'],\
    #                       subpb_name='ICD_init-thres')
    # iterh.append(stat.copy())
    elif idec_solver == 'BFGS':
        # -------BFGS alternative
        Wt, idh = pb.solver_bfgs(w0, verbo=1)
        At = Wt
        stat = _comp_iterhist(Wt, At, time=tini+idh.iloc[-1]['time'],\
                              subpb_name='ICD_init-BFGS')
        iterh.append(stat.copy())
    else:
        raise ValueError('solver %s not available' %idec_solver)
    # Wt = _threshold_hard(Wt, tol=5e-2)
    # stat = _comp_iterhist(Wt, At, time=tini+idh.iloc[-1]['time'],\
    #                       subpb_name='ICD_init-bfgs-thres')
    # iterh.append(stat.copy())
    #---------------------------------------------------------
    W0 = Wt.copy()
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

                # Att = _threshold_hard(At, tol=5e-2)
                # Att = _threshold_hard_rel(At, tol=2e-1)
                Att = _threshold_hard(At, tol=2e-1)
                dostop_h, ht = _stopp_proxdag(Att, h_old)
                if dostop_h:
                    # h_old = ht
                    # At = Att
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
    # return At, pd.DataFrame(iterh, columns=iterh[0].keys()), idh
    return At, pd.DataFrame(iterh, columns=iterh[0].keys()), idh, W0

def run_icid(X, lambda_1=1e-1, idec_lambda1=1e-1, \
                sigma_0=1.0, k=25, \
                beta_2 = 0.7, \
                gamma_2=1.0, maxit_prox_inner=500, \
                maxit_ama=15,\
                Theta=None, \
                W_true=None, opt_ic='sk',\
                fista_initop = 'zeros', \
                idec_solver='fista' \
            ):
    def sp_ice_quic(X):
        # --------------- QUIC
        n_samples, d = X.shape
        X = X - np.mean(X, axis=0, keepdims=True)
        print("IC using QUIC.. ")
        emp_cov = np.dot(X.T, X) / n_samples
        #       Run in "path" mode
        #       path = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5 ])
        # path = np.linspace(1.0, 0.1, 4)
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
        # model = GraphicalLassoCV(alphas=list(np.logspace(np.log10(8e-2), np.log10(1e0),10))).fit(X)
        # model = GraphicalLassoCV(alphas=list(np.linspace(5e-1, 3e0,10)))
        # model = GraphicalLassoCV().fit(X)
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
    elif opt_ic is 'ideal':
        # ----ideal
        t0 = timer()
        prec_est = sp_ic_ideal(X)
        tg = timer() - t0
    else:
        # ----some input Theta
        t0 = timer()
        prec_est = Theta
        tg = timer() - t0
    # ----naive inversion
    # prec_est, cov_ = sp_ice_naive(X)
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
    # wnew, iterh, idh = AMA_independece_decomp(Prec_input, k=k, \
    wnew, iterh, idh, w0 = AMA_independece_decomp(Prec_input, k=k, \
                                    W_true = W_true, sigma_0=sigma_0,\
                                    lambda_1=idec_lambda1, \
                                    beta_2=beta_2, gamma_2=gamma_2, \
                                    maxiter=maxit_ama, \
                                    maxit_prox_inner=maxit_prox_inner,\
                                    idec_solver=idec_solver,\
                                    epsilon=1e-2)
    iterh['time'] += tg
    res = pd.concat([res, iterh])
    # return wnew, res, idh
    return (wnew, res, idh, w0)

def run_oicid_loc2(Prec, W_true, iarg=None, Si=None, \
                Winit=None, \
                lambda_1=1e-1, idec_lambda1=1e-1, \
                sigma_0=1.0, k=25, \
                c0=500, beta_2 = 0.7, \
                gamma_2=1.0, maxit_prox_inner=500, \
                opt_ic='sk', tol_h = 1e-12,\
                idec_solver='fista' \
                ):
    """
        iarg:   Index of the central node of the local o-icid problem
        Si:     Boolean array of length d indicating the neighors of iarg
    """
    if Si is None:
        iarg = 1
        Si = get_markovblanket(W_true, node_index=[iarg])[0]
    res = []
    # ----------
    acc = utils.count_accuracy((W_true)!=0, Prec !=0)
    print('Prec vs Btrue:')
    print(acc)
    stats = {'niter': -1,
            'time': 0,
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
    # -------- Produce local input information from Si
    cmask = np.zeros(Prec.shape)
    cmask[iarg,:] = 1
    cmask[:,iarg] = 1
    # Record the support of Prec without self-loops
    tmp = Prec.copy()
    tmp          = tmp - np.diag(np.diag(tmp))
    tmp = np.double(tmp!=0) # off-diagonals of Prec
    Bmask = cmask.copy() * tmp.copy()
    W_tloc = W_true.copy()
    W_tloc[Bmask==0] = 0
    # ---------------------------------------------------------
    t0 = timer()
    # pb = SpMatDecomp_local(Prec_loc, Si, Wtrue=W_tloc, \
    pb = SpMatDecomp_local(Prec, Bmask=Bmask, \
                           Wtrue=W_tloc, \
                           lambda1=idec_lambda1,\
                           invsigma_0=1/sigma_0,\
                           maxiter=30000)
    # w0 = pb.initialize_w(option='zeros') #
    w0 = pb.initialize_w(w0=Winit) #
    tini = timer() - t0
    iterh=[]
    # FISTA solver for ic-decomp
    Wt, idh = pb.solver_fista_linesearch(w0, verbo=1)
    At = Wt
    stat = _comp_iterhist(Wt, At, Prec, W_true=W_tloc, \
                              time=tini+idh.iloc[-1]['time'],\
                               subpb_name='ICD_init-FISTA')
    if False:
        #---------------------------------------------------------
        maxit_2 =10
        W0 = Wt.copy()
        h_old = np.Inf
        At = Wt
        DOSTOP_EXCEPTION = False
        for t in range(20):
            # -- averaging and explore
            t0 = timer()
            Wt = _average_explore(Wt, At)
            ti = timer() - t0
            # ---- measure optimality of Wt, gather stats
            stat = _comp_iterhist(Wt, At, Prec_loc, W_true=W_tloc, \
                                  stat=stat, time=ti, niter=t+1,\
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
                    dostop_h, ht = _stopp_proxdag(Att, h_old, beta_2)
                    if dostop_h:
                        break
                    else:
                        gamma_2 = gamma_2 * 3
                except (ZeroDivisionError, ValueError, OverflowError):
                    print('///// AMA of ICID terminate with exception from LoRAM\n')
                    DOSTOP_EXCEPTION = True
                    break
            stat = _comp_iterhist(Wt, At, Prec_loc, W_true=W_tloc, \
                                 stat=stat, time=t2, niter=t+1,\
                                 subpb_name='proxDAG')
            iterh.append(stat.copy())
            dostop_all = _stopp_criteria(At, tol_h=tol_h, stat=stat, niter=t+1, time=ti)
            if dostop_all or DOSTOP_EXCEPTION:
                At = _threshold_hard(At, tol=1e-2)
                break
    else:
        iterh.append(stat.copy())
    iterh = pd.DataFrame(iterh, columns=iterh[0].keys())
    res = pd.concat([res, iterh])
    return (At, res, idh, pb)


def run_oicid_locx(X, lambda_1=1e-1, idec_lambda1=1e-1, \
                sigma_0=1.0, k=25, \
                beta_2 = 0.7, \
                gamma_2=1.0, maxit_prox_inner=500, \
                W_true=None, opt_ic='sk',\
                idec_solver='fista' \
                ):
    def sp_ic_ideal(X):
        d = X.shape[1]
        return (np.eye(d)-W_true) @ (np.eye(d)-W_true).T  / sigma_0
    def pick_subset(Prec, Ids, iIds_vu, S_vu=None):
        # Pick the most dense MB subset that does not intersect with S_vu by
        #   half
        # INPUT:
        #       S_vu:    Binary indication array of length d
        iarg = iIds_vu + 1
        S_pre = (Prec[Ids[iarg],:] !=0)
        while sum(S_pre * S_vu) >  0.5 * sum(S_pre):
            iarg += 1
            if (iarg < Prec.shape[0]-1):
                stop = False
                S_pre = (Prec[Ids[iarg],:] !=0)
            else:
                stop = True
        S_vu += S_pre
        S_vu = (S_vu != 0)
        return S_pre, iarg, S_vu, stop
    res = []
    # Inverse covariance matrix
    t0 = timer()
    prec_est = sp_ic_ideal(X)
    tg = timer() - t0
    # ----------
    Prec_input = prec_est
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
    # # ICDecomp and Loram-AltMin
    # w0, iterh, idh = independece_decomp(Prec_input, k=k, \
    #                                 W_true = W_true, sigma_0=sigma_0,\
    #                                 lambda_1=idec_lambda1, \
    #                                 beta_2=beta_2, gamma_2=gamma_2, \
    #                                 maxit_prox_inner=maxit_prox_inner,\
    #                                 idec_solver=idec_solver,\
    #                                 epsilon=1e-2)
    # iterh['time'] += tg
    # res = pd.concat([res, iterh])
    # TODO: divide and conquer steps based on w0
    w0, iterh, idh = None, None, None
    ds = np.sum(Prec_input!=0, axis=0)
    Ids =  sorted(range(len(ds)), key=lambda k: ds[k], reverse=True)
    # Ids =  sorted(range(len(ds)), key=lambda k: ds[k])
    print(ds[Ids])
    # Pick one subset S
    S_vu = (Prec_input[0,:] > np.Inf)
    stop = False
    iarg = -1
    d = Prec_input.shape[0]
    SS = np.linspace(0,d-1,d)
    npass = 0
    icenters = []
    while not stop: # while min-frequency < 1
        # S, iIds_vu, S_vu, stop = pick_subset(Prec_input, Ids, iIds_vu, S_vu)
        npass += 1
        iarg += 1
        S_pre = (Prec_input[Ids[iarg],:] !=0)
        while sum(S_pre * S_vu) >  0.6 * sum(S_pre): # fresh enough
        # while max(S_vu) < 3: # max-frequency < freq_max: # fresh enough
            iarg += 1
            if (iarg < Prec_input.shape[0]-1):
                stop = False
                S_pre = (Prec_input[Ids[iarg],:] !=0)
                # print('we try iarg=%d, for Si of size %d' % (iarg, sum(S_pre)))
            else:
                stop = True
                break
            #print(sum(S_pre * S_vu))
            # print(sum(S_pre))
        S_vu += S_pre
        S_vu = (S_vu != 0)  # Extension: Use integer values of S_vu will enable
                            # circulating over [d] for multiple times (epochs)
        print('\n=====Refinement: %d-th pass:' % npass)
        print('we will conduct local Causal Disc on S=MB(%d) (of %d nodes):'
                    %(sum(S_pre), Ids[iarg]))
        icenters.append(Ids[iarg])
        print(SS[S_pre])
        print('Union S covers %d nodes:' % sum(S_vu))
        print(SS[S_vu])
        print('Uncovered nodes are:')
        print(SS[np.invert(S_vu)])
        # Do something with S
        wnew = w0
    return wnew, res, idh, icenters


# Auxilary functions

def _comp_h_plain(B):
    return np.trace(expm(abs(B))) - B.shape[0]
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
def _stopp_proxdag(A, h_old, beta_2):
    ht = _comp_h_plain(A) #
    #
    return ((ht <= beta_2 * h_old) ), ht
def _stopp_criteria(B, stat=None, time=0, niter=0, tol_h=1e-12, epsilon=1e-12):
    val = False
    if stat is not None:
        if (stat['hval'] < tol_h) or (stat['gap'] < epsilon):
            print('-----AltMin Stopping criteria attained!')
            val = True
    return val
def _comp_grad(B, S):
    d = B.shape[0]
    phib = np.eye(d) - B - B.T + B @ B.T #
    gradf = (2/sigma_0) * \
            (S - phib/sigma_0) @ (np.eye(d)-B)
    return gradf
def _comp_loss_grad(B, S, sigma_0):
    d = B.shape[0]
    phib = np.eye(d) - B - B.T + B @ B.T #
    res_vec = (S - phib/sigma_0).ravel()
    fval = 0.5 * (res_vec **2).sum()
    gradf = (2/sigma_0) * \
            (S - phib/sigma_0) @ (np.eye(d)-B)
    return fval, gradf
def _comp_iterhist(W, A, S, stat=None, time=0, Binfo=None, \
                subpb_name=None, niter=0, \
                sigma_0 = 1.0, W_true= None, \
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
    fval, gradf = _comp_loss_grad(B, S, sigma_0)
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
        # print iter information every T iters
        # if niter % print_period == 0:
        print('Iter: %i | f: %.4e | h: %.3e | gap: %.3e | t(sec): %.2e' \
             % (niter, stat['fval'], stat['hval'], stat['gap'], stat['time']))
        print(acc)
    return stat


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
    # emp_cov += np.diag(2e-2*np.diag(emp_cov))
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

