import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as sigmoid

import matplotlib.pyplot as plt
from timeit import default_timer as timer
import time, os
import pandas as pd
import shutil
from sklearn.utils import resample

from icid import utils
# from loram.mf_projdag import Projdag
from icid.Loram import LoramProxDAG
# from loram.mf_projdag import lin_tspace_norm
from external.test_linear import notears_linear

"""
NOTE (14 feb 2022)
The ground truth matrix Wtrue was wrongly used in the computation of
accuracies: it is its transpose that our dxd matrix variable is approaching!
The code of notears produce the pair of graph and data in transpose form (dxd W
and nxd data) with respect to our model (dxd W and dxn data)!

Turned Wtrue into Wtrue.T !
"""

def thresholding_a(A, thres, option='global', K=5):
    if option is 'global':
        Cmax = max(abs(A.ravel()))
        A[abs(A) <= thres * Cmax] = 0
    elif option is 'colwise':
        for i in range(A.shape[0]):
            cvech   = A[:,i].ravel()
            Clmax = max(abs(cvech.ravel()))
            cvech[abs(cvech)<thres*Clmax] = 0
            A[:,i] = cvech
    elif option is 'knn':
        for i in range(A.shape[0]):
            cvech   = A[i,:].ravel()
            inds    = np.argsort(abs(cvech))[-K:]
            tmps    = cvech[inds]
            cvech[abs(cvech)< 0.3] = 0
            cvech[inds] = tmps
            A[i,:] = cvech
    else:
        raise ValueError('unknown threshold type')
    return A

def amplify_knn(W, Aref, K=5):
    # get setdiff
    sdiff = (W!=0) * (Aref==0)
    Wsub = np.zeros(W.shape)
    Wsub[sdiff] = W[sdiff]
    for i in range(W.shape[0]):
        # cvech   = np.concatenate((W[i,:].ravel(), W[:,i].ravel(), axis=None)
        cvech   = Wsub[:,i].ravel()
        inds    = np.argsort(abs(cvech))[-K:]
        # Wsub[inds,i] *= 2 * (np.sign(Wsub[inds,i]) != 0)
        Wsub[inds,i] *= 2
    W[sdiff] = Wsub[sdiff]
    return W

def _dist2_mask(W, Aref, mask):
    # D = np.zeros(W.shape)
    D = 1e-2 *(W-Aref)
    D[mask] = W[mask] - Aref[mask]
    return (D ** 2).sum(), D
def _dist2(W, Aref):
    D = W - Aref
    return (D ** 2).sum(), D
def _prox(W, Aref, gamma):
    # dist2, D = _dist2(W, Aref)
    mask = (Aref != 0)
    dist2, D = _dist2_mask(W, Aref, mask)
    return 0.5*gamma*dist2, gamma * D

def loss_(W, M, loss_type='l2'):
    """Evaluate value and gradient of loss."""
    ZM = W @ M
    if loss_type == 'l2':
        R = M - ZM
        lossval     = 0.5 / M.shape[1] * (R ** 2).sum()
        G_loss   = - 1.0 / M.shape[1] * R @ M.T
    else:
        raise ValueError('unknown loss type')
    return lossval, G_loss
    # dist2, D = _dist2(W)
    # loss     = 0.5 / M.shape[1] * (R ** 2).sum()  \
    #            + 0.5 * gamma * dist2
    # G_loss   = - 1.0 / M.shape[1] * R @ M.T + gamma * D

def comp_pls_bfgs(X, lambda1, w0=None, \
                    loss_type='l2', max_iter=100, gtol=None):
    """Solve min_W 0.5 |X-X B|^2 + lambda1 ‖B‖_1 // + gamma |W - Aref|

    Args:
        X (np.ndarray): [n, d] sample matrix
        lambda1 (float): l1 penalty parameter
        loss_type (str): l2, logistic, poisson
        max_iter (int): max num of dual ascent steps
        h_tol (float): exit if |h(w_est)| <= htol
        w_threshold (float): drop edge if |weight| < threshold
    Returns:
        W_est (np.ndarray): [d, d] estimated DAG
    """
    def _adj(w):
        """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
        return (w[:d * d] - w[d * d:]).reshape([d, d])
    def _loss(W):
        """Evaluate value and gradient of loss.
            NOTE: to improve the time efficiency, use C=X'X directly instead X
        """
        # M = X @ W
        # C = X.T @ X
        Theta = np.eye(d) - W - W.T + W @ W.T
        if loss_type == 'l2':
            # R = X - M
            # loss = 0.5 / X.shape[0] * (R ** 2).sum()
            # G_loss = - 1.0 / X.shape[0] * X.T @ R
            loss = 0.5 * (C*Theta).sum()
            G_loss = - C @ (np.eye(d) -W)
        # elif loss_type == 'logistic':
        #     loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
        #     G_loss = 1.0 / X.shape[0] * X.T @ (sigmoid(M) - X)
        # elif loss_type == 'poisson':
        #     S = np.exp(M)
        #     loss = 1.0 / X.shape[0] * (S - X * M).sum()
        #     G_loss = 1.0 / X.shape[0] * X.T @ (S - X)
        else:
            raise ValueError('unknown loss type')
        return loss, G_loss
    def _func(w):
        """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
        W = _adj(w)
        loss, G_loss = _loss(W)
        obj = loss + lambda1 * w.sum()
        G_smooth = G_loss
        g_obj = np.concatenate((G_smooth + lambda1, - G_smooth + lambda1),\
                                axis=None)
        return obj, g_obj
    t0 = timer()
    n, d = X.shape
    C = X.T @ X / n
    if w0 is None:
        w_est = np.zeros(2 * d * d)  # double w_est into (w_pos, w_neg)
    else:
        w_est = w0
    rho, alpha, h = 1.0, 0.0, np.inf  # double w_est into (w_pos, w_neg)
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    """ start iterations """
    if gtol is None:
        opts = {'iprint':40,'maxcor':30}
    else:
        opts = {'gtol':gtol, 'iprint':40,'maxcor':30}
    if d < 800:
        opts['iprint'] = -1
    sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, \
            bounds=bnds, options=opts)
    w_new = sol.x
    ti = timer() - t0
    Wnew = _adj(w_new)
    # get some more stats
    _, g_obj = _func(w_new)
    _, G_loss = _loss(Wnew)
    return Wnew, w_new, ti, {'loss_nol1':G_loss,\
                             'loss_wl1':g_obj, 'lambda1':lambda1}

def fit_solver_bfgs(M, lambda1, Aref, gamma=1, w0=None, \
                    loss_type='l2', max_iter=100, gtol=None):
    """Solve min_W L(W; M) + lambda1 ‖W‖_1 + gamma |W - Aref|

    Args:
        M (np.ndarray): [d, n] sample matrix
        lambda1 (float): l1 penalty parameter
        loss_type (str): l2, logistic, poisson
        max_iter (int): max num of dual ascent steps
        h_tol (float): exit if |h(w_est)| <= htol
        w_threshold (float): drop edge if |weight| < threshold

    Returns:
        W_est (np.ndarray): [d, d] estimated DAG
    """
    MMt = M @ (M.T)
    def _adj(w):
        """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
        return (w[:d * d] - w[d * d:]).reshape([d, d])
    def _loss(W, M):
        """Evaluate value and gradient of loss."""
        ZM = W @ M
        if loss_type == 'l2':
            loss     = 0.5 / M.shape[1] * ((M-ZM) ** 2).sum()
            ZMMt = W @ MMt
            G_loss   = - 1.0 / M.shape[1] * (MMt - ZMMt)
        elif loss_type == 'log-l2':
            lsq = 0.5 / M.shape[1] * ((M-ZM) ** 2).sum()
            loss     = np.log(lsq)
            ZMMt = W @ MMt
            G_loss   = (ZMMt - MMt) / (M.shape[1] * lsq)
        else:
            raise ValueError('unknown loss type')
        return loss, G_loss
    def _func(w):
        """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
        W = _adj(w)
        loss, G_loss = _loss(W, M)
        loss_prox, G_prox = _prox(W, Aref, gamma)
        obj = loss + lambda1 * w.sum() + loss_prox
        G_smooth = G_loss + G_prox
        g_obj = np.concatenate((G_smooth + lambda1, - G_smooth + lambda1),\
                                axis=None)
        return obj, g_obj
    t0 = timer()
    d, n = M.shape
    if w0 is None:
        w_est = np.zeros(2 * d * d)  # double w_est into (w_pos, w_neg)
    else:
        w_est = w0
    rho, alpha, h = 1.0, 0.0, np.inf  # double w_est into (w_pos, w_neg)
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    """ start iterations """
    # sol = sopt.minimize(_func, w_est, method='Newton-CG', jac=True, \
    #          options={'disp': True})
    if gtol is None:
        opts = {'iprint':40,'maxcor':30}
    else:
        opts = {'gtol':gtol, 'iprint':40,'maxcor':30}
    if d < 800:
        opts['iprint'] = -1
    sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, \
            bounds=bnds, options=opts)
                          # bounds=bnds)
                          # bounds=bnds, options={'maxiter': 200})
    w_new = sol.x
    ti = timer() - t0
    Wnew = _adj(w_new)
    # get some more stats
    _, g_obj = _func(w_new)
    _, G_loss = _loss(Wnew, M)
    _, G_prox = _prox(Wnew, Aref, gamma)
    return Wnew, w_new, ti, {'loss_nol1':G_loss,'dist_loram':G_prox,'loss_wl1':g_obj, 'lambda1':lambda1, 'gamma': gamma}

def _prox_l2(W, A):
    return ((W-A) ** 2).sum()
def _loss_l2(W, M):
    return 0.5 / M.shape[1] * ((M-W@M) ** 2).sum()

def get_iterate_a1(M, lambda1, Wtrue=None, loss_type='l2'):
    """Run once the data fitting solver
    """
    t0 = timer()
    d, n = M.shape
    Aref = np.zeros([d,d])    # initial reference matrix
    if loss_type == 'l2':
        M = M - np.mean(M, axis=1, keepdims=True)
    gap = np.Inf
    gap_old = gap
    ti = timer() - t0
    """ initialize """
    gamma_1 = 1 # should try 0 to see if this works fine
    thres = 1e-10
    print('---Fit proximal linear SEM by L-BFGS...')
    Wt, wt_, ti, grads = fit_solver_bfgs(M, lambda1, Aref, gamma=gamma_1,\
                                  Wtrue=Wtrue)
    Cmax = max(abs(Wt.ravel()))
    Wpost = Wt
    Wpost[abs(Wpost) <= thres * Cmax] = 0
    return Wpost

def select_lambda1(M, list_values=None, sp_target=3e-3, loss_type='l2'):
    if list_values is None:
        list_values = np.linspace(0.1, 0.8, num=8)
    d = M.shape[0]
    list_values = np.flipud(np.sort(list_values)) # Descending order
    sps = []
    for i in range(len(list_values)):
        print('Now testing the %d-th largest in the list: %f' \
                %(i+1, list_values[i]))
        Wi = get_iterate_a1(M,list_values[i], loss_type=loss_type)
        sps.append(np.count_nonzero(Wi) / d**2)
    ind = np.argmin(abs(np.asarray(sps)-sp_target))
    print('---The value selected is the %i-th largest in the list.'% ind)
    return ind, list_values, sps

def run_bootstrap_pls(M, lambda1, thres=5e-2, W_true=None, T=20, loss_type='l2'):
    print('Conducting bootstrap with Penalized Least Squares')
    d,n = M.shape
    Aref = np.zeros([d,d])    # reference matrix
    accs = []
    W_bag = np.zeros([d,d])    # initial 'ensemble' matrices
    B_bag = np.zeros([d,d])    #
    tt = 0
    Ws = []
    for t in range(T):
        Xv = resample(M.T, replace=True, n_samples=n) # [n_samples, n_features]
        Wt, wt_, ti, grads = fit_solver_bfgs(Xv.T, lambda1, Aref, \
                                gamma=0, loss_type=loss_type)
        print('--new Wt obtained in %.3e seconds--' % ti)
        # Increment the ensemble matrix
        tt += ti
        acc = utils.count_accuracy((W_true.T)!=0, Wt !=0)
        print(acc)
        accs.append(acc.copy())
        # store Wt
        Ws.append(Wt)
    return Ws, tt, pd.DataFrame(accs)

def run_bootstrap_glasso(M, lambda1, thres=5e-2, W_true=None, T=20, loss_type='l2'):
    print('Conducting bootstrap with Graphical LASSO')
    d,n = M.shape
    Aref = np.zeros([d,d])    # reference matrix
    accs = []
    W_bag = np.zeros([d,d])    # initial 'ensemble' matrices
    B_bag = np.zeros([d,d])    #
    tt = 0
    for t in range(T):
        Xv = resample(M.T, replace=True, n_samples=n) # [n_samples, n_features]
        Wt, wt_, ti, grads = fit_solver_bfgs(Xv.T, lambda1, Aref, \
                                gamma=0, loss_type=loss_type)
        print('--new Wt obtained in %.3e seconds--' % ti)
        # Increment the ensemble matrix
        B_bag += (Wt != 0)
        W_bag += Wt
        tt += ti
        acc = utils.count_accuracy((W_true.T)!=0, Wt !=0)
        print(acc)
        accs.append(acc.copy())
    B_bag *= 1/T
    B_bag[B_bag < 0.5] = 0
    W_bag *= 1/T
    # W_bag[B_bag == 0] = 0
    W_bag[abs(W_bag) < thres] = 0
    return W_bag, B_bag, wt_, tt, grads, pd.DataFrame(accs)

def run_altmin_bootstrap(M, lambda1, k=40, beta_1=0.1, beta_2=0.3, \
               Wtrue=None, loss_type='l2', \
               c0=20, alpha=5e0, max_iter=20, \
               maxit_1=20, maxit_2=20, tol_h=1e-3, \
               eps_d=1e-3, thres=1e-2, gamma_1=0, \
               gamma_2=1):
    """Solve min_W L(W; M) + lambda1 ‖W‖_1 s.t. W in DAG using LoRAM-based
            DAG-projection
    Args:
        M (np.ndarray):         [d, n] sample matrix
        lambda1 (float):        l1 penalty parameter
        loss_type (str):        l2, logistic, poisson
        max_iter (int):         max num of dual ascent steps
        h_tol (float):          exit if |h(w_est)| <= htol
        w_threshold (float):    drop edge if |weight| < threshold
    Returns:
        W_est (np.ndarray): [d, d] estimated DAG
    """
    def _comp_stats(W, A, ti, gap=None, h=None, dh=None, gradf=None,
            stats=None, sub=None, verbo=None):
        if stats is None:
            stats = {'sub': sub, 'time':0, 'loss':0, 'relerr':np.nan, \
                     'err':np.nan, 'splevel':np.nan,'gap':np.nan,\
                     'hval':np.nan, 'crit_primal':np.nan, \
                     'crit_primal_tr':np.nan,\
                     'gradh_norm': np.nan, 'loss_l1':np.nan,\
                     'fdr':np.nan, 'tpr':np.nan,'fpr':np.nan,\
                     'shd':np.nan}
        stats['time']    = ti + stats['time']
        stats['sub']     = sub
        if verbo == 2:
            """ Compute iterate stats """

            d = W.shape[0]
            if sub is 'a' or sub is '0':
                mat = A
            else:
                mat = W
            if gap is None:
                dist2 = _prox_l2(W,A)
                stats['gap']  = np.sqrt(dist2)
            else:
                stats['gap']  = gap
            if gradf is not None:
                At = Wtrue
                _, G_loss = loss_(At, M)
                v_1 = G_loss + gradf['lambda1'] * np.sign(At)
                v_1[At==0] = 0 # Retain only the residual v_1 on nonzero entries of A
                v_2 = abs(G_loss) - gradf['lambda1']
                # Retain only the residual v_2 on zeros of A and where |gradf| depasses
                # the threshold lambda1
                v_2[At!=0] = 0
                v_2[v_2<0] = 0
                stats['crit_primal_tr'] = np.sqrt(sum(v_1.ravel()**2) + \
                                        sum(v_2.ravel()**2)) / d
                At = W
                _, G_loss = loss_(At, M)
                _, G_prox = _prox(At, A, gradf['gamma'])
                v_1 = G_loss + G_prox + gradf['lambda1'] * np.sign(At)
                v_1[At==0] = 0 # Retain only the residual v_1 on nonzero entries of A

                v_2 = abs(G_loss+ G_prox) - gradf['lambda1']
                # Retain only the residual v_2 on zeros of A and where |gradf| depasses
                # the threshold lambda1
                v_2[At!=0] = 0
                v_2[v_2<0] = 0
                stats['crit_primal'] = np.sqrt(sum(v_1.ravel()**2) + \
                                        sum(v_2.ravel()**2)) / d
            if dh is not None:
                stats['gradh_norm'] = dh
            if h is None:
                stats['hval'] = np.trace(slin.expm(abs(mat))) - d  # attention: exp trace here, very costly!
            else:
                stats['hval'] = h
            loss  = _loss_l2(mat, M)
            stats['loss']       = loss
            stats['loss_l1']    = loss + lambda1 * abs(mat).sum()
            stats['norm']    = np.linalg.norm(mat)
            stats['relerr']  = np.linalg.norm(mat-Wtrue) / np.linalg.norm(Wtrue)
            stats['err']     = np.linalg.norm(mat-Wtrue)
            stats['splevel'] = np.count_nonzero(mat)
            acc = utils.count_accuracy(Wtrue!=0, mat!=0)
            print(acc)
            stats['fdr'] = acc['fdr']
            stats['tpr'] = acc['tpr']
            stats['fpr'] = acc['fpr']
            stats['shd'] = acc['shd']
        return stats, {'W': W, 'A': A}
    t0 = timer()
    d, n = M.shape
    wold = np.zeros(2 * d * d)   # double w_est into (w_pos, w_neg)
    Aref = np.zeros([d,d])    # initial reference matrix
    if loss_type == 'l2':
        M = M - np.mean(M, axis=1, keepdims=True)
    """ initialize """
    ith_all = []
    ith_mats = []
    ht = np.Inf; gap = np.Inf
    h_old = ht; gap_old = gap
    stats = None
    ti = timer() - t0
    stats, mats = _comp_stats(Aref, Aref, ti, sub='0')
    ith_all.append(stats.copy())
    ith_mats.append(mats.copy())
    """ Start iterations """
    print('Iter |  f value | h value | dist(A,LoRAM) ')
    for it in range(max_iter):
        if (gap < d* eps_d):
            print('=====> AltMin terminates with success.')
            break
        t1 = 0
        for ii in range(maxit_1):
            print('---Fit proximal linear SEM by L-BFGS (trial %i/%i)...'\
                    %(ii+1, maxit_1))
            if it == 0 and d>=800:
                gtol = 3.5e-1
            else:
                gtol = None
            if it == 0:
                Wt, Bt, wt_, ti, grads, \
                        accs_boot = run_bootstrap_glasso(M, lambda1, \
                                                thres=5e-2, W_true=Wtrue)
            else:
                Wt, wt_, ti, grads = fit_solver_bfgs(M, lambda1, Aref, \
                                gamma=gamma_1,w0=wold, Wtrue=Wtrue, gtol=gtol)
            t1 += ti
            gap = np.sqrt( ((Wt-Aref)**2).sum()  )
            wold = wt_
            if (gap <= beta_1 * gap_old) or (gap <= d*eps_d):
                gap_old = gap
                break
            else:
                gamma_1 = 1 + gamma_1 * 5
        if it == 0:
            Winit = Wt
        # Wt[abs(Wt) <= 0.6*thres ] = 0
        """ Compute iterate stats """
        stats, _ = _comp_stats(Wt, Aref, t1, gap=gap, gradf=grads, stats=stats,
                sub='w', verbo=2)
        ith_all.append(stats.copy())
        print('Iter: %i | f: %.4e | h: %.3e | gap: %.3e | t(sec): %.2e' \
                     % (it, stats['loss'], stats['hval'], stats['gap'], stats['time']))
        """ Projection step """
        t2 = 0
        for j in range(maxit_2):
            print('---Fit proximal DAG matrix using LoRAM-AccGD (trial %i/%i)...'\
                    % (j+1, maxit_2))
            pb   = Projdag(Wt/c0, k, M=M)
            ith2, x_sol = pb.run_projdag(alpha=gamma_2, maxiter=500)
            Aloram, Sca = pb.get_adjmatrix_dense(x_sol) # a dense np matrix
            Aref      = Aloram * c0

            # ht_pre =  np.trace(slin.expm(abs(Aref))) - d # NOTE: disable for large d
            """ hard threshold step """
            Apost = thres_knn(Aref, thres)
            # # relative threshold value
            # Clmax = max(abs(Aref.ravel()))
            # Apost = Aref
            # Apost[abs(Apost) <= thres * Clmax] = 0
            # (not used) absolute threshold value
            # (aborted) idea of varying thres
            """ Descent criterion """
            ht =  np.trace(slin.expm(abs(Apost))) - d # NOTE: disable for large d
            # Compute the gradient norm instead
            # the gradient info is in pb.iterdb['gradh']
            dhtd = ith2.iloc[-1]['gradnorm']
            # print('=========== gradh at thresholded point is (%.2e)' \
            #                 % (dhtd))
            t2 += ith2.iloc[-1]['time']
            if (ht <= beta_2 * h_old) or (ht <= tol_h):
            # if True: #(dhtd <= beta_2 * h_old): #or (dht <= tol_h):
                # h_old = max(ith2.iloc[-200:]['gradnorm'])
                h_old = ht
                Aref = Apost
                break
            else:
                gamma_2 = gamma_2 * 5
            # gamma_2 *= 5.0
        """ Compute and gather double iter stats """
        stats, mats = _comp_stats(Wt, Aref, t2, h=ht,\
                                    dh=dhtd, stats=stats, sub='a',verbo=2)
        ith_all.append(stats.copy())
        ith_mats.append(mats.copy())
        """ Gather global iter stats """
        print('Iter: %i | f: %.4e | h: %.4e | gap: %.3e | t(sec): %.2e' \
              %(it, stats['loss'], stats['hval'], stats['gap'], stats['time']))
    return Wt, Aref, ith_all, ith2, Winit, ith_mats, accs_boot


def run_altmin(M, lambda1, k=40, beta_1=0.1, beta_2=0.3, \
               Wtrue=None, loss_type='l2', \
               c0=20, alpha=5e0, max_iter=20, \
               maxit_1=20, maxit_2=20, miter_2=500, tol_h=1e-3, \
               eps_d=1e-3, thres=1e-2, gamma_1=1, \
               gamma_2=1):
    """Solve min_W L(W; M) + lambda1 ‖W‖_1 s.t. W in DAG using LoRAM-based
            DAG-projection
    Args:
        M (np.ndarray):         [d, n] sample matrix
        lambda1 (float):        l1 penalty parameter
        loss_type (str):        l2, logistic, poisson
        max_iter (int):         max num of dual ascent steps
        h_tol (float):          exit if |h(w_est)| <= htol
        w_threshold (float):    drop edge if |weight| < threshold
    Returns:
        W_est (np.ndarray): [d, d] estimated DAG

    TODO:
        - (12 mar) change line 204 to 'gamma1 = 0' and line 225 to 'gamma1 = 1 + 5*gamma1'
    """
    def _comp_stats(W, A, ti, gap=None, h=None, dh=None, gradf=None,
            stats=None, sub=None, verbo=None):
        if stats is None:
            stats = {'sub': sub, 'time':0, 'loss':0, 'relerr':np.nan, \
                     'err':np.nan, 'splevel':np.nan,'gap':np.nan,\
                     'hval':np.nan, 'crit_primal':np.nan, \
                     'crit_primal_tr':np.nan,\
                     'gradh_norm': np.nan, 'loss_l1':np.nan,\
                     'fdr':np.nan, 'tpr':np.nan,'fpr':np.nan,\
                     'shd':np.nan}
        stats['time']    = ti + stats['time']
        stats['sub']     = sub
        if verbo == 2:
            """ Compute iterate stats """

            d = W.shape[0]
            if sub is 'a' or sub is '0':
                mat = A
            else:
                mat = W
            if gap is None:
                dist2 = _prox_l2(W,A)
                stats['gap']  = np.sqrt(dist2)
            else:
                stats['gap']  = gap
            if gradf is not None:
                At = Wtrue
                _, G_loss = loss_(At, M)
                v_1 = G_loss + gradf['lambda1'] * np.sign(At)
                v_1[At==0] = 0 # Retain only the residual v_1 on nonzero entries of A

                v_2 = abs(G_loss) - gradf['lambda1']
                # Retain only the residual v_2 on zeros of A and where |gradf| depasses
                # the threshold lambda1
                v_2[At!=0] = 0
                v_2[v_2<0] = 0
                stats['crit_primal_tr'] = np.sqrt(sum(v_1.ravel()**2) + \
                                        sum(v_2.ravel()**2)) / d
                At = W
                _, G_loss = loss_(At, M)
                _, G_prox = _prox(At, A, gradf['gamma'])
                v_1 = G_loss + G_prox + gradf['lambda1'] * np.sign(At)
                v_1[At==0] = 0 # Retain only the residual v_1 on nonzero entries of A

                v_2 = abs(G_loss+ G_prox) - gradf['lambda1']
                # Retain only the residual v_2 on zeros of A and where |gradf| depasses
                # the threshold lambda1
                v_2[At!=0] = 0
                v_2[v_2<0] = 0
                stats['crit_primal'] = np.sqrt(sum(v_1.ravel()**2) + \
                                        sum(v_2.ravel()**2)) / d
            if dh is not None:
                stats['gradh_norm'] = dh
            if h is None:
                stats['hval'] = np.trace(slin.expm(abs(mat))) - d  # attention: exp trace here, very costly!
            else:
                stats['hval'] = h
            loss  = _loss_l2(mat, M)
            stats['loss']       = loss
            stats['loss_l1']    = loss + lambda1 * abs(mat).sum()
            stats['norm']    = np.linalg.norm(mat)
            stats['relerr']  = np.linalg.norm(mat-Wtrue) / np.linalg.norm(Wtrue)
            stats['err']     = np.linalg.norm(mat-Wtrue)
            stats['splevel'] = np.count_nonzero(mat)
            acc = utils.count_accuracy(Wtrue!=0, mat!=0)
            print(acc)
            stats['fdr'] = acc['fdr']
            stats['tpr'] = acc['tpr']
            stats['fpr'] = acc['fpr']
            stats['shd'] = acc['shd']
        return stats, {'W': W, 'A': A}
    t0 = timer()
    d, n = M.shape
    wold = np.zeros(2 * d * d)   # double w_est into (w_pos, w_neg)
    Aref = np.zeros([d,d])    # initial reference matrix
    if loss_type == 'l2':
        M = M - np.mean(M, axis=1, keepdims=True)
    """ initialize """
    ith_all = []
    ith_mats = []
    ht = np.Inf; gap = np.Inf
    h_old = ht; gap_old = gap
    stats = None
    ti = timer() - t0
    stats, _ = _comp_stats(Aref, Aref, ti, sub='0')
    ith_all.append(stats.copy())
    """ Start iterations """
    print('Iter |  f value | h value | dist(A,LoRAM) ')
    for it in range(max_iter):
        # if (stats['hval'] < tol_h) and (gap < d*eps_d):
        if (gap < d* eps_d):
            print('=====> AltMin terminates with success.')
            break
        t1 = 0
        for ii in range(maxit_1):
            print('---Fit proximal linear SEM by L-BFGS (trial %i/%i)...'\
                    %(ii+1, maxit_1))
            if it == 0 and d>=800:
                gtol = 3.5e-1
            else:
                gtol = None
            Wt, wt_, ti, grads = fit_solver_bfgs(M, lambda1, Aref, \
                                gamma=gamma_1,w0=wold, \
                                loss_type=loss_type, gtol=gtol)
            t1 += ti
            gap = np.sqrt( ((Wt-Aref)**2).sum()  )
            wold = wt_
            if (gap <= beta_1 * gap_old) or (gap <= d*eps_d):
                gap_old = gap
                break
            else:
                gamma_1 = 1 + gamma_1 * 5
        if it == 0:
            Winit = Wt
        # Wt[abs(Wt) <= 0.6*thres ] = 0
        Wt = amplify_knn(Wt, Aref, K=2)
        """ Compute iterate stats """
        stats, _ = _comp_stats(Wt, Aref, t1, gap=gap, gradf=grads, stats=stats,
                sub='w', verbo=2)
        ith_all.append(stats.copy())
        print('Iter: %i | f: %.4e | h: %.3e | gap: %.3e | t(sec): %.2e' \
                     % (it, stats['loss'], stats['hval'], stats['gap'], stats['time']))
        """ Projection step """
        t2 = 0
        for j in range(maxit_2):
            print('---Fit proximal DAG matrix using LoRAM-AccGD (trial %i/%i)...'\
                    % (j+1, maxit_2))
            # pb   = Projdag(Wt/c0, k )
            pb   = LoramProxDAG(Wt/c0, k )
            ith2, x_sol = pb.run_projdag(alpha=gamma_2, maxiter=miter_2)
            Aloram, Sca = pb.get_adjmatrix_dense(x_sol) # a dense np matrix
            Aref      = Aloram * c0
            """ hard threshold step """
            Apost = thresholding_a(Aref, thres, option='global', K=4)
            # (refactored) relative threshold value
            # (not used) absolute threshold value
            # (aborted) idea of varying thres
            """ Descent criterion """
            ht =  np.trace(slin.expm(abs(Apost))) - d # NOTE: disable for large d
            # Compute the gradient norm instead
            # the gradient info is in pb.iterdb['gradh']
            dhtd = ith2.iloc[-1]['gradnorm']
            # print('=========== gradh at thresholded point is (%.2e)' \
            #                 % (dhtd))
            t2 += ith2.iloc[-1]['time']
            if (ht <= beta_2 * h_old) or (ht <= tol_h):
            # if True: #(dhtd <= beta_2 * h_old): #or (dht <= tol_h):
                # h_old = max(ith2.iloc[-200:]['gradnorm'])
                h_old = ht
                Aref = Apost
                break
            else:
                gamma_2 = gamma_2 * 5
            # gamma_2 *= 5.0
        """ Compute and gather double iter stats """
        stats, mats = _comp_stats(Wt, Aref, t2, h=ht,\
                                    dh=dhtd, stats=stats, sub='a',verbo=2)
        ith_all.append(stats.copy())
        ith_mats.append(mats.copy())
        """ Gather global iter stats """
        print('Iter: %i | f: %.4e | h: %.4e | gap: %.3e | t(sec): %.2e' \
              %(it, stats['loss'], stats['hval'], stats['gap'], stats['time']))
    return Wt, Aref, ith_all, ith2, Winit, ith_mats


if __name__ == '__main__':
    utils.set_random_seed(1)
    n = 800
    d = 200 #100
    k = 60
    deg = 0.6
    spr = deg*d / d**2
    s0, graph_type, sem_type = int(np.ceil(spr*d**2)), 'ER', 'gauss'
    B_true = utils.simulate_dag(d, s0, graph_type)
    W_true = utils.simulate_parameter(B_true)
    X = utils.simulate_linear_sem(W_true, n, sem_type)
    X = X.T            # we take the transpose as input
    B_true = B_true.T  # we take the transpose as the ground truth graph
    W_true = W_true.T
    C0 = 300
    # lambda_1, tolh = 2e-3, 1e-7
    lambda_1, tolh = 2e-1, 1e-7
    print('-------preparation ok, ready to run altmin---')
    W_est, Aref, ith_all, ith2, Winit, stats = \
                                  run_altmin(X, lambda1=lambda_1, k=k, \
                                             c0=C0, Wtrue=W_true,\
                                             tol_h=tolh, eps_d=1e-7,\
                                             thres=1e-2, max_iter=40,\
                                             beta_1=0.2, beta_2=0.4,\
                                             gamma_2=1.0, loss_type='l2',\
                                             miter_2=500)
    accs_boot = None
    # W_est, Aref, ith_all, ith2, Winit, stats, accs_boot = \
    #                           run_altmin_bootstrap(X, lambda1=lambda_1, k=k, \
    #                                          c0=C0, Wtrue=W_true,\
    #                                          tol_h=tolh, eps_d=1e-7,\
    #                                          thres=1e-1, max_iter=40,\
    #                                          beta_1=0.2, beta_2=0.4,\
    #                                          gamma_2=1.0)

    """ Evaluations and record results """
    Cmax = max(abs(W_est.ravel()))
    # W_est[abs(W_est) <= 1e-2 * Cmax] = 0
    W_est[abs(W_est) <= 5e-2 ] = 0
    # assert utils.is_dag(W_est)
    acc = utils.count_accuracy(B_true, W_est != 0)
    print(acc)

    """ RUNNING NOTEARS """
    if False:
        t1 = timer()
        West_no, ith_no = notears_linear(X.T, lambda1=lambda_1, \
                                    h_tol=tolh, loss_type='l2', \
                                    Wtrue=W_true.T)
        t1 = timer() - t1
        acc_no = utils.count_accuracy(B_true.T != 0, West_no != 0)
        print('NOTEARS-----results in %.2e (seconds):'% t1)
        print(acc_no)

    """ save results """
    timestr = time.strftime("%H%M%S%m%d")
    fdir = '../outputs/altm_linearsem_%s' % timestr
    if not os.path.exists(fdir):
        os.makedirs(fdir)
    shutil.copy2('./altmin_linear_sem.py', '%s/altmin_linear_sem.py.txt' % fdir)
    if accs_boot is not None:
        accs_boot.to_csv('%s/accs_boot.csv' % fdir)

    # np.savetxt('%s/M_.csv'% fdir, X, delimiter=',')
    # np.savetxt('%s/Wtrue_.csv' % fdir, W_true, delimiter=',')
    # np.savetxt('%s/West_.csv'% fdir, W_est, delimiter=',')
    res = []
    res.append([acc['fdr'],acc['tpr'],acc['fpr'],acc['shd']])
    df_res = \
    pd.DataFrame(res, columns= ["FDR", "TPR", \
                    "FPR", "SHD"]).to_csv('%s/accuracy_.csv' %fdir)
    df_thd = pd.DataFrame(ith_all).to_csv('%s/ith_all.csv' %fdir)

    plt.spy(W_true, markersize=4)
    plt.savefig('%s/W_true.eps' % fdir, format='eps')
    plt.close()
    plt.spy(Aref, markersize=4)
    plt.savefig('%s/Wsol_2.eps' % fdir, format='eps')
    plt.close()
    plt.spy(W_est, markersize=4)
    plt.savefig('%s/W_sol.eps' % fdir, format='eps')
    plt.close()

    Ce = max(abs(W_est.ravel()))
    Ct = max(abs(W_true.ravel()))
    Ce2 = max(abs(Aref.ravel()))
    W_est = W_est * Ct / Ce
    West2 = Aref * Ct / Ce2
    plt.figure(figsize=(21,3))
    plt.plot(West2.flatten(),'r', linewidth=2,label='W (ours)')
    plt.plot(W_true.flatten(), 'b-', linewidth=2 , alpha=0.4, label='W true')
    plt.legend(loc='center left', bbox_to_anchor= (1.04, 0.5), ncol=1, borderaxespad=0, frameon=False)
    plt.savefig('%s/Wvec2_overlap.pdf' % fdir, format="pdf",transparent=True, bbox_inches = 'tight')
    plt.close()
    plt.figure(figsize=(21,3))
    plt.plot(W_est.flatten(),'r', linewidth=2,label='W (ours)')
    plt.plot(W_true.flatten(), 'b-', linewidth=2 , alpha=0.4, label='W true')
    plt.legend(loc='center left', bbox_to_anchor= (1.04, 0.5), ncol=1, borderaxespad=0, frameon=False)
    plt.savefig('%s/Wvec_overlap.pdf' % fdir, format="pdf",transparent=True, bbox_inches = 'tight')
    plt.close()
    #
    plt.figure(figsize=(21,3))
    plt.plot(Winit.flatten(),'r', linewidth=2,label='Winit (ours)')
    plt.plot(W_true.flatten(), 'b-', linewidth=2 , alpha=0.4, label='W true')
    plt.legend(loc='center left', bbox_to_anchor= (1.04, 0.5), ncol=1, borderaxespad=0, frameon=False)
    plt.savefig('%s/Wini_overlap.pdf' % fdir, format="pdf",transparent=True, bbox_inches = 'tight')

