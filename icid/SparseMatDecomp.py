"""Computation of independence-based decomposition (ID) of a sparse
inverse covariance matrix. The ID problem is defined in the form of

minimize |S - phi(B)|^2, such that supp(B) \subset supp(S),

where S is an input matrix that is sparse and symmetric positive definite,
and phi(B) = (1/s)(I-B) (I-B)' is a quadratic matrix function of B.

This is part of the ICID algorithm for high-dimensional causal discovery
from inverse covariance matrices.

Contact: shuyu.dong@inria.fr
"""

import numpy as np
from scipy.linalg import expm
from timeit import default_timer as timer
import time
import pywt
import pandas as pd
import scipy.optimize as sopt

from scipy.sparse import coo_matrix, csc_matrix
from icid import spmaskmatmul
from icid import utils

def _threshold_soft(W, threshold):
    TW = np.abs(W) - threshold
    TW[TW<=0] = 0
    return np.sign(W) * TW
class SparseMatDecomp():
    def __init__(self, Prec, lambda1=0, \
                 loss_type='l2', maxiter=1000,\
                 invsigma_0=1.0, tol=1e-7,\
                 Wtrue=None):
        """
        Inputs
        """
        self.info = {'time': time.strftime("%H%M%S%m%d")}
        self.S       = Prec
        Sv       = Prec.ravel()
        self.Svec = Sv[Sv!=0]
        tmp = Prec.copy()
        # Mask must not contain diagonal
        ind_2d = np.nonzero(tmp)
        self.I = ind_2d[0].astype(np.uintc)
        self.J = ind_2d[1].astype(np.uintc)
        tmp          = tmp - np.diag(np.diag(tmp))
        self.support = (tmp!=0) # off-diagonals of Prec
        self.d       = Prec.shape[0]
        self.Wtrue   = Wtrue
        # Problem and opt hyper parameters
        self.maxiter = maxiter
        self.tol = tol
        self.lambda1 = lambda1
        self.invsigma_0 = invsigma_0
        self.loss_type  = loss_type
        if loss_type is not 'l2':
            raise ValueError('loss %s is not available yet' % loss_type)
        #
        u,s,v = np.linalg.svd(Prec)
        self.lipschitz_loss = max(max(s),1)

    def _loss_sp(self, B):
        """Evaluate value and gradient of loss of
            l(B) = 0.5 * |S  - invsigma_0 * phi(B)|_F^2, where
                 = - tr(S phi(B)) + 0.5*invsigma_0 * |phi(B)|_F^2, where
        phi(B) = (I-B)(I-B)^T
        """
        B[self.support==0] = 0 #
        A = np.eye(self.d) - B
        # phib, _, _ = spmaskmatmul.spmaskmatmul_abs(A, A.T, self.I, self.J)
        # residual = self.Svec - self.invsigma_0*phib
        phib = A @ A.T
        R = self.S - self.invsigma_0*phib
        # R = np.asarray(coo_matrix((residual, (self.I,self.J)),shape=(self.d,self.d)).todense())
        # loss = 0.5 * (residual ** 2).sum()
        loss = 0.5 * np.linalg.norm(R) ** 2
        G_, _, _= spmaskmatmul.spmaskmatmul_abs(R, A.T, self.I, self.J)
        G_ *= 2*self.invsigma_0
        G_loss = np.asarray(coo_matrix((G_, (self.I,self.J)),shape=(self.d,self.d)).todense())
        G_loss = G_loss - np.diag(np.diag(G_loss))
        return loss, G_loss

    def _loss(self, B):
        """Evaluate value and gradient of loss of
            l(B) = 0.5 * |S  - invsigma_0 * phi(B)|_F^2, where
                 = - tr(S phi(B)) + 0.5*invsigma_0 * |phi(B)|_F^2, where
        phi(B) = (I-B)(I-B)^T
        """
        # B[self.support==0] = 0 # TODO: remove this line, since the
                          # gradient is already in the subspace
        quadb = B @ B.T # --- TRY: use splr
        phib = np.eye(self.d) - B - B.T + quadb
        residual = self.S - self.invsigma_0*phib
        loss = 0.5 * np.linalg.norm(residual) ** 2
        G_loss = 2*self.invsigma_0 * residual @ (np.eye(self.d)-B)
        # Project G_loss back onto the subspace of the support graph
        G_loss[self.support==0] = 0
        return loss, G_loss

    def _func(self, W):
        """Evaluate value and gradient of f = loss + l1 """
        loss, G_loss = self._loss(W)
        obj = loss + self.lambda1 * abs(W).sum()
        g_obj = G_loss + self.lambda1 * np.sign(W)
        return obj, g_obj

    def _update_info(self, lambda1=None, invsigma_0=None):
        # !! This function is deprecated since all parameters are constants needs no updating in this algorithm..
        if lambda1 is not None:
            self.lambda1 = lambda1
        if invsigma_0 is not None:
            self.invsigma_0 = invsigma_0

    def initialize_w(self, w0=None, option='zeros'):
        if w0 is not None:
            w = w0
        else:
            if option is 'zeros':
                w = np.zeros([self.d,self.d])
            elif option is 'gaussian':
                w = np.random.normal(size=[self.d,self.d])
                w[self.support==0] = 0
            else:
                raise ValueError('unknown threshold type')
        return w

    def gd_f_linesearch(self, w, s0=1, c1=0.8, beta=0.5, maxevals=10):
        ss = s0
        f0, G_loss = self._loss(w)
        for i in range(maxevals):
            wn = w - ss * G_loss #
            loss, G_t = self._loss(wn)
            if loss > f0 - c1 * ss* (G_loss * G_loss).sum():
                # print('backtracking now ...')
                ss *= beta
            else:
                break
        if (i+1) == maxevals:
            print('linesearch failed')
        wn = wn - np.diag(np.diag(wn))
        return wn, ss, loss, G_t, {'nevals':i+1, 'is_succ': (i+1<maxevals)}

    def solver_ista_linesearch(self, w0, ls_c1=0.8, ls_beta=0.5):
        w  = w0
        t1 = 0
        stats, _ = self._iter_comp_stats(w, t1)
        iterh    = []
        iterh.append(stats.copy())
        fold = stats['fobj']
        ss = self.lipschitz_loss / 100
        for i in range(3000):
            ti = timer()
            wp, ss, loss, Gt, lsinfo = self.gd_f_linesearch(w,s0=ss, c1=ls_c1, beta=ls_beta)
            w = pywt.threshold(wp, ss*self.lambda1, 'soft')
            t1 += timer() - ti

            stats, _ = self._iter_comp_stats(w, t1, stepsize=ss,ls_info=lsinfo)
            iterh.append(stats.copy())
            fval = stats['fobj']
            if i % 50 == 1:
                print('iter: %d | f: %.7e | loss: %.2e | ss: %.2e | nnz: %d | sp: %.2e | gloss: %.2e | optima: %.2e | time: %.2e' %(i+1, stats['fobj'], stats['loss'], stats['stepsize'], stats['nnz'], stats['sp'], stats['gradnorm'], stats['optimality'], stats['time']) )
            if abs(fold-fval) / abs(fold) < 1e-7:
                break
            fold = fval
        return w, pd.DataFrame(iterh, columns=iterh[0].keys())

    def solver_fista_linesearch(self, w0, ls_c1=0.8, ls_beta=0.5, verbo=2):
        y = w0
        x = y
        stats, _ = self._iter_comp_stats(x, 0)
        iterh    = []
        iterh.append(stats.copy())
        fold = stats['fobj']
        # ss = self.lipschitz_loss / 100
        ss = 1.0
        # Optional
        if verbo > 1:
            loss_0, _ = self._loss(self.Wtrue)
            f_0, g_0  = self._func(self.Wtrue)
            print('f(Wtrue) = %.7e | loss: %.4e' %(f_0,loss_0))
        # FISTA iterations
        for i in range(self.maxiter):
            t0 = timer()
            xold = x
            yplus, ss, loss, Gt, lsinfo = self.gd_f_linesearch(y, s0=ss, c1=ls_c1, beta=ls_beta)
            # yplus, ss, loss, Gt, lsinfo = self.gd_f_linesearch(y, s0=1.0, c1=ls_c1, beta=ls_beta)
            # x = pywt.threshold(yplus, ss*self.lambda1, 'soft')
            x = _threshold_soft(yplus, ss*self.lambda1)
            y = x + i * (x - xold) / (i+3)
            ti = timer() - t0
            stats, _ = self._iter_comp_stats(x, ti, \
                            stats=stats, stepsize=ss,\
                            ls_info=lsinfo, verbo=1)
            iterh.append(stats.copy())
            fval = stats['fobj']
            # Print iteration status
            if (verbo==1 and i % 200 == 0) or (verbo>1):
                pstats, _ = self._iter_comp_stats(x, 0, \
                            stats=stats, stepsize=ss,\
                            ls_info=lsinfo, verbo=2)
                msg_line = 'iter: %d | f: %.7e | loss: %.2e | ss: %.2e | nnz: %d | nnz-g: %d | sp: %.2e | gloss: %.2e | optima: %.2e | time: %.2e' \
                        %(i+1, pstats['fobj'], pstats['loss'], pstats['stepsize'], pstats['nnz'], pstats['nnz_g'], pstats['sp'], pstats['gradnorm'], pstats['optimality'], pstats['time'])
                print(msg_line)
            #
            # Stopping criterion
            if abs(fold-fval) / abs(fold) < self.tol and \
                    stats['optimality'] < 1e-4:
                break
            fold = fval
        return x, pd.DataFrame(iterh, columns=iterh[0].keys())

    def solver_bfgs(self, w0, ls_c1=0.8, ls_beta=0.5, verbo=2):
        def _adj(w):
            """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
            return (w[:self.d * self.d] - w[self.d * self.d:]).reshape([self.d, self.d])
        def _func(w):
            """convert function """
            W = _adj(w)
            loss, G_loss = self._loss(W)
            obj = loss + self.lambda1 * abs(W).sum()
            g_obj = np.concatenate((G_loss + self.lambda1, - G_loss + self.lambda1), axis=None)
            return obj, g_obj
        d = self.d
        w0 = np.zeros(2 * self.d * self.d)
        x = w0
        stats, _ = self._iter_comp_stats(_adj(x), 0)
        iterh    = []
        iterh.append(stats.copy())
        fold = stats['fobj']
        bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
        I, J = np.nonzero(self.support)
        supp = [(I[i], J[i]) for i in range(len(I))]
        bnds = [(0, None) if (i, j) in supp else (0, 0) for _ in range(2) for i in range(self.d) for j in range(self.d)]
        opts={'disp': False, 'maxcor': 10, 'ftol': 2.220446049250313e-09, 'gtol': 1e-05, 'eps': 1e-08, 'maxfun': 15000, 'maxiter': 15000, 'iprint': 100, 'maxls': 20, 'finite_diff_rel_step': None}
        t0 = timer()
        sol = sopt.minimize(_func, w0, method='L-BFGS-B', jac=True, bounds=bnds,\
                            options=opts)
        x = _adj(sol.x)
        ti = timer() - t0
        stats, _ = self._iter_comp_stats(x, ti, \
                            stats=stats, stepsize=np.nan,ls_info=None)
        iterh.append(stats.copy())
        fval = stats['fobj']
        msg_line = 'sol | f: %.7e | loss: %.2e | ss: %.2e | nnz: %d | nnz-g: %d | sp: %.2e | gloss: %.2e | optima: %.2e | time: %.2e' \
                    %(stats['fobj'], stats['loss'],
                stats['stepsize'], stats['nnz'], stats['nnz_g'], stats['sp'], stats['gradnorm'], stats['optimality'], stats['time'])
        print(msg_line)
        loss_0, _ = self._loss(self.Wtrue)
        f_0, g_0  = self._func(self.Wtrue)
        print('f(Wtrue) = %.7e | loss: %.4e' %(f_0,loss_0))
        return x, pd.DataFrame(iterh, columns=iterh[0].keys())

    def _eval_hfun(self, A, loram_sigma='abs'):
        # A must be a dense matrix
        if loram_sigma is 'abs':
            val =  np.trace(expm(abs(A))) - self.d
        else:
            raise ValueError('unknown LoRAM:sigma type')
        return val

    def _iter_comp_stats(self, w, time, stats=None, stepsize=None, \
                         ls_info=None, verbo=1):
        if stats is None:
            stats = {'iter':-1, 'time':0, 'stepsize':np.nan,
                    'gradnorm':np.nan, 'optimality':np.nan,
                    'fobj':np.nan, 'loss':np.nan, 'ls_nevals':np.nan,
                    'ls_succ':np.nan, 'nnz':np.nan, 'nnz_g':np.nan, 'deg':np.nan, 'hfun':np.nan, 'sp':np.nan}
        stats['iter']    += 1
        stats['time']    += time
        loss, Gt          = self._loss(w)
        stats['loss']     = loss
        fobj, g_obj       = self._func(w)
        stats['fobj']     = fobj
        g_obj[w==0] = abs(Gt[w==0] - self.lambda1) * (abs(Gt[w==0])>self.lambda1)
        stats['optimality'] = np.linalg.norm(g_obj)
        if verbo > 1:
            stats['stepsize'] = stepsize
            stats['gradnorm'] = np.linalg.norm(Gt)
            stats['hfun']     = self._eval_hfun(w)
            stats['nnz']      = (w!=0).sum()
            stats['nnz_g']      = (Gt!=0).sum()
            stats['sp']       = (w!=0).sum() / (self.d **2)
            stats['deg']      = (w!=0).sum() / self.d
            if ls_info is not None:
                stats['ls_nevals'] = ls_info['nevals']
                stats['ls_succ']  = ls_info['is_succ']
            # Accuracy
            if self.Wtrue is None:
                stats['fdr'] = np.nan
                stats['tpr'] = np.nan
                stats['fpr'] = np.nan
                stats['shd'] = np.nan
            else:
                acc = utils.count_accuracy(self.Wtrue!=0, w!=0)
                stats['fdr'] = acc['fdr']
                stats['tpr'] = acc['tpr']
                stats['fpr'] = acc['fpr']
                stats['shd'] = acc['shd']
        return stats, {'W': w}

