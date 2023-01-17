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
class SpMatDecomp_Nv():
    def __init__(self, Prec, lambda1=0, \
                 loss_type='l2', maxiter=1000,\
                 invsigma_0=1.0, tol=1e-7,\
                 Wtrue=None):
        """
        Inputs
        """
        self.info = {'time': time.strftime("%H%M%S%m%d")}
        self.S       = Prec
        tmp = Prec.copy()
        # Mask must not contain diagonal
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
    """
        /// BEGIN - Model functions, may be overridden in subclass
    """
    def _loss(self, B, D):
        """Evaluate value and gradient of loss of
            l(B) = 0.5 * |S  - invsigma_0 * phi(B)|_F^2, where
                 = - tr(S phi(B)) + 0.5*invsigma_0 * |phi(B)|_F^2, where
        phi(B) = (I-B)(I-B)^T
        """
        phip=  (np.eye(self.d) - B) @ np.diag(D)
        phib = phip @ (np.eye(self.d) - B.T)
        residual = self.S - phib
        loss = 0.5 * np.linalg.norm(residual) ** 2
        Gb_loss = 2* residual @ (np.diag(D) - B@np.diag(D))
        Gb_loss += -np.diag(np.diag(Gb_loss))
        br = B.T @ residual
        Gd_loss = np.diag(-residual + br.T + br - br@B)
        # Project Gb_loss back onto the subspace of the support graph
        Gb_loss[self.support==0] = 0
        return loss, (Gb_loss, Gd_loss)

    def _func(self, B, D):
        """Evaluate value and gradient of f = loss + l1 """
        loss, G_loss = self._loss(B, D)
        obj = loss + self.lambda1 * abs(B).sum()
        g_obj = G_loss[0] + self.lambda1 * np.sign(B)
        return obj, (g_obj, G_loss[1])
    """
        /// END - Model functions, may be overridden in subclass
    """
    def initialize_w(self, w0=None, option='zeros'):
        if w0 is not None:
            w = w0
            w[self.support==0] = 0
        else:
            if option == 'iden':
                w = np.eye(self.d)
            elif option == 'zeros':
                w = np.zeros([self.d,self.d])
            elif option == 'gaussian':
                w = np.random.normal(size=[self.d,self.d])
                w[self.support==0] = 0
            else:
                raise ValueError('unknown initialization type')
        return w

    def gd_f_linesearch(self, w, d, s0=1, c1=0.8, beta=0.5, maxevals=10):
        # w:     dxd off diagonal matrix
        # d:     dx1 diagonal entries
        ss = s0
        d[d<1] = 1
        f0, G_loss = self._loss(w, d)
        for i in range(maxevals):
            wn = w - ss * G_loss[0] #
            dn = d - 1e-1* ss * G_loss[1] #
            dn[dn<1] = 1
            loss, G_t = self._loss(wn, dn)
            #if min(dn) <= 0: # safeguard positiveness
            #    ss *= beta
            #else:
            if loss > f0 - c1 * ss* (G_loss[0] * G_loss[0]).sum():
                # print('backtracking now ...')
                ss *= beta
            else:
                break
        if (i+1) == maxevals:
            print('linesearch failed')
        wn = wn - np.diag(np.diag(wn))
        return (wn,dn), ss, loss, G_t, {'nevals':i+1, 'is_succ': (i+1<maxevals)}

    def solver_fista_linesearch(self, w0, ls_c1=0.8, ls_beta=0.5, \
                                toprint={'iter':'%d','loss':'%.4e', \
                                         'stepsize':'%.2e', \
                                         'gradnorm':'%.3e', \
                                         'nnz':'%d'}, \
                                verbo=2):
        y = w0
        x = y
        d = np.ones(w0.shape[0])
        iterh    = []
        stats, _ = self._iter_comp_stats(self.Wtrue, time=-np.inf, d=d, stats=None) # not to be included in the real iter history
        iterh.append(stats.copy())
        stats, _ = self._iter_comp_stats(x, 0, d=d)
        iterh.append(stats.copy())
        fold = stats['fobj']
        # ss = self.lipschitz_loss / 100
        ss = 1.0
        # Optional
        if verbo > 1:
            loss_0, _ = self._loss(self.Wtrue, d)
            f_0, g_0  = self._func(self.Wtrue, d)
            print('f(Wtrue) = %.7e | loss: %.4e' %(f_0,loss_0))
        # FISTA iterations
        for i in range(int(self.maxiter)):
            t0 = timer()
            xold = x
            dold = d
            ydplus, ss, loss, Gt, lsinfo = self.gd_f_linesearch(y, d, s0=ss, c1=ls_c1, beta=ls_beta)
            # yplus, ss, loss, Gt, lsinfo = self.gd_f_linesearch(y, s0=1e0, c1=ls_c1, beta=ls_beta)
            yplus, dplus = ydplus[0], ydplus[1]
            x = _threshold_soft(yplus, ss*self.lambda1)
            y = x + i * (x - xold) / (i+3)
            d = dplus + i * (dplus - dold) / (i+3)
            ti = timer() - t0
            stats, _ = self._iter_comp_stats(x, ti, d=d, \
                            stats=stats, stepsize=ss,\
                            ls_info=lsinfo, verbo=1)
            iterh.append(stats.copy())
            fval = stats['fobj']
            # Print iteration status
            if (verbo==1 and i % 200 == 0) or (verbo>1):
                pstats, _ = self._iter_comp_stats(x, 0, d=d,\
                            stats=stats.copy(), stepsize=ss,\
                            ls_info=lsinfo, verbo=2)
                msg_ = ' | '.join('{}: {}'.format(key, ff%pstats[key]) \
                        for key, ff in toprint.items() )
                fval = pstats['fobj']
                print(msg_)
            # Stopping criterion
            if abs(fold-fval) / abs(fold) < self.tol and \
                    stats['optimality'] < 1e-4:
                break
            fold = fval
        return (x,d), pd.DataFrame(iterh, columns=iterh[0].keys())

    def _iter_comp_stats(self, w, time, stats=None, stepsize=None, \
                           d=None, ls_info=None, verbo=1):
        if stats is None:
            stats = {'iter':-1, 'time':0, 'stepsize':np.nan,
                    'gradnorm':np.nan, 'optimality':np.nan,
                    'fobj':np.nan, 'loss':np.nan, 'ls_nevals':np.nan,
                    'ls_succ':np.nan, 'nnz':np.nan, 'nnz_g':np.nan, 'deg':np.nan, 'sp':np.nan}
        stats['iter']    += 1
        stats['time']    += time
        loss, Gt2          = self._loss(w,d)
        Gt = Gt2[0]
        stats['loss']     = loss
        fobj, g_obj2       = self._func(w,d)
        g_obj = g_obj2[0]
        stats['fobj']     = fobj
        g_obj[w==0] = abs(Gt[w==0] - self.lambda1) * (abs(Gt[w==0])>self.lambda1)
        stats['optimality'] = np.linalg.norm(g_obj)
        stats['gradnorm'] = np.linalg.norm(Gt) + np.linalg.norm(Gt2[1])
        if verbo > 1:
            stats['stepsize'] = stepsize
            stats['nnz']      = (w!=0).sum()
            stats['nnz_g']      = (Gt!=0).sum()
            stats['sp']       = (w!=0).sum() / (self.d **2)
            stats['deg']      = (w!=0).sum() / self.d
            if ls_info != None:
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

class ExactSpMatDecompNv_primal(SpMatDecomp_Nv):
    def __init__(self, Prec, Delta, rho, lambda1=1e-1, \
                 iter_alm = 0, id_primal = 0, \
                 loss_type='l2', maxiter=1000,\
                 tol=1e-7, Wtrue=None):
        """INPUT
            Delta:     Dual variable, dxd array containing Lagrange multipliers
            rho:       augmentation parameter
        """
        # Record the support of Prec
        self.supp_prec = (Prec!=0) #
        # Prototype initialization Python 3.x:
        super().__init__(Prec, lambda1=lambda1, \
                        loss_type=loss_type, maxiter=maxiter,\
                        tol=tol, Wtrue=Wtrue)
        # Container for the subclass quantities:
        self.dual_var = Delta
        self.dual_grad = None
        self.aux = {'iter_alm': iter_alm, \
                    'id_primal': id_primal, \
                    'rho_alm': rho \
                     }

    def _loss(self, B, D):
        """Evaluate value and gradient of the augmented Lagrangian
        """
        rho = self.aux['rho_alm']
        # 'loss' term (in the superclass) is zero here, all loss terms are now subject to equality constraint
        phip=  (np.eye(self.d) - B) @ np.diag(D)
        phib = phip @ (np.eye(self.d) - B.T)
        residual = self.S - phib

        # Primal function value of the augmented Lagrangian:
        dualop = self.dual_var
        inn_primaldual = sum((dualop * residual).ravel())
        aug_l2 = 0.5 * rho * np.linalg.norm(residual)**2
        val = inn_primaldual + aug_l2
        # full gradient
        Gb_lag = (2*rho*residual + (dualop+dualop.T)) @ (np.diag(D) - B@np.diag(D))
        # Gb_lag += -np.diag(np.diag(Gb_lag)) # addressed by mask of 'support'
        # gradient in D
        R = 2*rho*residual + (dualop + dualop.T)
        br = B.T @ R
        Gd_lag = np.diag(-R + br.T + br - br@B)
        # Project gradient onto the subspace of the support graph
        Gb_lag[self.support==0] = 0
        return val, (Gb_lag, Gd_lag)

    # (#117) TODO: to continue, updated till here
    def _iter_comp_stats(self, B, time, stats=None, stepsize=None, \
                          d=None, ls_info=None, verbo=1):
        stats, _ = super()._iter_comp_stats(B, time, stats=stats, stepsize=stepsize,\
                                            d=d, ls_info=ls_info, verbo=verbo)

        # Info of this iterator
        stats['iter_alm'] = self.aux['iter_alm']
        stats['id_primal'] = self.aux['id_primal']
        stats['rho'] = self.aux['rho_alm']
        rho = self.aux['rho_alm']
        # Residual on Support and complementary of Support
        phip=  (np.eye(self.d) - B) @ np.diag(d)
        phib = phip @ (np.eye(self.d) - B.T)
        residual = self.S - phib
        res = residual.copy()
        res[self.supp_prec==0] = 0
        stats['residual_s'] = np.linalg.norm(res) #
        stats['residual_c'] = np.linalg.norm(residual-res) #
        # Aug Lagrangian terms
        aug_l2 =0.5 * rho * np.linalg.norm(residual)**2
        stats['residual_all'] = np.sqrt(2*aug_l2 / rho)
        stats['augl2'] = aug_l2
        stats['inn_primaldual'] = sum((self.dual_var * residual).ravel())
        stats['primal']         = stats['loss'] # given by the class _loss function
        if stats['residual_all'] > 0:
            stats['ratio_c'] = stats['residual_c'] / stats['residual_all']
        else:
            stats['ratio_c'] = np.nan
        #
        if verbo <= 1: # complete the stats despite low verbo
            stats['stepsize'] = stepsize
            stats['hfun']     = np.nan #self._eval_hfun(w)
            stats['nnz']      = (B!=0).sum()
            stats['sp']       = (B!=0).sum() / (self.d **2)
            stats['deg']      = (B!=0).sum() / self.d
            if ls_info != None:
                stats['ls_nevals'] = ls_info['nevals']
                stats['ls_succ']  = ls_info['is_succ']
            # Accuracy
            if self.Wtrue is None:
                stats['fdr'] = np.nan
                stats['tpr'] = np.nan
                stats['fpr'] = np.nan
                stats['shd'] = np.nan
            else:
                stats['wdist'] = np.linalg.norm(B-self.Wtrue)
                acc = utils.count_accuracy(self.Wtrue!=0, B!=0)
                stats['fdr'] = acc['fdr']
                stats['tpr'] = acc['tpr']
                stats['fpr'] = acc['fpr']
                stats['shd'] = acc['shd']
        return stats, _

