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
import time, os
import pandas as pd
import pywt

from icid import utils


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
        #
        u,s,v = np.linalg.svd(Prec)
        self.lipschitz_loss = max(max(s),1)

    def _loss(self, B):
        """Evaluate value and gradient of loss of
            l(B) = 0.5 * |S  - invsigma_0 * phi(B)|_F^2, where
                 = - tr(S phi(B)) + 0.5*invsigma_0 * |phi(B)|_F^2, where
        phi(B) = (I-B)(I-B)^T
        """
        B[self.support==0] = 0 # TODO: remove this line, since the
                          # gradient is already in the subspace
        symb = B + B.T
        quadb = B @ B.T
        phib = np.eye(self.d) - symb + quadb
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
        t1 = 0
        stats, _ = self._iter_comp_stats(x, t1)
        iterh    = []
        iterh.append(stats.copy())
        fold = stats['fobj']
        ss = self.lipschitz_loss / 100
        for i in range(self.maxiter):
            ti = timer()
            xold = x
            yplus, ss, loss, Gt, lsinfo = self.gd_f_linesearch(y, s0=ss, c1=ls_c1, beta=ls_beta)
            x = pywt.threshold(yplus, ss*self.lambda1, 'soft')
            y = x + i * (x - xold) / (i+3)
            t1 += timer() - ti
            stats, _ = self._iter_comp_stats(x, t1, stepsize=ss,ls_info=lsinfo)
            iterh.append(stats.copy())
            fval = stats['fobj']
            msg_line = 'iter: %d | f: %.7e | loss: %.2e | ss: %.2e | nnz: %d | sp: %.2e | gloss: %.2e | optima: %.2e | time: %.2e' %(i+1, stats['fobj'], stats['loss'], stats['stepsize'], stats['nnz'], stats['sp'], stats['gradnorm'], stats['optimality'], stats['time'])
            if verbo==1 and i % 50 == 0:
                print(msg_line)
            if verbo==2:
                print(msg_line)
            if abs(fold-fval) / abs(fold) < self.tol:
                break
            fold = fval
        return x, pd.DataFrame(iterh, columns=iterh[0].keys())

    def _eval_hfun(self, A, loram_sigma='abs'):
        # A must be a dense matrix
        if loram_sigma is 'abs':
            val =  np.trace(expm(abs(A))) - self.d
        else:
            raise ValueError('unknown LoRAM:sigma type')
        return val

    def _iter_comp_stats(self, w, time, stats=None, stepsize=None, ls_info=None):
        if stats is None:
            stats = {'iter':-1, 'time':0, 'stepsize':np.nan, 'gradnorm':np.nan, 'optimality':np.nan, 'fobj':np.nan, 'loss':np.nan, 'ls_nevals':np.nan, 'ls_succ':np.nan, 'nnz':np.nan, 'deg':np.nan, 'hfun':np.nan, 'sp':np.nan}
        stats['iter']    += 1
        stats['time']    += time
        stats['stepsize'] = stepsize
        if ls_info is not None:
            stats['ls_nevals'] = ls_info['nevals']
            stats['ls_succ']  = ls_info['is_succ']
        stats['nnz']      = (w!=0).sum()
        stats['sp']       = (w!=0).sum() / (self.d **2)
        stats['deg']      = (w!=0).sum() / self.d
        loss, Gt          = self._loss(w)
        stats['loss']     = loss
        stats['gradnorm'] = np.linalg.norm(Gt)
        fobj, g_obj       = self._func(w)
        stats['fobj']     = fobj
        g_obj[w==0] = abs(Gt[w==0] - self.lambda1) * (abs(Gt[w==0])>self.lambda1)
        stats['optimality'] = np.linalg.norm(g_obj)
        stats['hfun']     = self._eval_hfun(w)
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

