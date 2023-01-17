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
from icid.SparseMatDecomp import SparseMatDecomp, SpMatDecomp_primal, SpMatDecomp_primalA, ExactSpMatDecomp_primal
from icid.SpMatDecomp_Nv import ExactSpMatDecompNv_primal
from aux.gen_settings import get_markovblanket


def oicid_solver_alm(Prec, lambda1=0, maxiter=10, \
                     solver_primal='FISTA',\
                     maxiter_primal=1e3, \
                     epsilon=1e-10,
                     Wtrue=None):
    """
        Augmented Lagrangian Method (ALM)
    """
    # initialization
    rho = 1e0
    Delta = np.zeros(Prec.shape) # zeros
    resc = np.Inf
    max_rho = 1e10
    iterh = None
    id_primal = -1
    t0 = timer()
    B = np.zeros(Prec.shape)
    for ii in range(int(maxiter)):
        # Primal descent
        while rho < max_rho:
            id_primal += 1
            pb = SpMatDecomp_primal(Prec=Prec, Delta=Delta, rho=rho, \
                                    lambda1=lambda1, \
                                    iter_alm=ii, \
                                    id_primal=id_primal, \
                                    Wtrue=Wtrue, maxiter=int(maxiter_primal))
            tini = timer() - t0
            if solver_primal == 'FISTA':
                Bt, idh = pb.solver_fista_linesearch(B, verbo=1,\
                                toprint={'iter':'%d', \
                                    'loss':'%.4e', 'residual_c':'%.2e', \
                                    'stepsize':'%.2e', \
                                    'gradnorm':'%.3e', \
                                    'nnz':'%d','rho':'%.2e'})
            elif solver_primal == 'BFGS':
                Bt, idh = pb.solver_bfgs(B, verbo=1)
            else:
                raise ValueError('solver %s not available' %idec_solver)
            if idh.iloc[-1]['residual_c'] > 0.65 * resc:
                rho *= 5
            else:
                # Concatenate pandas dataframe idh
                if ii == 0:
                    iterh = idh
                else:
                    # initial time is the last iteration time
                    idh['time'] += iterh.iloc[-1]['time']
                    iterh = pd.concat([iterh, idh])
                break
        # update
        B = Bt
        acc = utils.count_accuracy(Wtrue!=0, B!=0)
        print(acc)
        # Dual ascent
        dual_grad = (np.eye(pb.d)-B) @ (np.eye(pb.d)-B).T
        dual_grad[pb.support!=0] = 0
        Delta += rho * dual_grad
        # Stopping critera
        resc = idh.iloc[-1]['residual_c']
        if (resc < epsilon or idh.iloc[-1]['loss'] < 1e-4) or rho >= max_rho:
            break
    return (B, iterh, idh)


# Formulation in matrix A, works but less accurate in discrete accuracy scores than formulation 'B' above.
# But formulation A works for unknown and non-equivariance case (NV) while formulation B now works for equivariance (EV) case only
def oicid_solverA_alm(Prec, maxiter=10, \
                     lambda1 = 1e-1, lambdad2 = 1e-1, \
                     solver_primal='FISTA',\
                     maxiter_primal=1e3, \
                     taux_rho=1.1, \
                     epsilon=1e-10,
                     Wtrue=None,\
                     verbo=2):
    """
        Augmented Lagrangian Method (ALM)
    """
    # initialization
    rho = 1e0
    rho_old = 1e0
    Delta = np.zeros(Prec.shape) # zeros
    resall = np.Inf
    max_rho = 1e5
    iterh = None
    A = np.eye(Prec.shape[0])
    #
    optinfo={'conv_reason': None, 'code_conver': 1}
    for ii in range(int(maxiter)):
        id_primal = 0
        # Primal descent
        while rho < max_rho:
            id_primal += 1
            pb = SpMatDecomp_primalA(Prec=Prec, Delta=Delta, rho=rho, \
                                    lambdad2=lambdad2, lambda1=lambda1, \
                                    iter_alm=ii, \
                                    id_primal=id_primal, \
                                    Wtrue=Wtrue, maxiter=int(maxiter_primal))
            if solver_primal == 'FISTA':
                At, idh = pb.solver_fista_linesearch(A, verbo=verbo,\
                                toprint={'iter':'%d', \
                                    'residual_all':'%.4e', \
                                    'stepsize':'%.2e', \
                                    'gradnorm':'%.3e', \
                                    'nnz':'%d','rho':'%.2e'})
            elif solver_primal == 'BFGS':
                At, idh = pb.solver_bfgs(A, verbo=verbo)
            else:
                raise ValueError('solver %s not available' %idec_solver)
            if idh.iloc[-1]['residual_all'] > 0.95 * resall:
                #              rho *= 1.1 #1.05  # generic
                rho *= taux_rho # for cases with low node degree
            else:
                # Accept solution Bt for this primal problem
                resall = idh.iloc[-1]['residual_all']
                # Concatenate pandas dataframe idh
                if ii == 0:
                    iterh = idh
                else:
                    # initial time is the last iteration time
                    idh['time'] += iterh.iloc[-1]['time']
                    iterh = pd.concat([iterh, idh])
                break
        # Update
        A = At
        B = (np.diag(np.diag(A)) - A) @ np.diag(1/np.diag(A))
        acc = utils.count_accuracy(Wtrue!=0, B!=0)
        print(acc)
        # Dual ascent
        phia = A @ A.T
        Delta += rho * (phia-Prec)
        # Stopping critera
        if (resall < epsilon):
            optinfo['conv_reason'] = 'Success: target precision reached'
            optinfo['code_conver'] = 2
            break
        # Too many increments of rho while rho < max_rho
        if id_primal > 10 or rho >= max_rho:
            msg_ = 'ALM rho parameter depasses max or too many increments of rho during one iteration!'
            print(msg_)
            optinfo['code_conver'] = 0
            optinfo['conv_reason'] = msg_
            break
    return (B, iterh, optinfo)

# Formulation (in matrix B) for exact fitting of Theta
# primal solver: BFGS works better in this formulation
def oicid_solver_alm_exactfit(Prec, maxiter=10, \
                     lambda1 = 1e-1, \
                     solver_primal='FISTA',\
                     maxiter_primal=1e3, \
                     taux_rho=1.1, \
                     epsilon=1e-10,
                     Wtrue=None,\
                     verbo=2):
    """
        Augmented Lagrangian Method (ALM)
    """
    # initialization
    rho = 1e0
    rho_old = 1e0
    Delta = np.zeros(Prec.shape) # zeros
    resall = np.Inf
    max_rho = 1e5
    iterh = None
    B = np.zeros(Prec.shape)
    # Bave = np.zeros(Prec.shape) # averaging // not used here
    optinfo={'conv_reason': None, 'code_conver': 1}
    for ii in range(int(maxiter)):
        id_primal = 0
        # Primal descent
        while rho < max_rho:
            id_primal += 1
            pb = ExactSpMatDecomp_primal(Prec=Prec, Delta=Delta, rho=rho, \
                                    lambda1=lambda1, \
                                    iter_alm=ii, \
                                    id_primal=id_primal, \
                                    Wtrue=Wtrue, maxiter=int(maxiter_primal))
            if solver_primal == 'FISTA':
                Bt, idh = pb.solver_fista_linesearch(B, verbo=verbo,\
                                toprint={'iter':'%d', \
                                    'residual_all':'%.4e', \
                                    'stepsize':'%.2e', \
                                    'gradnorm':'%.3e', \
                                    'nnz':'%d','rho':'%.2e'})
            elif solver_primal == 'BFGS':
                Bt, idh = pb.solver_bfgs(B, verbo=verbo)
            else:
                raise ValueError('solver %s not available' %idec_solver)
            if idh.iloc[-1]['residual_all'] > 0.95 * resall:
                #              rho *= 1.1 #1.05  # generic
                rho *= taux_rho # for cases with low node degree
            else:
                # Accept solution Bt for this primal problem
                resall = idh.iloc[-1]['residual_all']
                # Concatenate pandas dataframe idh
                if ii == 0:
                    iterh = idh
                else:
                    # initial time is the last iteration time
                    idh['time'] += iterh.iloc[-1]['time']
                    iterh = pd.concat([iterh, idh])
                break
        # Update
        B = Bt
        acc = utils.count_accuracy(Wtrue!=0, B!=0)
        print(acc)
        # Dual ascent
        phib = np.eye(pb.d) - B - B.T + B @ B.T
        Delta += rho * (Prec - phib)
        # Stopping critera
        if (resall < epsilon):
            optinfo['conv_reason'] = 'Success: target precision reached'
            optinfo['code_conver'] = 2
            break
        # Too many increments of rho while rho < max_rho
        if id_primal > 10 or rho >= max_rho:
            msg_ = 'ALM rho parameter depasses max or too many increments of rho during one iteration!'
            print(msg_)
            optinfo['code_conver'] = 0
            optinfo['conv_reason'] = msg_
            break
    return (B, iterh, optinfo)


# Nv case: exact fitting of Theta
def oicidNV_solver_alm_exactfit(Prec, maxiter=10, \
                     lambda1 = 1e-1, \
                     solver_primal='FISTA',\
                     maxiter_primal=1e3, \
                     taux_rho=1.1, \
                     epsilon=1e-10,
                     Wtrue=None,\
                     verbo=2):
    """
        Augmented Lagrangian Method (ALM)
    """
    # initialization
    rho = 1e0
    rho_old = 1e0
    Delta = np.zeros(Prec.shape) # zeros
    resall = np.Inf
    max_rho = 1e5
    iterh = None
    B = np.zeros(Prec.shape)
    # Bave = np.zeros(Prec.shape) # averaging // not used here
    optinfo={'conv_reason': None, 'code_conver': 1}
    for ii in range(int(maxiter)):
        id_primal = 0
        # Primal descent
        while rho < max_rho:
            id_primal += 1
            pb = ExactSpMatDecompNv_primal(Prec=Prec, Delta=Delta, rho=rho, \
                                    lambda1=lambda1, \
                                    iter_alm=ii, \
                                    id_primal=id_primal, \
                                    Wtrue=Wtrue, maxiter=int(maxiter_primal))
            if solver_primal == 'FISTA':
                Bt, idh = pb.solver_fista_linesearch(B, verbo=verbo,\
                                toprint={'iter':'%d', \
                                    'residual_all':'%.4e', \
                                    'stepsize':'%.2e', \
                                    'gradnorm':'%.3e', \
                                    'nnz':'%d','rho':'%.2e'})
            # elif solver_primal == 'BFGS':
            #     Bt, idh = pb.solver_bfgs(B, verbo=verbo)
            else:
                raise ValueError('solver %s not available' %idec_solver)
            if idh.iloc[-1]['residual_all'] > 0.95 * resall:
                #              rho *= 1.1 #1.05  # generic
                rho *= taux_rho # for cases with low node degree
            else:
                # Accept solution Bt for this primal problem
                resall = idh.iloc[-1]['residual_all']
                # Concatenate pandas dataframe idh
                if ii == 0:
                    iterh = idh
                else:
                    # initial time is the last iteration time
                    idh['time'] += iterh.iloc[-1]['time']
                    iterh = pd.concat([iterh, idh])
                break
        # Update
        B = Bt[0]
        acc = utils.count_accuracy(Wtrue!=0, B!=0)
        print(acc)
        # Dual ascent
        phip=  (np.eye(Prec.shape[0]) - B) @ np.diag(Bt[1])
        phib = phip @ (np.eye(Prec.shape[0]) - B.T)
        Delta += rho * (Prec - phib)
        # Stopping critera
        if (resall < epsilon):
            optinfo['conv_reason'] = 'Success: target precision reached'
            optinfo['code_conver'] = 2
            break
        # Too many increments of rho while rho < max_rho
        if id_primal > 10 or rho >= max_rho:
            msg_ = 'ALM rho parameter depasses max or too many increments of rho during one iteration!'
            print(msg_)
            optinfo['code_conver'] = 0
            optinfo['conv_reason'] = msg_
            break
    return (Bt, iterh, optinfo)

