"""
The class Loram provides algorithms for general-purpose optimization of the
low-rank additive model (LoRAM).

The subclass LoramPrxoDAG implements the Loram-AGD algorithm of [1] for the
computation of proximal mappings with respect to the DAG characteristic
function h.

Reference:

[1]:

URL of the package

Contact:
"""

import numpy as np
import random
from timeit import default_timer as timer
import pandas as pd

from icid import spmaskmatmul
from icid import utils
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import expm, expm_multiply
from icid.splr_expmv import splr_expmv_inexact1_paired, splr_expmv_inexact_1, splr_expmv_inexact_1b, splr_sigma_abs, splr_func_abs


class Loram():

    def __init__(self, d, k, I=None, J=None):
        """
        Arguments
        - d (int)           : number of nodes
        - k (int)           : number of latent dimensions
        """
        self.d = d
        self.k = k
        self.I = I
        self.J = J
        self.z = {'X':np.ones((self.d, self.k)), 'Y': np.ones((self.d, self.k)) }
        self.tol_h = 1e-16
        self.tol_grad = 1e-16

    """ GEOMETRY OF THE PRODUCT SPACE
    """
    def scalar_times(self, scalar, dir_):
        x_ = scalar * dir_['X']
        y_ = scalar * dir_['Y']
        return {'X':x_, 'Y': y_}

    def lin_comb(self, scalar_1, dir_1, scalar_2=None, dir_2=None):
        x_ = scalar_1 * dir_1['X'] + scalar_2 * dir_2['X']
        y_ = scalar_1 * dir_1['Y'] + scalar_2 * dir_2['Y']
        return {'X':x_, 'Y': y_}

    def elem_inner_prod(self, mat_1, mat_2):
        return np.sum(mat_1 * mat_2)

    def lin_inner_prod(self, dir_1, dir_2):
        """ Inner product in the euclidean space of R(dxk) x R(dxk)
        """
        return self.elem_inner_prod(dir_1['X'], dir_2['X']) + \
                self.elem_inner_prod(dir_1['Y'], dir_2['Y'])

    def lin_tspace_norm(self, dir_):
        return np.sqrt(self.elem_inner_prod(dir_['X'], dir_['X']) + \
                self.elem_inner_prod(dir_['Y'], dir_['Y']))

    def lin_tspace_odot(self, dir_):
        return {'X': dir_['X'] * dir_['X'], \
                'Y': dir_['Y'] * dir_['Y']}

    def lin_tspace_divide(self, dir_1, dir_2):
        return {'X': dir_1['X'] / dir_2['X'], \
                'Y': dir_1['Y'] / dir_2['Y']}

    """ LoRAM MATRIX OPERATIONS
    """
    def get_adjmatrix_dense(self, z):
        loram_vec, loram_abs, _= spmaskmatmul.spmaskmatmul_abs(z['X'], z['Y'], \
                                                      self.I, self.J)
        return np.asarray(csc_matrix((loram_vec, (self.I, self.J)), \
                            shape=(self.d, self.d)).todense()), max(loram_abs)

    def get_scale_loram(self, z):
        """computes the infinity norm of the loram variable """
        _, loram_abs, _ = spmaskmatmul.spmaskmatmul_abs(z['X'], z['Y'], \
                                                      self.I, self.J)
        return max(loram_abs)

    def get_adj_matrix(self, z):
        splr_vec, _, _= spmaskmatmul.spmaskmatmul_abs(z['X'], z['Y'], \
                                                      self.I, self.J)
        return csc_matrix((splr_vec, (self.I, self.J)), shape=(self.d, self.d))

    def _gen_I_J(self, rho):
        m = np.ceil(rho * self.d **2)
        supp = random.sample(range(self.d**2), \
                             min(np.inf, np.ceil(m).astype(np.int)))
        # Convert 1d index supp into 2d index of matrices
        x, y = np.unravel_index(supp, (self.d,self.d))
        self.I = np.array(x).astype(np.uintc)
        self.J = np.array(y).astype(np.uintc)

    def _get_I_J_zref(self, zref):
        """ Get the index set of all edges in the reference (non-DAG) graph
        """
        ind_2d = np.nonzero(zref)
        return {'I': ind_2d[0].astype(np.uintc),
                'J': ind_2d[1].astype(np.uintc)}


    """ ELEMENTARY FUNCTIONS
    """
    def _comp_naive_func_h(self, z):
        """ The exponential trace of po(XY') minus d """
        _, sigmaZ, _= spmaskmatmul.spmaskmatmul_abs(z['X'], z['Y'], self.I, self.J)
        sca = 1
        mat_z = csc_matrix((sigmaZ/sca, (self.I, self.J)), shape=(self.d, self.d))
        val =  np.trace(expm(mat_z).todense()) - self.d
        return val

    def _comp_grad_h_exact(self, z):
        A_, dA_, sp_zvec = splr_sigma_abs(z['X'], z['Y'], self.I, self.J)
        # Form A' (transpose of A)
        At = csc_matrix((A_, (self.J, self.I)), shape=(self.d, self.d))
        # Form the matrix of mask(Z)
        dA = csc_matrix((dA_, (self.I, self.J)), shape=(self.d, self.d))
        # Record the matrices to iterdb
        S = dA.multiply(expm(At))
        gh_x = S.dot(z['Y'])
        gh_y = (S.T).dot(z['X'])
        return {'X': gh_x, 'Y': gh_y}, At, dA, sp_zvec

    def _comp_grad_h_inexact1(self, z):
        A_, dA_, sp_zvec = splr_sigma_abs(z['X'], z['Y'], self.I, self.J)
        """ Form A' (transpose of A) """
        At = csc_matrix((A_, (self.J, self.I)), shape=(self.d, self.d))
        """ Form the matrix of mask(Z) """
        dA = csc_matrix((dA_, (self.I, self.J)), shape=(self.d, self.d))
        """ Record the matrices to iterdb """
        return {'X': splr_expmv_inexact_1(At, dA, z['Y']), \
                 'Y': splr_expmv_inexact_1(At.T, dA.T, z['X'])}, At, dA, sp_zvec

    def _comp_grad_h_inexact1b(self):
        A_, dA_, sp_zvec = splr_sigma_abs(z['X'], z['Y'], self.I, self.J)
        # Form A' (transpose of A)
        At = csc_matrix((A_, (self.J, self.I)), shape=(self.d, self.d))
        # Form the matrix of mask(Z)
        dA = csc_matrix((dA_, (self.I, self.J)), shape=(self.d, self.d))
        # Record the matrices to iterdb
        return {'X': splr_expmv_inexact_1b(At, dA, z['Y']), \
                 'Y': splr_expmv_inexact_1b(At.T, dA.T, z['X'])}, At, dA, sp_zvec

    def _compute_grad_h_atsparse(self, thres=1e-2):
        A_, dA_, sp_zvec = splr_sigma_abs(self.z['X'], self.z['Y'], self.I, self.J)
        Cmax = max(abs(A_))
        dA_[abs(A_)<thres*Cmax] = 0.0
        A_[abs(A_)<thres*Cmax] = 0.0
        # Form A' (transpose of A)
        At = csc_matrix((A_, (self.J, self.I)), shape=(self.d, self.d))
        # Form the matrix of mask(Z)
        dA = csc_matrix((dA_, (self.I, self.J)), shape=(self.d, self.d))
        # Record the matrices to iterdb
        return {'X': splr_expmv_inexact_1b(At, dA, self.z['Y']), \
                 'Y': splr_expmv_inexact_1b(At.T, dA.T, self.z['X'])}, At, dA


    """ OTHER AUXILARY FUNCTIONS
    """
    def init_factors(self, z_init):
        # print('Initialize z with a given point')
        self.z = z_init

    def init_factors_gaussian(self, scale = 1e-2):
        # print('Default initialization method: Gaussian matrices')
        z = {'X': np.random.normal(scale=scale, size=[self.d,self.k]),
             'Y': np.random.normal(scale=scale, size=[self.d,self.k])}
        self.z = z
        return z

    def init_factors_zero(self):
        return {'X': np.zeros([self.d,self.k]),
                'Y': np.zeros([self.d,self.k])}

    """ stepsize computations """
    def _comp_stepsize_bb_raw(self, z, z_old, dir_, dir_old):
        """ dir_desc is a pair of matrices in R(dxk) x R(dxk) """
        z   = self.lin_comb(1, z, -1, z_old)
        y   = self.lin_comb(-1, dir_, 1, dir_old)
        # sbb1
        sbb = min(1e1, max(1e-15, \
                self.lin_inner_prod(z, z) / self.lin_inner_prod(z, y) ))
        # sbb2
        # sbb = min(1e3, max(1e-15, lin_inner_prod(z, y) / lin_inner_prod(y, y)))
        return sbb

    def _comp_stepsize_bb(self ):
        z = self.iterdb['z']
        z_old = self.iterdb_old['z']
        dir_ = self.iterdb['dir_desc']
        dir_old = self.iterdb_old['dir_desc']
        """ dir_desc is a pair of matrices in R(dxk) x R(dxk) """
        z   = self.lin_comb(1, z, -1, z_old)
        y   = self.lin_comb(-1, dir_, 1, dir_old)
        # sbb1
        sbb = min(1e1, max(1e-15, \
                self.lin_inner_prod(z, z) / self.lin_inner_prod(z, y) ))
        # sbb2
        # sbb = min(1e3, max(1e-15, lin_inner_prod(z, y) / lin_inner_prod(y, y)))
        return sbb

    def _gd_fixedstep(self, alpha, stepsize):
        """
        Perform gradient descent with respect to the primal function
                F = f + alpha * h
        """
        grad_f, grad_h, sp_zvec, At, dA, res_vec \
                    = self._comp_grad_f_h(self.z)
        _dir_desc   = self.lin_comb(-1/alpha, grad_f, -1, grad_h)
        self.iterdb = {'z': self.z, 'grad_f':grad_f, \
                       'grad_h': grad_h, 'dir_desc': _dir_desc, \
                       'sp_zvec': sp_zvec, 'A':At, 'dA':dA, \
                       'res_vec': res_vec, 'iter':0}
        self.z      = self.lin_comb(1, self.z, stepsize , _dir_desc)
        return stepsize

    def solver_primal_accGD(self, z_init,  s_init=1e-3):
        """ Accelrated gradient descent for solving the primal problem
            alpha     :  dual parameter
        """
        alpha = self.alpha
        x_old = z_init

        # 1st iteration
        t0 = timer()
        self._gd_fixedstep(alpha, s_init)
        x_new = self.z
        ti = timer() - t0

        iterhist = []
        stat = self._comp_iterhist(x_old, time=ti, niter=0, alpha=alpha)
        iterhist.append(stat.copy())

        for i in range(self.maxiter):
            t0 = timer()
            # The auxilary point y is maintained by self.z
            self.iterdb_old = self.iterdb.copy()
            x_old = x_new
            # Compute the BB stepsize:
            grad_f, grad_h, sp_zvec, At, dA, res_vec \
                        = self._comp_grad_f_h(self.z)
            _dir_desc   = self.lin_comb(-1/alpha, grad_f, -1, grad_h)
            self.iterdb = {'z': self.z, 'grad_f':grad_f, \
                           'grad_h': grad_h, 'dir_desc': _dir_desc, \
                           'sp_zvec': sp_zvec, 'A':At, 'dA':dA, \
                           'res_vec': res_vec, 'iter': i+1}
            stepsize    = max(self._comp_stepsize_bb(), 1e-20)
            # Gradient descent with Nesterov's acceleration:
            x_new       = self.lin_comb(1, self.z, stepsize , _dir_desc)
            self.z      = self.lin_comb(1, x_new, i/(i+3), \
                                    self.lin_comb(1, x_new, -1, x_old))
            ti = timer() - t0
            # Get iter stats
            stat = self._comp_iterhist(x_new, time=ti, niter=i+1, \
                        stat=stat, \
                        stepsize=stepsize, alpha=alpha, \
                        xold=x_old, verbo=1)
            iterhist.append(stat.copy())
            if (i+1) % 100 == 0:
                stat_p = self._comp_iterhist(x_new, time=0, niter=i+1, \
                        stat=stat, \
                        stepsize=stepsize, alpha=alpha,\
                        xold=x_old, verbo=2)
                print("%d| f: %.2e | alpha: %.2e | h: %.7e | gradn: %.3e | s: %.4e| %.3e (sec)" % (i+1, stat_p['f_residual'], \
                        alpha, stat_p['hval'], stat_p['gradnorm'], \
                        stepsize, stat_p['time']))
            if stat['gradnorm'] <= self.tol_grad: #or self.stat['gradnorm'] <= 0.4*iterhist[min(i,max(i-200,100))]['gradnorm']:
                break
        iterhist.append(stat.copy())
        iterh = pd.DataFrame(iterhist)
        return x_new, iterh

    def solve_agd(self, z_init,  s_init=1e-3):
    # NOTE (#612): bug fix in '_comp_grad_f_precomp' which acts on self.z, while this
    # algorithm does not update self.z at all!
        """ Accelrated gradient descent for solving the primal problem
            alpha     :  dual parameter
        """

        # 1st iteration by gradient descent with line search
        t0 = timer()
        y_old = z_init
        y, aux = self._gd_linesearch(y_old, s0=1e-3)
        ti = timer() - t0
        #       Store info of 'y' in iterdb
        dir0 = self.scalar_times(-1, aux['grad0'])
        self.iterdb_old = {'z': y_old, 'dir_desc': dir0, \
                'A':aux['A0'],'res_vec': aux['res_vec0']}
        self.iterdb = self.iterdb_old.copy()
        stat = self._comp_iterhist(z_init, time=0, niter=0)
        iterhist = []
        iterhist.append(stat.copy())

        self.iterdb['z'] = y
        self.iterdb['dir_desc'] = self.scalar_times(-1, aux['grad'])
        self.iterdb['A'] = aux['At']
        self.iterdb['res_vec'] = aux['res_vec']
        #       Gather iter history stats
        stat = self._comp_iterhist(y, time=ti, stat=stat)
        iterhist.append(stat.copy())

        # Store info of 'x' in as ('x', 'x_old')
        x = y
        # The auxilary point y is maintained by self.z
        for i in range(self.maxiter):
            t0 = timer()
            # Compute the BB stepsize:
            sbb = self._comp_stepsize_bb()
            stepsize    = max(sbb, 1e-20)
            # Save current info of 'y' and 'x' as old before AGD
            self.iterdb_old = self.iterdb.copy() # info of 'y'
            x_old  = x                           # info of 'x'
            # Nesterov's AGD updates
            # Descent dir at y is always sync'ed with 'y'
            y_dir_desc = self.iterdb['dir_desc']
            x      = self.lin_comb(1, y, stepsize , y_dir_desc)
            y      = self.lin_comb(1, x, (i+1)/(i+4), \
                                self.lin_comb(1, x, -1, x_old))
            # Compute gradient info of new 'y' and sync into iterdb
            grad_y, info = self._comp_grad(y)
            self.iterdb['dir_desc'] = self.scalar_times(-1, grad_y)
            self.iterdb['z'] = y
            self.iterdb['A'] = info['At']
            self.iterdb['res_vec'] = info['res_vec']
            ti = timer() - t0

            # Get iter stats
            stat = self._comp_iterhist(x, time=ti, stat=stat, niter=i)
            iterhist.append(stat.copy())

            # Print iter info
            if (i+1) % 1 == 0:
                print("%d| f: %.2e | alpha: %.2e | h: %.7e | gradn: %.3e | s: %.4e| %.3e (sec)" % (i+1, stat['f_residual'], \
                        self.alpha, stat['hval'], stat['gradnorm'], \
                        stepsize, ti))
            if stat['gradnorm'] <= 1e-6: #
                break
        return x, pd.DataFrame(iterhist)

    def _gd_linesearch(self, w, s0=1, c1=0.8, beta=0.5, maxevals=10):
        ss = s0
        grad, info0 = self._comp_grad(w)
        F0, _,_ = self._comp_func(w,res_vec=info0['res_vec'])
        gradnorm2 = self.lin_inner_prod(grad , grad)
        for i in range(maxevals):
            wn = self.lin_comb(1, w, -ss, grad)#
            Ft, _,_ = self._comp_func(wn)
            if Ft > F0 - c1 * ss* gradnorm2:
                # print('backtracking now ...')
                ss *= beta
            else:
                break
        if (i+1) == maxevals:
            print('=========> F0=%.7e'%F0)
            print('=========> Ft=%.7e'%Ft)
            print('=========> gradnorm2=%.3e'%gradnorm2)
            print('linesearch failed')
        grad_t, info = self._comp_grad(wn)
        return wn, {'stepsize':ss, 'Fval':Ft, 'grad': grad_t, \
                    'A0': info0['At'], 'At': info['At'], \
                    'res_vec0': info0['res_vec'], \
                    'res_vec': info['res_vec'], \
                    'grad0': grad, 'nevals':i+1, 'is_succ': (i+1<maxevals)}


    def solver_gd_linesearch(self, z_init,  s_init=1e-3):
        """ Gradient descent for solving the primal problem
            alpha     :  dual parameter
        """
        alpha = self.alpha
        x = z_init

        # 1st iteration

        iterhist = []
        self.stat = self._comp_iterhist(x, time=0, niter=0)
        iterhist.append(self.stat.copy())

        for i in range(self.maxiter):
            t0 = timer()
            # _dir_desc   = self.lin_comb(-1/alpha, grad_f, -1, grad_h)
            x, aux = self._gd_linesearch(x )
            self.iterdb = {'z': x, 'grad':aux['grad'], \
                           'dir_desc': self.scalar_times(-1, aux['grad']), \
                           'stepsize': aux['stepsize'], \
                           'A':aux['At'], \
                           'res_vec': aux['res_vec'], 'iter': i+1}
            ti = timer() - t0
            # Get iter stats
            if (i+1) % 1 == 0: #or abs((i+1)-maxiter)<22:
                self.stat = self._comp_iterhist(x, time=ti, niter=i+1,
                        stepsize=aux['stepsize'], alpha=alpha,\
                        verbo=2)
            else:
                self.stat = self._comp_iterhist(x, time=ti, niter=i+1,
                        stepsize=aux['stepsize'], alpha=alpha,\
                        verbo=1)
            iterhist.append(self.stat.copy())

            if (i+1) % 1 == 0:
                print("%d| f: %.5e | alpha: %.2e | h: %.5e | gradn: %.3e | s: %.2e| %.3e (sec)" % (i+1, self.stat['f_residual'], \
                        alpha, self.stat['hval'], self.stat['gradnorm'], \
                        aux['stepsize'], ti))
            if self.stat['gradnorm'] <= 1e-20: #
                break
        iterh = pd.DataFrame(iterhist)
        return x, iterh


class LoramProxDAG(Loram):
    """
    Parameters:
    -------------
     d (int)           : number of nodes
     k (int)           : number of latent dimensions
     Z0 (2darray)      : input matrix of the proximal mapping
                            argmin_B h(B) + alpha/2 |B - Z0|^2
     alpha             : parameter of

    Returns:
    ---------------
    z :     Point in the product input space of LoRAM
    Zsol :  DAG candidate matrix via the mapping LoRAM(z)
    """

    def __init__(self, Z0, k, maxiter=500, alpha=1.0, tol_h=1e-9,
                tol_grad=1e-9):
        d = Z0.shape[0]
        super().__init__(d,k)
        self.Z0 = Z0
        self.maxiter = maxiter
        self.alpha = alpha
        self.tol_h = tol_h
        self.tol_grad = tol_grad
        self.z = {'X':np.ones((self.d, self.k)), \
                  'Y': np.ones((self.d, self.k)) }
        # Update index set
        candidate_set = self._get_I_J_zref(Z0)
        self.I, self.J = candidate_set['I'], candidate_set['J']
        self.iterdb = []
        self.iterdb_old = []

    def _comp_grad_f_precomp(self, sp_zvec):
        """
        Compute the gradient of the residual = |po(B - Z0)|^2
        """
        res_vec = sp_zvec - self.Z0[self.I, self.J]
        Mat = csc_matrix((res_vec, (self.I, self.J)), shape=(self.d, self.d))
        df_x =  Mat.dot(self.z['Y'])
        df_y = (Mat.T).dot(self.z['X'])
        return {'X': df_x, 'Y': df_y}, res_vec
    def _comp_grad_f_h(self, z):
        grad_h, At, dA, sp_zvec = self._comp_grad_h_inexact1(z)
        grad_f, res_vec = self._comp_grad_f_precomp(sp_zvec)
        return grad_f, grad_h, sp_zvec, At, dA, res_vec
    def _comp_func(self,z):
        hval        = self._comp_naive_func_h(z)
        _, _, sp_zvec = splr_sigma_abs(z['X'], z['Y'], self.I, self.J)
        res_vec = sp_zvec - self.Z0[self.I, self.J]
        fval = 0.5 * (res_vec **2).sum()
        return fval/self.alpha + hval, hval, fval
    def _comp_grad(self, z):
        grad_h, At, dA, sp_zvec = self._comp_grad_h_inexact1(z)
        grad_f, res_vec = self._comp_grad_f_precomp(sp_zvec)
        return self.lin_comb(1/self.alpha, grad_f, 1, grad_h), \
                {'At':At,'sp_zvec':sp_zvec, 'res_vec':res_vec}


    def run_projdag(self, alpha=None, scale=1e-2, \
                        maxiter=None, tol_h=None, \
                        tol_grad=None):
        if alpha is not None:
            self.alpha = alpha
        if maxiter is not None:
            self.maxiter = maxiter
        if tol_h is not None:
            self.tol_h = tol_h
        if tol_grad is not None:
            self.tol_grad = tol_grad

        """ Initialize the factor matrices"""
        z_ = self.init_factors_gaussian(scale=scale)
        self.Zinit = np.asarray(self.get_adj_matrix(z_).todense())
        """ Start iterations"""
        # x_sol, iterhist = self.solve_agd(z_)
        x_sol, iterhist = self.solver_primal_accGD(z_)
        return iterhist, x_sol


    """OTHER FUNCTIONS
    """
    def _comp_iterhist(self, xnew, time=0, niter=0, \
                        stat=None, stepsize=None, alpha=None, \
                        xold = None,
                        verbo=2):
        """
        A function to compute the total mean square error
        """
        if stat is None:
           stat = {'iter': 0, 'time': 0, 'Fval': np.nan, \
                         'f_residual': np.nan, \
                         'hval': np.nan, \
                         'gradnorm': np.nan, \
                         'Anorm': np.nan, 'stepsize':np.nan, \
                         'agd_dx': np.nan, \
                         }
        iterdb = self.iterdb
        tt = stat['time'] + time
        stat['time'] =  tt
        stat['iter'] =  niter
        stat['gradnorm']   = self.lin_tspace_norm(iterdb['dir_desc'])
        stat['stepsize']   = stepsize
        stat['Anorm']      = np.linalg.norm(iterdb['A'].todense())
        stat['nnz'] =  (abs(iterdb['A'])>0).sum()
        # if x_old is None:
        #     xold = self.iterdb_old['z']
        # stat['agd_dx'] = self.lin_tspace_norm(self.lin_comb(1,xnew, -1, xold))
        if verbo == 2:
            F, hval, fval = self._comp_func(xnew)
            stat['Fval']       = F
            stat['f_residual'] = fval
            stat['hval']       = hval
        return stat

class LoramICDecomp(Loram):
    """
    Parameters:
    -------------
     d (int)           : number of nodes
     k (int)           : number of latent dimensions
     S (2darray)       : inverse covariance matrix of the ic-decomp problem
                            argmin_B h(B) + (1/2 alpha) |S - phi(B)|^2
     alpha             : parameter of

    Returns:
    ---------------
    z :     Point in the product input space of LoRAM
    Zsol :  DAG candidate matrix via the mapping LoRAM(z)
    """

    def __init__(self, S, k, maxiter=500, alpha=1.0, sigma0=1.0, \
                    tol_h=1e-9, tol_grad=1e-9):
        d = S.shape[0]
        super().__init__(d,k)
        self.S = S
        self.maxiter = maxiter
        self.alpha = alpha
        self.sigma0 = sigma0
        self.tol_h = tol_h
        self.tol_grad = tol_grad
        self.z = {'X':np.ones((self.d, self.k)), \
                  'Y': np.ones((self.d, self.k)) }
        # Update index set
        candidate_set, mask = self._get_I_J_icdecomp(S)
        self.I, self.J = candidate_set['I'], candidate_set['J']
        self.mask = mask

        self.iterdb = []
        self.iterdb_old = []

    def _get_I_J_icdecomp(self, Prec):
        """ Get the index set of all edges in the reference (non-DAG) graph
        """
        # Mask must not contain diagonal
        tmp = Prec
        tmp = tmp - np.diag(np.diag(tmp))
        ind_2d = np.nonzero(tmp)
        return {'I': ind_2d[0].astype(np.uintc),
                'J': ind_2d[1].astype(np.uintc)}, (tmp != 0)

    def _comp_grad_f_precomp(self, z, sp_zvec):
        """
        Compute the gradient of the residual = (1/2) |S - phi(At))|^2
        """
        B = csc_matrix((sp_zvec, (self.I, self.J)), shape=(self.d, self.d)).toarray()
        phib = np.eye(self.d) - B - B.T + B @ B.T #
        res_vec = (self.S - phib/self.sigma0).ravel()
        # phib = phib.todense()
        smat = (2/self.sigma0) * \
                (self.S - phib/self.sigma0) @ (np.eye(self.d)-B)
        smat[self.mask == 0] = 0
        df_x =  smat @ z['Y']
        df_y = smat.T @ z['X']
        return {'X': df_x, 'Y': df_y}, res_vec
    def _comp_grad_f_h(self, z):
        grad_h, At, dA, sp_zvec = self._comp_grad_h_inexact1(z)
        grad_f, res_vec = self._comp_grad_f_precomp(z, sp_zvec)
        return grad_f, grad_h, sp_zvec, At, dA, res_vec
    def _comp_grad(self, z):
        grad_h, At, dA, sp_zvec = self._comp_grad_h_inexact1(z)
        grad_f, res_vec = self._comp_grad_f_precomp(z, sp_zvec)
        return self.lin_comb(1, grad_f, self.alpha, grad_h), \
                {'At':At,'sp_zvec':sp_zvec, 'res_vec':res_vec}
    def _comp_func(self, z, res_vec=None):
        hval        = self._comp_naive_func_h(z)
        if res_vec is None:
            _, _, sp_zvec = splr_sigma_abs(z['X'], z['Y'], self.I, self.J)
            B = csc_matrix((sp_zvec, (self.I, self.J)), shape=(self.d, self.d)).toarray()
            phib = np.eye(self.d) - B - B.T + B @ B.T #
            res_vec = (self.S - phib/self.sigma0).ravel()
        fval = 0.5 * (res_vec **2).sum()
        return fval + self.alpha * hval, hval, fval


    def run_(self, alpha=None, sigma0=None, maxiter=None, \
                        tol_h=None, tol_grad=None):
        if alpha is not None:
            self.alpha = alpha
        if sigma0 is not None:
            self.sigma0 = sigma0
        if maxiter is not None:
            self.maxiter = maxiter
        if tol_h is not None:
            self.tol_h = tol_h
        if tol_grad is not None:
            self.tol_grad = tol_grad

        """ Initialize the factor matrices"""
        z_ = self.init_factors_gaussian(scale=1e-1)
        self.Zinit = np.asarray(self.get_adj_matrix(z_).todense())
        """ Start iterations"""
        # x_sol, iterhist = self.solver_primal_accGD(z_)
        # x_sol, iterhist = self.solve_agd(z_)
        x_sol, iterhist = self.solver_gd_linesearch(z_)
        return iterhist, x_sol

    """OTHER FUNCTIONS
    """
    def _comp_iterhist(self, xnew, time=0, niter=0, \
                        stat=None, stepsize=None, alpha=None, \
                        xold = None,
                        verbo=2):
        """
        A function to compute the total mean square error
        """
        if stat is None:
           stat = {'iter': 0, 'time': 0, 'Fval': np.nan, \
                         'f_residual': np.nan, \
                         'hval': np.nan, \
                         'gradnorm': np.nan, \
                         'Anorm': np.nan, 'stepsize':np.nan, \
                         'agd_dx': np.nan, \
                         }
        iterdb = self.iterdb
        tt = stat['time'] + time
        stat['time'] =  tt
        stat['iter'] =  niter
        stat['stepsize']   = stepsize
        if niter > 0:
            stat['gradnorm']   = self.lin_tspace_norm(iterdb['dir_desc'])
            stat['Anorm']      = np.linalg.norm(iterdb['A'].todense())
            stat['nnz'] =  (abs(iterdb['A'])>0).sum()
            if verbo == 2:
                F, hval, fval = self._comp_func(xnew, res_vec=iterdb['res_vec'])
                stat['Fval']       = F
                stat['f_residual'] = fval
                stat['hval']       = hval
        else:
            stat['gradnorm']   = np.nan
            stat['Anorm']      = np.nan
            stat['nnz'] =  np.nan
        return stat


