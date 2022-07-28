import numpy as np

import matplotlib.pyplot as plt
from timeit import default_timer as timer
import time, os
import pandas as pd
import shutil

from icid import utils
from icid.icid import run_icid, AMA_independece_decomp
from external.test_linear import notears_linear
from external.altmin_linear_sem import run_altmin
from external.pyquic.py_quic_w import quic

from sklearn.covariance import GraphLassoCV, GraphLasso
from sklearn.metrics import  confusion_matrix, classification_report

def gen_data_linsem(d, n, deg, graph_type='ER', sem_type='gauss'):
    #
    spr = deg*d / d**2
    s0  = int(np.ceil(spr*d**2))
    B_true = utils.simulate_dag(d, s0, graph_type)
    W_true = utils.simulate_parameter(B_true)
    X = utils.simulate_linear_sem(W_true, n, sem_type)
    return X, W_true, B_true

def test_ice_quic(fdir,  ntests=8,  fin=None, fname=None,\
                        graphtype='ER', semtype='gauss', \
                        run_methods={'quic':1,'sklearn':0}):
    def _hard_threshold(XP, delta=5e-2):
        Xs = XP.copy()
        for j in range(Xs.shape[0]):
            X = (XP[j,:].copy()).reshape([d,d])
            diagx = np.diag(X)
            X = X - np.diag(diagx)
            cmax = max(abs(X.ravel()))
            X[abs(X)<delta*cmax] = 0
            X += np.diag(diagx)
            Xs[j,:] = X.ravel()
        return Xs
    utils.set_random_seed(1)
    inp0 = {'n' : 600, \
            'd' : 0, \
            'k' : 25, \
            'deg' : 0.3, \
            'graph_type':graphtype, 'sem_type' : semtype,\
            'lambda_1': 5e0}
    if fin is not None:
        list_    = pd.read_csv(fin)
        # columns of list_ are d, n, deg, lambda_1
        inps = []
        for i in range(len(list_)):
            inp_ = inp0.copy()
            inp_['d'] = list_['d'][i]
            inp_['n'] = list_['n'][i]
            inp_['deg'] = list_['deg'][i]
            inps.append(inp_.copy())
    else:
        # Only d varies
        d0 = 100
        inps = []
        for i in range(ntests):
            inp0['d'] += d0
            inp0['n'] = 4*inp0['d']
            inps.append(inp0.copy())
    df_inps = pd.DataFrame(inps).to_csv('%s/inputs_all.csv' %fdir)
    shutil.copy2('./test_main.py', '%s/test_main.py.txt' % fdir)
    res = []
    for i in range(len(inps)):
        print('-------Test %d/%d------' %(i+1,len(inps)))
        n,d,k,deg,graph_type,sem_type,lambda_1 = inps[i]['n'], \
                                        inps[i]['d'],\
                                        inps[i]['k'],\
                                        inps[i]['deg'],\
                                        inps[i]['graph_type'],\
                                        inps[i]['sem_type'],\
                                        inps[i]['lambda_1']
        if 'run_pls' in list_.keys():
            run_pls,run_icd,run_no=list_['run_pls'][i],\
                               list_['run_icd'][i], \
                               list_['run_notears'][i]
        else:
            run_pls, run_icd, run_no = 1,1,1
        spr = deg*d / d**2
        s0 = int(np.ceil(spr*d**2))
        print('d= %d' % d)
        print('deg = %f' % deg)
        print(graph_type)
        print('lambda_1 init= %f' % lambda_1)
        B_true = utils.simulate_dag(d, s0, graph_type)
        if graph_type is 'SF':
            # In the case of scale-free (SF) graphs, hubs are mostly causes,
            # rather than effects, of its neighbors
            B_true = B_true.T
        W_true = utils.simulate_parameter(B_true)
        X = utils.simulate_linear_sem(W_true, n, sem_type)
        sigma_0 = 1
        P_true  = (np.eye(d)-W_true) @ (np.eye(d)-W_true).T  / sigma_0

        # --------------- Alg 1: QUIC
        n_samples = X.shape[0]
        X = X - np.mean(X, axis=0, keepdims=True)
        print("IC using naive matrix inversion.. ")
        emp_cov = np.dot(X.T, X) / n_samples
        #       Run in "path" mode
        #       path = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5 ])
        path = np.linspace(1.0, 0.1, 8)
        XP, WP, optP, cputimeP, iterP, dGapP = quic(S=emp_cov, L=lambda_1,
                mode="path", path=path, tol=1e-16, max_iter=100, msg=1)
        #------------------- Alt 2: sklearn
        # Estimate the covariance and sparse inverse covariance
        XP_sk = np.zeros(XP.shape)
        tsk = 0
        fail_sk = False
        for j in range(XP.shape[0]):
            t0 = timer()
            model = GraphLasso(alpha=lambda_1*path[j])
            try:
                model.fit(X)
                cov_ = model.covariance_
                prec_ = model.precision_
                tsk += timer() - t0
                XP_sk[j,:] = prec_.ravel()
            except FloatingPointError:
                print("Oops!  Alpha value not working for glasso (sklearn).. ")
                model = []
                XP_sk[j,:] = XP[j,:]
                tsk += cputimeP[j]
                fail_sk = True
        #-------------- Evaluate scores
        for j in range(XP.shape[0]):
            print('L value = %f' % path[j])
            XP = _hard_threshold(XP, delta=1e-1)
            sp = (XP[j,:] !=0).sum() / d**2
            sp_sk = (XP_sk[j,:] !=0).sum() / d**2
            print('sparsity of solutions is %f / %f' %(sp,sp_sk))
            # acc_sc = accuracy_score( (P_true.ravel())!=0, XP[j,:] !=0)
            # acc_sc_sk = accuracy_score( (P_true.ravel())!=0, XP_sk[j,:] !=0)
            tn, fp, fn, tp = confusion_matrix((P_true.ravel())!=0, XP[j,:] !=0).ravel()
            tn_, fp_, fn_, tp_ = confusion_matrix((P_true.ravel())!=0, XP_sk[j,:] !=0).ravel()
            tpr = tp / (tp+fn)
            fpr = fp / (fp+tn)
            tpr_sk = tp_ / (tp_+fn_)
            fpr_sk = fp_ / (fp_+tn_)

            print('TPR are %f / %f' % (tpr, tpr_sk) )
            print('FPR are %f / %f' % (fpr, fpr_sk) )
            res_cl = classification_report( (P_true.ravel())!=0, XP[j,:] !=0)
            res_cl_sk = classification_report( (P_true.ravel())!=0, XP_sk[j,:] !=0)
            print(res_cl)
            print('sklearn cla report:')
            print(res_cl_sk)
            res.append({'d':d,'deg': deg, 'n':n, 'graph_type': graph_type,
                'alg':'QUIC', 'lambda_1': lambda_1 * path[j],
                    'tpr':tpr, 'fpr':fpr, 'nnz':sp*d**2,
                    'time': cputimeP.sum(),
                     'fail_sk': 'na'})
            res.append({'d':d,'deg': deg, 'n':n, 'graph_type': graph_type,
                       'alg':'sklearn', 'lambda_1': lambda_1 * path[j],
                    'tpr':tpr_sk, 'fpr':fpr_sk, 'nnz':sp_sk*d**2,
                    'time': tsk,
                     'fail_sk': fail_sk})
    df_res = pd.DataFrame(res).to_csv('%s/res_all.csv' %fdir)


def test_linsem_altm_v_notears(fdir,  ntests=8,  fin=None, fname=None,\
                        graphtype='ER', semtype='gauss', \
                        run_methods={'icd':1,'pls':0,'notears':0}):
    utils.set_random_seed(1)
    inp0 = {'n' : 600, \
            'd' : 0, \
            'k' : 25, \
            'deg' : 0.3, \
            'graph_type':graphtype, 'sem_type' : semtype,\
            'lambda_1': 2e-1}
    if fin is not None:
        list_    = pd.read_csv(fin)
        # columns of list_ are d, n, deg, lambda_1
        inps = []
        for i in range(len(list_)):
            inp_ = inp0.copy()
            inp_['d'] = list_['d'][i]
            inp_['n'] = list_['n'][i]
            inp_['deg'] = list_['deg'][i]
            # inp_['lambda_1'] = list_['lambda1'][i]
            inps.append(inp_.copy())
    else:
        # Only d varies
        d0 = 100
        inps = []
        for i in range(ntests):
            inp0['d'] += d0
            inp0['n'] = 4*inp0['d']
            inps.append(inp0.copy())
    df_inps = pd.DataFrame(inps).to_csv('%s/inputs_all.csv' %fdir)
    shutil.copy2('./test_main.py', '%s/test_main.py.txt' % fdir)
    res = []
    for i in range(len(inps)):
        print('-------Test %d/%d------' %(i+1,len(inps)))
        n,d,k,deg,graph_type,sem_type,lambda_1 = inps[i]['n'], \
                                        inps[i]['d'],\
                                        inps[i]['k'],\
                                        inps[i]['deg'],\
                                        inps[i]['graph_type'],\
                                        inps[i]['sem_type'],\
                                        inps[i]['lambda_1']
        if 'run_pls' in list_.keys():
            run_pls,run_icd,run_no=list_['run_pls'][i],\
                               list_['run_icd'][i], \
                               list_['run_notears'][i]
        else:
            run_pls, run_icd, run_no = 1,1,1
        spr = deg*d / d**2
        s0 = int(np.ceil(spr*d**2))
        print(d)
        print(s0)
        print(graph_type)
        print(lambda_1)
        B_true = utils.simulate_dag(d, s0, graph_type)
        B_true = B_true.T # In the case of scale-free (SF) graphs, hubs are mostly causes, rather than effects, of its neighbors
        W_true = utils.simulate_parameter(B_true)
        X = utils.simulate_linear_sem(W_true, n, sem_type)
        tolh = 1e-7
        """ Alg 1-----PLS-Loram-Altmin """
        if run_methods['pls'] and run_pls:
            C0 = 300
            print('-------preparation ok, ready to run LoRAM-Altmin---')
            W_est, Aref, ith_all, ith2, Winit, stats = \
                                          run_altmin(X.T, lambda1=lambda_1, k=k, \
                                                     c0=C0, Wtrue=W_true.T,\
                                                     tol_h=tolh, eps_d=1e-8,\
                                                     thres=5e-2, max_iter=40,\
                                                     beta_1=0.3, beta_2=0.5,\
                                                     gamma_2=1.0)
            """ Evaluations and record results """
            Cmax = max(abs(W_est.ravel()))
            W_est[abs(W_est) <= 9e-2 ] = 0
            acc = utils.count_accuracy(B_true.T, W_est != 0)
            print(acc)
            res.append({'d':d, 'alg':'ours', \
                    'deg': deg, 'n':n, 'graph_type': graph_type,\
                'shd':acc['shd'], 'tpr':acc['tpr'], 'fdr':acc['fdr'], 'fpr':acc['fpr'], 'nnz':acc['nnz'], 'time': ith_all[-1]['time']})
            """ SAVE RESULTS """
            pd.DataFrame(ith_all).to_csv('%s/tid%d_ith_pls.csv' %(fdir,i))

        """ Alg 3-----ICD-Loram-Altmin """
        if run_methods['icd'] and run_icd:
            print('-------Ready to run ICD-LoRAM-Altmin---')
            tols = [2e-1] #[1e-1, 2e-1, 3e-1]
            itol = 0
            # lambda_1 = 1e-1
            for tol in tols:
                itol += 1
                W_icd, ith_icd  = run_icid(X, lambda_1=lambda_1,
                                            sigma_0=1.0, k=k,
                                            beta_2 = 0.7, tol_prec=tol, \
                                           gamma_2=1.0, maxit_prox_inner=500, W_true=W_true)

                """ Evaluations and record results """
                # Cmax = max(abs(W_icd.ravel()))
                # W_icd[abs(W_icd) <= 9e-2 ] = 0
                acc = utils.count_accuracy(B_true, W_icd != 0)
                print(acc)
                res.append({'d':d, 'alg':'ICD (ours)',
                        'deg': deg, 'n':n, 'graph_type': graph_type,
                        'tol_prec': tol, \
                    'shd':acc['shd'],
                    'tpr':acc['tpr'], 'fdr':acc['fdr'], 'fpr':acc['fpr'],
                    'nnz':acc['nnz'], 'time': ith_icd.iloc[-1]['time']})
                """ SAVE RESULTS """
                pd.DataFrame(ith_icd).to_csv('%s/tid%d_tol%d_ith_icd.csv'\
                                              %(fdir,i,itol))

        """ Alg 2------NOTEARS """
        if run_methods['notears'] and run_no:
            print('-------Test %d/%d------NoTears' %(i+1,len(inps)))
            t1 = timer()
            West_no, ith_no = notears_linear(X, lambda1=lambda_1, \
                                        h_tol=tolh, loss_type='l2', \
                                        Wtrue=W_true)
            t1 = timer() - t1
            acc_no = utils.count_accuracy(B_true != 0, West_no != 0)
            print('NOTEARS-----results in %.2e (seconds):'% t1)
            print(acc_no)
            res.append({'d':d, 'alg':'NoTears',
                    'deg': deg, 'n':n, 'graph_type': graph_type,
                'shd':acc_no['shd'], 'tpr':acc_no['tpr'], 'fdr':acc_no['fdr'], 'fpr':acc_no['fpr'], 'nnz':acc_no['nnz'], 'time':ith_no[-1]['time']})
            """ SAVE RESULTS """
            pd.DataFrame(ith_no).to_csv('%s/tid%d_ith_notears.csv' %(fdir,i))
    df_res = pd.DataFrame(res).to_csv('%s/res_all.csv' %fdir)

def test_icdecomp_altmin(X, alpha, sigma_0=1.0, lambda_1=1e-1, k=25,
        gamma_2=1.0, miter_2=500, W_true=None, fdir=None, tid=None, fname=None):
    res = []
    n,d = X.shape
    deg = (W_true !=0).sum() / d
    Cm0 = max(abs(W_true.ravel()))

    P_true  = (np.eye(d)-W_true) @ (np.eye(d)-W_true).T  / sigma_0
    #
    print('\n==============================\n')
    print('Test with d:%d, n:%d, deg: %f' %(d,n,deg))
    acc = utils.count_accuracy((W_true)!=0, P_true !=0)
    stats = {'matrix_name': 'Prec_true', 'd': d, 'n':n, 'deg': deg, 'shd': acc['shd'], 'tpr':acc['tpr'],'fdr':acc['fdr'],'fpr':acc['fpr'], 'tpr':acc['tpr'], 'nnz':acc['nnz'], 'alpha_glasso':np.nan, 'lambda1':np.nan, 'sigma0':np.nan, 'gamma_2':np.nan, 'runtime': np.nan, 'runtime_pdecomp':np.nan}
    acc = utils.count_accuracy((W_true)!=0, P_true !=0)
    res.append(stats.copy())
    # Inverse covariance estimation
    t0 = timer()
    prec_est, cov_est, model = get_sparse_ice(X, alpha=alpha, fname='deg%f' % deg)
    tg = timer() - t0
    #
    acc = utils.count_accuracy((W_true)!=0, prec_est !=0)
    stats['matrix_name'], stats['shd'], stats['tpr'], stats['fdr'], stats['fpr'], stats['nnz'],  stats['alpha_glasso'],  stats['runtime'] = 'Prec_GLasso', acc['shd'], acc['tpr'], acc['fdr'], acc['fpr'], acc['nnz'], alpha, tg
    res.append(stats.copy())
    plot_wflatten_overlap(prec_est[:20,:20], P_true[:20,:20], fdir=fdir,
            fname='prec_est', flags={'w':'Prec', 'wref':'True Prec'})
    acc = utils.count_accuracy((P_true)!=0, prec_est !=0)
    print('=====Acc of prec_est vs Prec-true:')
    print(acc)
    # ICDecomp and Loram-AltMin
    t0 = timer()
    wnew, iterh = AMA_independece_decomp(prec_est, k=25, W_true = W_true,\
                                        epsilon=1e-2)
    tf = timer() - t0
    #
    tt = iterh.iloc[-1]['time']
    Cm = max(abs(wnew.ravel()))
    wnew[abs(wnew)<5e-2*Cm] = 0
    acc = utils.count_accuracy(W_true !=0, wnew !=0)
    stats['matrix_name'], stats['shd'], stats['tpr'], stats['fdr'],\
    stats['fpr'], stats['nnz'], stats['lambda1'], stats['sigma0'],\
    stats['runtime'] = 'DAG_LoramAltMin', acc['shd'], acc['tpr'], acc['fdr'], acc['fpr'], acc['nnz'], lambda_1, sigma_0, tt
    res.append(stats.copy())
    print('alpha=%0.2e | lambda1=%.2e | sigma_0=%.2e | Iterhist of FISTA:' % (alpha, lambda_1, sigma_0))
    # print(iterh)
    Cm = max(abs(wnew.ravel()))
    plot_wflatten_overlap(Cm0*wnew/Cm, W_true, fdir=fdir, fname='%d_preDAG'%tid, flags={'w':'W', 'wref':'True DAG'})
    print('\n==============================\n')
    print('All results:')
    df_res=pd.DataFrame(res, columns=res[0].keys())
    print(df_res)
    df_res.to_csv('%s/all_res_deg%d.csv' %(fdir,tid))
    iterh.to_csv('%s/iterh_loramaltmin_deg%d.csv' %(fdir,tid))

def get_sparse_ice(X, alpha=1e-2, fdir=None, fname=None):
    # Source: https://scikit-learn.org/stable/auto_examples/covariance/plot_sparse_cov.html
    if fdir is None:
        fdir = 'outputs/misc'
    if not os.path.exists(fdir):
        os.makedirs(fdir)
    n_samples = X.shape[0]
    # Estimate the covariance and sparse inverse covariance
    emp_cov = np.dot(X.T, X) / n_samples

    model = GraphLasso(alpha=alpha)
    try:
        model.fit(X)
        cov_ = model.covariance_
        prec_ = model.precision_
    except FloatingPointError:
        print("Oops!  Alpha value not working for glasso (sklearn).. ")
        cov_ = emp_cov
        prec_ = np.linalg.inv(emp_cov)
        cmax = max(abs(prec_.ravel()))
        prec_[abs(prec_)<1e-1 * cmax] = 0
    return prec_, cov_, model

def get_sparse_ice_plot(X, cov_true=None, prec_true=None, dag=None, fdir=None, fname=None):
    # Source: https://scikit-learn.org/stable/auto_examples/covariance/plot_sparse_cov.html
    if fdir is None:
        fdir = 'outputs/misc'
    if not os.path.exists(fdir):
        os.makedirs(fdir)
    n_samples = X.shape[0]
    # Estimate the covariance and sparse inverse covariance
    emp_cov = np.dot(X.T, X) / n_samples

    model = GraphLassoCV()
    model.fit(X)
    cov_ = model.covariance_
    prec_ = model.precision_

    # Plot the results
    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(left=0.02, right=0.98)

    # plot the covariances
    covs = [
        ("Empirical", emp_cov),
        ("GraphLassoCV", cov_),
        ("True", cov_true),
    ]
    vmax = cov_.max()
    for i, (name, this_cov) in enumerate(covs):
        plt.subplot(1, 3, i + 1)
        plt.imshow(
            this_cov, interpolation="nearest", vmin=-vmax, vmax=vmax,
            # cmap=plt.cm.RdBu_r
        )
        plt.xticks(())
        plt.yticks(())
        plt.title("%s" % name)
    plt.savefig('%s/%s_%s.pdf' % (fdir, fname, 'covariance_sklearn'), transparent=True, bbox_inches = 'tight')
    plt.close()
    emp_prec = np.linalg.inv(emp_cov)
    # plot the precisions
    plt.rcParams.update({'font.size': 12})
    precs = [
        ("Empirical", emp_prec),
        ("GraphLasso", prec_),
        ("True", prec_true),
        ("DAG", dag),
    ]
    vmax = 0.9 * prec_.max()
    for i, (name, this_prec) in enumerate(precs):
        ax = plt.subplot(1, len(precs), i + 1)
        plt.imshow(
            np.ma.masked_equal(this_prec, 0),
            # interpolation="nearest",
            vmin=-vmax,
            vmax=vmax,
            # cmap=plt.cm.RdBu_r,
        )
        plt.xticks(())
        plt.yticks(())
        plt.title("%s" % name)
        if hasattr(ax, "set_facecolor"):
            ax.set_facecolor(".7")
        else:
            ax.set_axis_bgcolor(".7")
    plt.savefig('%s/%s_%s.pdf' % (fdir, fname, 'ice_sparse_sklearn'), transparent=True, bbox_inches = 'tight')
    plt.close()
    # flattened matrix overlap
    plt.figure(figsize=(21,3))
    plt.plot(emp_prec.flatten(),'r', linewidth=2,label='Empirical')
    plt.plot(prec_true.flatten(), 'b-', linewidth=2 , alpha=0.4, label='Prec true')
    plt.legend(loc='center left', bbox_to_anchor= (1.04, 0.5), ncol=1, borderaxespad=0, frameon=False)
    plt.savefig('%s/%s_prec_overlap_emp.pdf' % (fdir,fname), format="pdf",transparent=True, bbox_inches = 'tight')
    plt.close()

    plt.figure(figsize=(21,3))
    plt.plot(prec_.flatten(),'r', linewidth=2,label='GLasso')
    plt.plot(prec_true.flatten(), 'b-', linewidth=2 , alpha=0.4, label='Prec true')
    plt.legend(loc='center left', bbox_to_anchor= (1.04, 0.5), ncol=1, borderaxespad=0, frameon=False)
    plt.savefig('%s/%s_prec_overlap_gl.pdf' % (fdir,fname), format="pdf",transparent=True, bbox_inches = 'tight')
    plt.close()

    plt.figure(figsize=(21,3))
    plt.plot(prec_.flatten(),'r', linewidth=2,label='GLasso')
    plt.plot(dag.flatten(), 'b-', linewidth=2 , alpha=0.4, label='DAG true')
    plt.legend(loc='center left', bbox_to_anchor= (1.04, 0.5), ncol=1, borderaxespad=0, frameon=False)
    plt.savefig('%s/%s_dag_overlap_gl.pdf' % (fdir,fname), format="pdf",transparent=True, bbox_inches = 'tight')
    plt.close()

    plt.figure(figsize=(21,3))
    plt.plot(emp_prec.flatten(),'r', linewidth=2,label='Empirical')
    plt.plot(dag.flatten(), 'b-', linewidth=2 , alpha=0.4, label='DAG true')
    plt.legend(loc='center left', bbox_to_anchor= (1.04, 0.5), ncol=1, borderaxespad=0, frameon=False)
    plt.savefig('%s/%s_dag_overlap_emp.pdf' % (fdir,fname), format="pdf",transparent=True, bbox_inches = 'tight')
    plt.close()
    return cov_, prec_

if __name__ == '__main__':
    if False:
        # ---------------test 7b
        d = 100
        n = 8*d
        timestr = time.strftime("%H%M%S%m%d")
        fdir = '../outputs/glasso_loramicd_%s' % timestr
        if not os.path.exists(fdir):
            os.makedirs(fdir)
        degs= [0.3, 0.6, 0.9]
        tid=0
        for deg in degs:
            tid += 1
            spr = deg*d / d**2
            s0, graph_type, sem_type = int(np.ceil(spr*d**2)), 'ER', 'gauss'
            B_true = utils.simulate_dag(d, s0, graph_type)
            W_true = utils.simulate_parameter(B_true)
            X = utils.simulate_linear_sem(W_true, n, sem_type)
            X = X - np.mean(X, axis=0, keepdims=True)
            test_loram_icdecomp(X, alpha=2e-1, gamma_2=1e-5, W_true = W_true, fdir=fdir, tid=tid)

    if True:
        # test 5b (ER)
        timestr = time.strftime("%H%M%S%m%d")
        # fin = 'aux/list_d_n_deg.csv'
        fin = 'aux/list_dndeg_2.csv'
        fout = '../outputs/linsem_oursvnotears_d_%s' % timestr
        if not os.path.exists(fout):
            os.makedirs(fout)
        test_linsem_altm_v_notears(fout, fin=fin, \
                graphtype='ER', \
                run_methods={'icd':1,'pls':0,'notears':0})

    if False:
        # test 5c (ER): evaluate quic for ice
        timestr = time.strftime("%H%M%S%m%d")
        fin = 'aux/list_dndeg_2b.csv'
        fout = '../outputs/ice_quic_d_%s' % timestr
        if not os.path.exists(fout):
            os.makedirs(fout)
        test_ice_quic(fout, fin=fin, \
                graphtype='ER', \
                run_methods={'quic':1,'sklearn':0})

    if False:
        # test 5b (SF)
        timestr = time.strftime("%H%M%S%m%d")
        fin = 'aux/list_dndeg_3.csv'
        fout = '../outputs/linsem_oursvnotears_d_%s' % timestr
        if not os.path.exists(fout):
            os.makedirs(fout)
        test_linsem_altm_v_notears(fout, fin=fin, \
                graphtype='SF', \
                run_methods={'icd':1,'pls':1,'notears':1})


