import numpy as np

from timeit import default_timer as timer
import time, os, sys
import pandas as pd
import shutil

from aux.dag_utils import get_data_sachs
from icid import utils

import tabulate
# from sklearn.grid_search import GridSearchCV
from sklearn.datasets import make_sparse_spd_matrix
from sklearn.covariance import GraphicalLasso, GraphicalLassoCV, ledoit_wolf
import matplotlib.pyplot as plt
from inverse_covariance import (
    QuicGraphicalLasso,
    QuicGraphicalLassoCV,
    QuicGraphicalLassoEBIC,
    AdaptiveGraphicalLasso,
    ModelAverage,
)


def r_input(val):
    if sys.version_info[0] >= 3:
        return eval(input(val))

    return raw_input(val)

def multiplot(named_mats, suptitle):
    num_rows = len(named_mats) / 3
    num_plots = int(np.ceil(num_rows / 4.))
    for nn in range(num_plots):
        plt.figure(figsize=(10, 8))
        plt.subplots_adjust(left=0.02, right=0.98, hspace=0.4)
        for i, item in enumerate(named_mats[nn * 4 * 3 : (nn + 1) * 4 * 3]):
            lam = None
            if len(item) == 3:
                name, this_mat, lam = item
            elif len(item) == 2:
                name, this_mat = item

            vmax = np.abs(this_mat).max()
            ax = plt.subplot(4, 3, i + 1)
            plt.imshow(
                np.ma.masked_values(this_mat, 0),
                interpolation="nearest",
                vmin=-vmax,
                vmax=vmax,
                cmap=plt.cm.RdBu_r,
            )
            plt.xticks(())
            plt.yticks(())
            if lam is None or lam == "":
                plt.title("{}".format(name))
            else:
                plt.title("{}\n(lam={:.2f})".format(name, lam))

        plt.suptitle(suptitle + " (page {})".format(nn), fontsize=14)
        plt.show()


def show_results(covs, precs):
    multiplot(covs, "Covariance Estimates")
    multiplot(precs, "Precision Estimates")

# def quic_graph_lasso(X, num_folds, metric):
#     """Run QuicGraphicalLasso with mode='default' and use standard scikit
#     GridSearchCV to find the best lambda.
#     Primarily demonstrates compatibility with existing scikit tooling.
#     """
#     print("QuicGraphicalLasso + GridSearchCV with:")
#     print("   metric: {}".format(metric))
#     search_grid = {
#         "lam": np.logspace(np.log10(0.01), np.log10(1.0), num=100, endpoint=True),
#         "init_method": ["cov"],
#         "score_metric": [metric],
#     }
#     model = GridSearchCV(QuicGraphicalLasso(), search_grid, cv=num_folds, refit=True)
#     model.fit(X)
#     bmodel = model.best_estimator_
#     print("   len(cv_lams): {}".format(len(search_grid["lam"])))
#     print("   cv-lam: {}".format(model.best_params_["lam"]))
#     print("   lam_scale_: {}".format(bmodel.lam_scale_))
#     print("   lam_: {}".format(bmodel.lam_))
#     return bmodel.covariance_, bmodel.precision_, bmodel.lam_

def sp_ice_naive(X, lambda_1):
    # Estimate the covariance and sparse inverse covariance
    n_samples = X.shape[0]
    X = X - np.mean(X, axis=0, keepdims=True)
    # print("IC using naive matrix inversion.. ")
    emp_cov = np.dot(X.T, X) / n_samples
    cov_ = emp_cov
    prec_ = np.linalg.inv(emp_cov)
    prec_off = prec_.copy()
    prec_off = prec_off - np.diag(np.diag(prec_off))
    cmax = max(abs(prec_off.ravel()))
    #
    # Weps = - prec_off.copy()
    # Weps[abs(prec_off) > lambda_1 * cmax] = 0
    # Weps[Weps<0] = 0
    # dD = np.sum(Weps, axis=0)
    prec_off[abs(prec_off) < lambda_1 * cmax] = 0
    # correction step for #
    prec_sp = prec_off + np.diag(np.diag(prec_))
    # prec_sp = prec_off + np.diag(np.diag(prec_)) - np.diag(dD)
    cov_o = np.linalg.inv(prec_sp)
    return prec_sp, cov_o

def quic_graph_lasso_cv(X, metric):
    """Run QuicGraphicalLassoCV on data with metric of choice.
    Compare results with GridSearchCV + quic_graph_lasso.  The number of
    lambdas tested should be much lower with similar final lam_ selected.
    """
    print("QuicGraphicalLassoCV with:")
    print("   metric: {}".format(metric))
    model = QuicGraphicalLassoCV(
        cv=2,  # cant deal w more folds at small size
        n_refinements=6,
        n_jobs=1,
        init_method="cov",
        score_metric=metric,
    )
    model.fit(X)
    print("   len(cv_lams): {}".format(len(model.cv_lams_)))
    print("   lam_scale_: {}".format(model.lam_scale_))
    print("   lam_: {}".format(model.lam_))
    return model.covariance_, model.precision_, model.lam_

def empirical(X):
    """Compute empirical covariance as baseline estimator.
    """
    print("Empirical")
    cov = np.dot(X.T, X) / n_samples
    return cov, np.linalg.inv(cov)

def graph_lasso_cv(X, num_folds):
    """Estimate inverse covariance via scikit-learn GraphLassoCV class.
    """
    print("GraphLasso (sklearn)")
    model = GraphicalLassoCV(cv=num_folds)
    model.fit(X)
    print("   lam_: {}".format(model.alpha_))
    return model.covariance_, model.precision_, model.alpha_

def graph_lasso(X, lam):
    """Estimate inverse covariance via scikit-learn GraphLasso class.
    """
    print("GraphLasso (sklearn)")
    model = GraphicalLasso(alpha=lam)
    try:
        model.fit(X)
        cov_ = model.covariance_
        prec_ = model.precision_
    except FloatingPointError:
        print("Oops!  Alpha value not working for glasso (sklearn).. ")
        cov_, prec_ = None, None
    print("   lam_: {}".format(lam))
    return cov_, prec_, lam

def sk_ledoit_wolf(X):
    """Estimate inverse covariance via scikit-learn ledoit_wolf function.
    """
    print("Ledoit-Wolf (sklearn)")
    lw_cov_, _ = ledoit_wolf(X)
    lw_prec_ = np.linalg.inv(lw_cov_)
    return lw_cov_, lw_prec_

def _count_support_diff(m, m_hat):
    n_features, _ = m.shape

    m_no_diag = m.copy()
    m_no_diag[np.diag_indices(n_features)] = 0
    m_hat_no_diag = m_hat.copy()
    m_hat_no_diag[np.diag_indices(n_features)] = 0

    m_nnz = len(np.nonzero(m_no_diag.flat)[0])
    m_hat_nnz = len(np.nonzero(m_hat_no_diag.flat)[0])

    nnz_intersect = len(
        np.intersect1d(np.nonzero(m_no_diag.flat)[0], np.nonzero(m_hat_no_diag.flat)[0])
    )
    return m_nnz + m_hat_nnz - (2 * nnz_intersect)


def _selection(err_emps, fs, supp_diffs, nnzs):
    N = len(err_emps) //2
    c1 = np.argsort(err_emps.values)[:N]
    # c2 = np.argsort(abs(fs.values))[:N]
    c2 = np.argsort(fs.values)[:N]
    # c2 = np.argsort(5e-2*err_emps.values
    #                +fs.values)[:N]

    # Find the most sparse one among c2
    nnzvals = nnzs.values[c2]
    isel = np.argsort(nnzs.values[c2])[N//2]
    sel = c2[isel]
    idx = np.argsort(supp_diffs.values)[0]
    return c1,c2, nnzvals, sel, idx

def _criterion_trace(prec, prec_true, cov_true, cov_emp):
    #f = (prec * cov_emp).sum() / prec.shape[0]
    # f = (prec * cov_emp).sum() / (prec!=0).sum()
    # f = (prec * cov_emp).sum() - sign*logdet

    f = (prec * cov_emp).sum() # good already
    prec_ = prec + np.diag(9e-1*np.diag(prec))
    f = (prec * cov_emp).sum() - np.log(np.linalg.det(prec_))
    return f

def _comp_stats(prec, cov, prec_true, cov_true, cov_emp):
    error = np.linalg.norm(cov_true - cov, ord="fro")
    error_emp = np.linalg.norm(cov_emp - cov, ord="fro")
    supp_diff = _count_support_diff(prec_true, prec)
    nnz = (prec!=0).sum()
    nnz_true = (prec_true!=0).sum()
    return error, error_emp, supp_diff, nnz, nnz_true

if __name__ == '__main__':
    if len(sys.argv) < 2:
        timestr = time.strftime("%H%M%S%m%d")
        FDIR = 'outputs/dev_ic_%s' % timestr
    else:
        FDIR = sys.argv[1]
    if not os.path.exists(FDIR):
        os.makedirs(FDIR)
    print(FDIR)


    # Iterate through all problem settings ('pbs')
    res = []
    i = 0
    if True:
        # Get data
        X, Wtrue = get_data_sachs(normalize=True)
        print(X.shape)
        print('Above is the size of the Sachs data\n')

        # IC BY QUIC
        n_samples = X.shape[0]
        n_features = X.shape[1]
        d = n_features
        cv_folds = 3

        true_prec = (np.eye(d)-Wtrue) @ (np.eye(d)-Wtrue).T  / 1.0
        true_cov = np.linalg.inv(true_prec)
        # Empirical cov
        X_ = X - np.mean(X, axis=0, keepdims=True)
        emp_cov = np.dot(X.T, X) / n_samples

        plot_covs = [("True", true_cov), ("True", true_cov), ("True", true_cov)]
        plot_precs = [
            ("True", true_prec, ""),
            ("True", true_prec, ""),
            ("True", true_prec, ""),
        ]
        results = []

        if False:
            # Empirical
            start_time = time.time()
            cov, prec = empirical(X)
            end_time = time.time()
            ctime = end_time - start_time
            name = "Empirical"
            plot_covs.append((name, cov))
            plot_precs.append((name, prec, ""))
            error, err_emp, supp_diff, nnz, nnz_true = _comp_stats(prec, cov, \
                                                true_prec, true_cov, \
                                                    emp_cov)
            f = _criterion_trace(prec, true_prec, true_cov, emp_cov)
            results.append([name, error, err_emp, f, supp_diff, \
                            nnz, nnz_true, ctime, lam])
            # print("   frobenius error: {}".format(error))
            print("")

            # sklearn LedoitWolf
            start_time = time.time()
            cov, prec = sk_ledoit_wolf(X)
            end_time = time.time()
            ctime = end_time - start_time
            name = "Ledoit-Wolf (sklearn)"
            plot_covs.append((name, cov))
            plot_precs.append((name, prec, ""))
            # error = np.linalg.norm(true_cov - cov, ord="fro")
            # supp_diff = _count_support_diff(true_prec, prec)
            # results.append([name, error, supp_diff, ctime, ""])
            error, err_emp, supp_diff, nnz, nnz_true = _comp_stats(prec, cov, \
                                                true_prec, true_cov, \
                                                    emp_cov)
            f = _criterion_trace(prec, true_prec, true_cov, emp_cov)
            results.append([name, error, err_emp, f, supp_diff, \
                            nnz, nnz_true, ctime, lam])
            # print("   frobenius error: {}".format(error))
            print("")

        # Sparse empirical
        lams = np.linspace(1e-3,5e-2,20)
        j = 0
        for lam in lams:
            j +=1
            start_time = time.time()
            prec, cov = sp_ice_naive(X, lam)
            end_time = time.time()
            ctime = end_time - start_time
            name = "Sparse Empirical %d" % j
            plot_covs.append((name, cov))
            plot_precs.append((name, prec, lam))
            # error = np.linalg.norm(true_cov - cov, ord="fro")
            # supp_diff = _count_support_diff(true_prec, prec)
            error, err_emp, supp_diff, nnz, nnz_true = _comp_stats(prec, cov, \
                                                true_prec, true_cov, \
                                                emp_cov)
            f = _criterion_trace(prec, true_prec, true_cov, emp_cov)
            results.append([name, error, err_emp, f, supp_diff, \
                                nnz, nnz_true, ctime, lam])

        # tabulate errors
        print(
            tabulate.tabulate(
                results,
                headers=[
                    "Estimator",
                    "Error (Frobenius)",
                    "Error (w Emp Cov)",
                    "fit function",
                    "Support Diff",
                    "nnz (Prec)",
                    "nnz (Prec true)",
                    "Time",
                    "Lambda",
                ],
                tablefmt="pipe",
            )
        )
        print("")
        res=pd.DataFrame(results,
                     columns=[
                        "Estimator",
                        "Error (Frobenius)",
                        "Error (w Emp Cov)",
                        "fit function",
                        "Support Diff",
                        "nnz (Prec)",
                        "nnz (Prec true)",
                        "Time",
                        "Lambda"]
                     )
        c1,c2,c3,sel,idx = _selection(
                res['Error (w Emp Cov)'],\
                res['fit function'], \
                res['Support Diff'], \
                res['nnz (Prec)']
                )
        pd.set_option('display.max_columns', None)
        print("Criterion err-emp: {}".format(c1))
        print("Criterion fit fun: {}".format(c2))
        print("nnzs of best fits: {}".format(c3))
        print("=======Lambda selected vs Argmin SuppDiff(wrt true Prec)=====")
        print(res.iloc[[sel,idx]])
        print("============\n")
        res.to_csv('%s/ic_pb%s_resall.csv' %(FDIR,i))
        # display results
        # show_results(plot_covs, plot_precs)
        # r_input("Press any key to exit...")


