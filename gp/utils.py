import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from scipy.linalg import eigh_tridiagonal


def generate_data(x, sigma):
    """
    Generate the function to estimate
    :param x: sampling point, 1d array-like, n_samples_x
    :param sigma: standard deviation of the observation noise
    :return: noisy observation of true function,  n_samples_x
    """
    return np.sin(x) + 0.5 * np.sin(4 * x) + np.random.randn(x.shape[0]) * sigma


def rbfkernel(x, y, ls=0.2):
    """
    Evaluate RBF Kernel
    :param x: ndarray of shape (n_samples_x, n_features) or (n_samples_x,)
    :param y: ndarray of shape (n_samples_y, n_features) or (n_samples_y,)
    :param ls: length scale parameter
    :return: ndarray of shape (n_samples_x, n_samples_y)
    """
    gamma = 1 / 2 / (ls ** 2)
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    if y is None:
        y = x
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    kernel = rbf_kernel(x, y, gamma=gamma)
    return kernel


def conjugate_gradient(a, b, x_0, kmax=20, tol=1e-5):
    """
    Single conjugate gradient method for linear solve
    :param a: ndarray of shape (n, n)
    :param b: ndarray of shape (n, 1) or (n, )
    :param x_0: initialization
    :param kmax: max CG iteration
    :param tol: convergence threshold of residual
    :return: x = A^{-1}b
    """
    if len(b.shape) == 1:
        b = b.reshape(-1, 1)
    if len(x_0.shape) == 1:
        x_0 = x_0.reshape(-1, 1)
    r_k = b - np.dot(a, x_0)
    r2_old = np.dot(r_k.T, r_k)
    x_k = x_0
    p_k = r_k
    k = 0
    res = np.sqrt(r2_old[0, 0])
    while res > tol and k < kmax:
        a_pk = np.dot(a, p_k)
        # A_pk = a @ p_k
        alpha_k = r2_old / np.dot(p_k.T, a_pk)
        x_k = x_k + alpha_k * p_k
        r_k = r_k - alpha_k * a_pk
        # r_k = b - a@x_k  # explicit
        r2_new = np.dot(r_k.T, r_k)
        res = np.sqrt(r2_new[0, 0])
        k += 1
        beta_k = r2_new / r2_old
        p_k = r_k + beta_k * p_k
        r2_old = r2_new
    return x_k


def compute_tridiagonal_bcg(alpha_lst, beta_lst):
    """
    Compute the tridiagonal matrix T from Batch CG coefficients alpha and beta
    :param alpha_lst: a list of alpha arrays, length equal to CG iteractions, array dim = number of probe vectors
    :param beta_lst: a list of beta arrays, length equal to CG iteractions, array dim = number of probe vectors
    :return: diagonal td and off-diagonal-1 te of the tridiagonal matrix T
    """
    alpha_arr = np.array(alpha_lst)
    beta_arr = np.array(beta_lst)
    p, m = alpha_arr.shape
    td = 1 / alpha_arr + np.vstack((np.zeros((1, m)), beta_arr[0:(p - 1), :] / alpha_arr[0:(p - 1):, :]))
    te = np.sqrt(beta_arr[0:(p - 1), :]) / alpha_arr[0:(p - 1), :]
    return td, te


def batch_conjugate_gradient(am, bm, xm_0, kmax=20, tol=1e-8):
    """
    Conjugate gradient method (batch) with T matrices as output
    Algorithm2 in Appendix of Gardner 2018
    with a correction on the beta update equation
    and without preconditioning (found pivot Cholesky and incomplete Cholesky preconditioning
    ineffective in the case looked at)
    :param am: ndarray of shape (n, n)
    :param bm: ndarray of shape (n, m + 1), m prob vectors with last column as b
    :param xm_0: ndarray of shape (n, m + 1)
    :param kmax: max CG iteration
    :param tol: convergence threshold of residual
    :return: linear solve x = A^{-1}b, and the lanczos triangular matrix T
    """
    rm_k = bm - am @ xm_0  # current residual, n x m
    pm_k = rm_k  # "Search" directions for next solutions n x m
    r2_old_vec = np.sum(rm_k * rm_k, axis=0)
    res = np.sqrt(np.max(r2_old_vec))
    xm_k = xm_0
    k = 0
    alpha_lst = list()
    beta_lst = list()
    while res > tol and k < kmax:
        am_pk = am @ pm_k   # n x m
        alpha_k = r2_old_vec / np.sum(pm_k * am_pk, axis=0)
        alpha_lst.append(alpha_k)
        alpha_km = np.broadcast_to(alpha_k, xm_k.shape)
        xm_k = xm_k + pm_k * alpha_km
        rm_k = rm_k - am_pk * alpha_km  # implicit
        r2_new_vec = np.sum(rm_k * rm_k, axis=0)
        res = np.sqrt(np.max(r2_new_vec))
        k += 1
        beta_k = r2_new_vec / r2_old_vec
        beta_lst.append(beta_k)
        beta_km = np.broadcast_to(beta_k, rm_k.shape)
        pm_k = rm_k + pm_k * beta_km
        r2_old_vec = r2_new_vec
    t_d, t_e = compute_tridiagonal_bcg(alpha_lst, beta_lst)
    x_k = xm_k[:, -1]  # K^{-1}y
    return x_k, t_d, t_e


def batch_conjugate_gradient_needle(am_, bm_, xm_0_, kmax=20, tol=1e-8):
    """
    Conjugate gradient method (batch) with T matrices as output
    Algorithm2 in Appendix of Gardner 2018
    with a correction on the beta update equation
    and without preconditioning (found pivot Cholesky and incomplete Cholesky preconditioning
    ineffective in the case looked at)
    :param am_: needle array of shape (n, n) on cuda
    :param bm_: needle array of shape (n, m + 1) on cuda, m prob vectors with last column as b
    :param xm_0_: needle array of shape (n, m + 1) on cuda
    :param kmax: max CG iteration
    :param tol: convergence threshold of residual
    :return: linear solve x = A^{-1}b, and the lanczos triangular matrix T
    """
    xm_k_ = xm_0_
    rm_k_ = bm_ - am_ @ xm_k_  # current residual, n x m
    r2_old_vec_ = (rm_k_ * rm_k_).sum(axis=0)
    pm_k_ = rm_k_  # "Search" directions for next solutions n x m
    res_ = r2_old_vec_.max()
    res = np.sqrt(res_.numpy())
    k = 0
    alpha_lst = list()
    beta_lst = list()
    while res > tol and k < kmax:
        am_pk_ = am_ @ pm_k_
        alpha_k_ = r2_old_vec_ / ((pm_k_ * am_pk_).sum(axis=0))
        alpha_km_ = alpha_k_.broadcast_to(xm_k_.shape)
        xm_k_ = xm_k_ + pm_k_ * alpha_km_
        rm_k_ = rm_k_ - am_pk_ * alpha_km_
        r2_new_vec_ = (rm_k_ * rm_k_).sum(axis=0)
        res_ = r2_new_vec_.max()
        res = np.sqrt(res_.numpy())
        k += 1
        beta_k_ = r2_new_vec_ / r2_old_vec_
        beta_km_ = beta_k_.broadcast_to(rm_k_.shape)
        pm_k_ = rm_k_ + pm_k_ * beta_km_
        r2_old_vec_ = r2_new_vec_
        alpha_k = alpha_k_.numpy().flatten()
        alpha_lst.append(alpha_k)
        beta_k = beta_k_.numpy().flatten()
        beta_lst.append(beta_k)

    xm_k = xm_k_.numpy()
    t_d, t_e = compute_tridiagonal_bcg(alpha_lst, beta_lst)
    x_k = xm_k[:, -1]  # K^{-1}y
    return x_k, t_d, t_e


def logdet_ste_lanczos(t_d, t_e, n, m):
    """
    logdet(A) via stochastic trace estimator based on the Lanczos quadrature
    Assuming A = Q.tTQ, and T = Phi.t log(lambda) Phi
    logdet(A) = trace(logm(A)) = n/m sum_{i}^{m} Phi.t log(lambda) Phi[0, 0]
    :param t_d: (kmax, m) storing diagonal elements of m tridiagonal matrices T
    :param t_e: (kmax-1, m) storing off-diagonal elements of m tridiagonal matrices T
    :param n: the dim of A matrix
    :param m: the dim of T matrix
    :return: logdet(A) estimated by taking the mean
    """
    m_probe = t_d.shape[1]
    logdet_ste = 0
    t = 0
    for i in range(m_probe):
        d = t_d[:, i]
        e = t_e[:, i]
        w, v = eigh_tridiagonal(d, e)
        t += 1
        logmt_eig = v @ np.diag(np.log(w)) @ v.T
        inc = logmt_eig[0, 0]
        logdet_ste += inc * n / m
    return logdet_ste


def gp_prediction(tr_x, tr_y, te_x, para_opt, method_str):
    """
    Use the learned hyperparameter for GP prediction at new locations
    :param tr_x: training inputs (num_training, )
    :param tr_y: training outputs (num_training, )
    :param te_x: test inputs (num_test, )
    :param para_opt: optimized parameter [length_scale, likelihood_noise]
    :param method_str: the GP training method used
    :return: GP posterior mean at te_x, and the lower - upper bound estimated from posterior variance
    """
    if method_str == 'GPyTorch':
      ell = para_opt[0]
      post_sig_est = para_opt[1]
    else:
      ell = para_opt.x[0]
      post_sig_est = para_opt.x[1]
    k_x_xstar = rbfkernel(tr_x, te_x,  ls=ell)
    k_x_x = rbfkernel(tr_x, tr_x,  ls=ell)
    k_xstar_xstar = rbfkernel(te_x, te_x, ls=ell)
    post_mean = k_x_xstar.T @ np.linalg.inv((k_x_x + post_sig_est ** 2 * np.eye(tr_x.shape[0]))) @ tr_y
    post_cov = k_xstar_xstar - k_x_xstar.T @ np.linalg.inv((k_x_x + post_sig_est ** 2 * np.eye(
        tr_x.shape[0]))) @ k_x_xstar
    lw_bd = post_mean - 2 * np.sqrt(np.diag(post_cov))
    up_bd = post_mean + 2 * np.sqrt(np.diag(post_cov))
    return post_mean, lw_bd, up_bd


def plot_gp_inference(tr_x, tr_y, te_x, te_y, learned_hypers_opt, method_str):
    """
    Visualize GP inference results
    :param tr_x: training inputs (num_training, )
    :param tr_y: training outputs (num_training, )
    :param te_x: test inputs (num_test, )
    :param te_y:test outputs (num_test, )
    :param learned_hypers_opt: optimized parameter [length_scale, likelihood_noise]
    :param method_str: the GP training method used
    """
    post_mean, lw_bd, up_bd = gp_prediction(tr_x, tr_y, te_x, learned_hypers_opt, method_str)
    plt.figure()
    plt.scatter(tr_x, tr_y, label='Observed Data')
    plt.plot(te_x, te_y, linewidth=2., label='True Function')
    plt.plot(te_x, post_mean, linewidth=2., label='Org Predictive Mean')
    plt.fill_between(te_x, lw_bd, up_bd, alpha=0.25, label='Org 95% Set on True Func')
    plt.title(method_str)
    plt.legend()
    plt.show()
