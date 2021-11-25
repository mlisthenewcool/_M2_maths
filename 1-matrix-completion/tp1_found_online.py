from __future__ import division
import numpy as np
import logging

import numpy as np
from scipy.stats import bernoulli


def gen_mask(m, n, prob_masked=0.5):
    """
    Generate a binary mask for m users and n movies.
    Note that 1 denotes observed, and 0 denotes unobserved.
    """
    return 1 - bernoulli.rvs(p=prob_masked, size=(m, n))


def gen_factorization_without_noise(m, n, k):
    """
    Generate non-noisy data for m users and n movies with k latent factors.
    Draws factors U, V from Gaussian noise and returns U Vᵀ.
    """
    U = np.random.randn(m, k)
    V = np.random.randn(n, k)
    R = np.dot(U, V.T)
    return U, V, R


def gen_factorization_with_noise(m, n, k, sigma):
    """
    Generate noisy data for m users and n movies with k latent factors.
    Gaussian noise with variance sigma^2 is added to U V^T.
    Effect is a matrix with a few large singular values and many close to zero.
    """
    U = np.random.randn(m, k)
    V = np.random.randn(n, k)
    R = np.random.randn(m, n) * sigma + np.dot(U, V.T)
    return U, V, R

def svt_solve(A, mask, tau=None, delta=None, epsilon=1e-2, max_iterations=1000):
    """
    Solve using iterative singular value thresholding.
    [ Cai, Candes, and Shen 2010 ]
    Parameters:
    -----------
    A : m x n array
        matrix to complete
    mask : m x n array
        matrix with entries zero (if missing) or one (if present)
    tau : float
        singular value thresholding amount;, default to 5 * (m + n) / 2
    delta : float
        step size per iteration; default to 1.2 times the undersampling ratio
    epsilon : float
        convergence condition on the relative reconstruction error
    max_iterations: int
        hard limit on maximum number of iterations
    Returns:
    --------
    X: m x n array
        completed matrix
    """
    logger = logging.getLogger(__name__)
    Y = np.zeros_like(A)

    if not tau:
        tau = 5 * np.sum(A.shape) / 2
    if not delta:
        delta = 1.2 * np.prod(A.shape) / np.sum(mask)

    for _ in range(max_iterations):

        U, S, V = np.linalg.svd(Y, full_matrices=False)
        S = np.maximum(S - tau, 0)

        X = np.linalg.multi_dot([U, np.diag(S), V])
        Y += delta * mask * (A - X)

        recon_error = np.linalg.norm(mask * (X - A)) / np.linalg.norm(mask * A)
        if _ % 1 == 0:
            logger.info("Iteration: %i; Rel error: %.4f" % (_ + 1, recon_error))
        if recon_error < epsilon:
            break

    return X

import numpy as np


def calc_unobserved_rmse(U, V, A_hat, mask):
    """
    Calculate RMSE on all unobserved entries in mask, for true matrix UVᵀ.
    Parameters
    ----------
    U : m x k array
        true factor of matrix
    V : n x k array
        true factor of matrix
    A_hat : m x n array
        estimated matrix
    mask : m x n array
        matrix with entries zero (if missing) or one (if present)
    Returns:
    --------
    rmse : float
        root mean squared error over all unobserved entries
    """
    pred = np.multiply(A_hat, (1 - mask))
    truth = np.multiply(np.dot(U, V.T), (1 - mask))
    cnt = np.sum(1 - mask)
    return (np.linalg.norm(pred - truth, "fro") ** 2 / cnt) ** 0.5


def calc_validation_rmse(validation_data, A_hat):
    """
    Calculate validation RMSE on all validation entries.
    Parameters
    ----------
    validation_data : list
        list of tuples (i, j, r) where (i, j) are indices of matrix with entry r
    A_hat : m x n array
        estimated matrix
    """
    total_error = 0.0
    for (u, i, r) in validation_data:
        total_error += (r - A_hat[int(u),int(i)]) ** 2
    return np.sqrt(total_error / len(validation_data))

import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser


def plot_image(A):
    plt.imshow(A.T)
    plt.show()


if __name__ == "__main__":

    argparse = ArgumentParser()
    argparse.add_argument("--m", default=20, type=int)
    argparse.add_argument("--n", default=6, type=int)
    argparse.add_argument("--k", default=3, type=int)
    argparse.add_argument("--noise", default=0.1, type=float)
    argparse.add_argument("--mask-prob", default=0.75, type=float)

    args = argparse.parse_args()

    U, V, R = gen_factorization_without_noise(args.m, args.n, args.k)
    mask = gen_mask(args.m, args.n, args.mask_prob)

    plot_image(R)
    plot_image(mask)

    R_hat = svt_solve(R, mask)

    print(R)
    print(R_hat)

    plot_image(R_hat)
