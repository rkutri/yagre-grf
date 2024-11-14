import numpy as np


def norm(x):
    return np.sqrt(np.sum(np.square(x)))


def evaluate_isotropic_covariance_1d(iso_cov_fcn, grid):

    nDof = grid.size

    cov = np.zeros((nDof, nDof))

    for i in range(nDof):
        for j in range(nDof):
            cov[i, j] = iso_cov_fcn(norm(grid[i] - grid[j]))

    return cov


def evaluate_periodised_covariance_1d(cov_fcn, grid, alpha, nPrd=20):

    def prd_cov_fcn(x):

        prdCovVal = cov_fcn(x)

        for n in range(1, nPrd):

            prdCovVal += cov_fcn(x + alpha * n)
            prdCovVal += cov_fcn(x - alpha * n)

        return prdCovVal

    return evaluate_isotropic_covariance_1d(prd_cov_fcn, grid)
