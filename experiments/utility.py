import numpy as np

from scipy.fft import dct, dst


def norm(x):
    return np.sqrt(np.sum(np.square(x)))


"""
Evaluate the series

    \sqrt(2) \sum_{m=0}{n-1} a[m] cos(pi*m*x_k),

where x_k = k/(n+1) for k =0, 1, ..., n
"""


def cos_series(a):

    acopy = np.copy(a / np.sqrt(2.))
    acopy[0] *= np.sqrt(2.)

    return dct(np.append(acopy, 0.), norm="backward", type=1)


"""
Evaluate the series

    \sqrt(2) \sum_{m=1}{n} a[m] sin(pi*m*x_k),

where x_k = k/(n+1) for k = 0, 1, ..., n
"""


def sin_series(a):

    series = dst(np.copy(a[1:]) / np.sqrt(2.), norm="backward", type=1)
    series = np.append(np.insert(series, 0, 0.), 0.)

    return series


def even_series_2d(a):

    n0 = a.shape[0]
    n1 = a.shape[1]

    cc = np.zeros((n0, n1))

    for col in range(n1):
        cc[:, col] = cos_series(a[:, col])

    for row in range(n0):
        cc[row, :] = cos_series(cc[row, :])

    ss = np.zeros((n0, n1))

    for col in range(1, n1):
        ss[:, col] = sin_series(a[:, col])

    for row in range(1, n0):
        ss[row, :] = sin_series(ss[row, :])

    return cc - ss


def odd_series_2d(a):

    n0 = a.shape[0]
    n1 = a.shape[1]

    mix1 = np.zeros((n0 - 1, n1))

    for row in range(1, n0):
        mix1[row, :] = 0


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


def extract_diagonal_slice(matrix):

    n = matrix.shape[0]

    assert matrix.shape[1] == n

    return np.array([matrix[i, i] for i in range(n)])
