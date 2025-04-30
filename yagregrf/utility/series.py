import numpy as np
from scipy.fft import dct, dst


def cos_series(a):
    """
    Evaluate the cosine series:
        sqrt(2) * sum_{m=0}^{n-1} a[m] * cos(pi * m * x_k),
    where x_k = k / (n + 1), for k = 0, ..., n.
    """
    acopy = np.copy(a / np.sqrt(2.))
    acopy[0] *= np.sqrt(2.)
    return dct(np.append(acopy, 0.), norm="backward", type=1)


def sin_series(a):
    """
    Evaluate the sine series:
        sqrt(2) * sum_{m=1}^{n} a[m] * sin(pi * m * x_k),
    where x_k = k / (n + 1), for k = 0, ..., n.
    """
    seriesInner = dst(a[1:] / np.sqrt(2.), norm="backward", type=1)
    return np.concatenate(([0.], seriesInner, [0.]))


def batch_sin_series_rows(A):
    """
    Only gives the correct result if 2d array and first row then column
    """

    # SHOULD BE CORRECT NOW
    inner = dst(A[:, 1:] / np.sqrt(2.), norm="backward", type=1, axis=1)
    return np.pad(inner, pad_width=((0, 0), (1, 1)))


def batch_sin_series_cols(A):
    """
    Only gives the correct result if 2d array and first row then column
    """

    # SHOULD BE CORRECT NOW
    inner = dst(A[1:, :] / np.sqrt(2.), norm="backward", type=1, axis=0)
    return np.pad(inner, ((1, 1), (0, 0)))


def batch_cos_series_rows(A):
    """
    Only gives the correct result if 2d array and first row then column
    """

    # SHOULD BE CORRECT NOW
    B = A / np.sqrt(2.)
    B[:, 0] *= np.sqrt(2.)
    return dct(np.pad(B, ((0, 0), (0, 1))), norm="backward", type=1, axis=1)


def batch_cos_series_cols(A):
    """
    Only gives the correct result if 2d array and first row then column
    """

    B = A / np.sqrt(2)
    B[0, :] *= np.sqrt(2.)
    return dct(np.pad(B, ((0, 1), (0, 0))), norm="backward", type=1, axis=0)
