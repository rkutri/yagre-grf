import numpy as np

from scipy.fft import dct, dst


def cos_series(a):
    """
    Evaluate the series

        \sqrt(2) \sum_{m=0}{n-1} a[m] cos(pi*m*x_k),

    where x_k = k/(n+1) for k =0, 1, ..., n
    """

    acopy = np.copy(a / np.sqrt(2.))
    acopy[0] *= np.sqrt(2.)

    return dct(np.append(acopy, 0.), norm="backward", type=1)


def sin_series(a):
    """
    Evaluate the series

        \sqrt(2) \sum_{m=1}{n} a[m] sin(pi*m*x_k),

    where x_k = k/(n+1) for k = 0, 1, ..., n
    """

    series = dst(np.copy(a[1:]) / np.sqrt(2.), norm="backward", type=1)
    series = np.append(np.insert(series, 0, 0.), 0.)

    return series
