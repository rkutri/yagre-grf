import numpy as np

from scipy.special import gamma, kv


def matern_kappa(corrLength, smoothness):
    return np.sqrt(2. * smoothness) / corrLength


def matern_beta(smoothness, dim):
    return 0.5 * (smoothness + 0.5 * dim)


def matern_covariance_ptw(diff, corrLength, nu):

    if (np.abs(diff) < 1e-8):
        return 1.0

    r = np.abs(matern_kappa(corrLength, nu) * diff)

    return np.power(2., 1. - nu) / gamma(nu) * np.power(r, nu) * kv(nu, r)


def matern_fourier_ptw(s, corrLength, nu, dim=2):

    kappa = matern_kappa(corrLength, nu)
    beta = matern_beta(nu, dim)

    normConst = np.power(4. * np.pi, dim / 2) * gamma(2. * beta) / gamma(nu) \
        * np.power(kappa, 2. * nu)

    return normConst * np.power(kappa**2 + (2. * np.pi * s)**2, -2. * beta)


def cauchy_ptw(d, corrLength):

    return np.power(1. + np.square(np.abs(d) / corrLength), -1.)


def cauchy_fourier_ptw(s, corrLength):

    return np.pi * corrLength * np.exp(-corrLength * 2. * np.pi * np.abs(s))


def gaussian_ptw(d, corrLength):

    return np.exp(-0.5 * np.square(d / corrLength))


def gaussian_fourier_ptw(s, corrLength):

    return np.sqrt(2. * np.pi) * corrLength * \
        np.exp(-2. * np.square(corrLength * np.pi * s))
