import numpy as np

from scipy.special import gamma, kv


def matern_kappa(ell, nu):
    return np.sqrt(2. * nu) / ell


def matern_beta(nu, dim):
    return 0.5 * (nu + 0.5 * dim)


def matern_ptw(delta, ell, nu):

    if (np.abs(delta) < 1e-8):
        return 1.

    r = np.abs(matern_kappa(ell, nu) * delta)

    return np.power(2., 1. - nu) / gamma(nu) * np.power(r, nu) * kv(nu, r)


def matern_fourier_ptw(s, ell, nu, dim):

    kappa = matern_kappa(ell, nu)
    beta = matern_beta(nu, dim)

    normConst = np.power(4. * np.pi, dim / 2) * gamma(2. * beta) / gamma(nu) \
        * np.power(kappa, 2. * nu)

    return normConst * np.power(kappa**2 + (2. * np.pi * s)**2, -2. * beta)


def matern_fourier_1d_ptw(s, ell, nu):
    return matern_fourier_ptw(s, ell, nu, 1)


def cauchy_ptw(delta, ell):
    return np.power(1. + np.square(np.abs(delta) / ell), -1.)


def cauchy_fourier_1d_ptw(s, ell):
    return np.pi * ell * np.exp(-ell * 2. * np.pi * np.abs(s))


def gaussian_ptw(delta, ell):
    return np.exp(-0.5 * np.square(delta / ell))


def gaussian_fourier_1d_ptw(s, ell):
    return np.sqrt(2. * np.pi) * ell * np.exp(-2. * np.square(ell * np.pi * s))
