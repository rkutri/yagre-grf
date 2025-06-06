import numpy as np
from scipy.special import gamma, kv


def matern_kappa(ell, nu):
    return np.sqrt(2. * nu) / ell


def matern_beta(nu, dim):
    return 0.5 * (nu + 0.5 * dim)


def matern_ptw(delta, ell, nu, margVar=1.0):

    if np.abs(delta) < 1e-8:
        return margVar

    r = np.abs(matern_kappa(ell, nu) * delta)
    return margVar * np.power(2., 1. - nu) / gamma(nu) \
        * np.power(r, nu) * kv(nu, r)


def matern_fourier_ptw(s, ell, nu, dim=2, margVar=1.0):

    s = np.asarray(s)

    kappa = matern_kappa(ell, nu)
    beta = matern_beta(nu, dim)
    normConst = (4. * np.pi)**(dim / 2) * \
        gamma(2. * beta) / gamma(nu) * kappa**(2. * nu)

    sSq = np.sum((2. * np.pi * s)**2)
    return margVar * normConst * (kappa**2 + sSq)**(-2. * beta)


def matern_fourier_1d_ptw(s, ell, nu, margVar=1.0):
    return matern_fourier_ptw([s], ell, nu, dim=1, margVar=margVar)


def cauchy_ptw(delta, ell):
    return np.power(1. + np.square(np.abs(delta) / ell), -1.)


def cauchy_fourier_1d_ptw(s, ell, margVar=1.0):
    return np.pi * ell * margVar * np.exp(-ell * 2. * np.pi * np.abs(s))


def cauchy_fourier_ptw(s, ell, dim=2, margVar=1.0):
    s = np.asarray(s)
    s_norm = np.sqrt(np.sum(s**2))
    return margVar * (2. * np.pi * ell)**dim * \
        np.exp(-2. * np.pi * ell * s_norm)


def gaussian_ptw(delta, ell, margVar=1.0):
    return margVar * np.exp(-0.5 * np.square(delta / ell))


def gaussian_fourier_1d_ptw(s, ell, margVar=1.0):
    return np.sqrt(2. * np.pi) * ell * margVar * \
        np.exp(-2. * np.square(ell * np.pi * s))


def gaussian_fourier_ptw(s, ell, dim=2, margVar=1.0):
    s = np.asarray(s)
    s_sq = np.sum((2. * np.pi * s)**2)
    return margVar * (2. * np.pi)**(dim / 2) * ell**dim * \
        np.exp(-0.5 * s_sq * ell**2)
