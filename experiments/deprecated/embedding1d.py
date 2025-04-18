import numpy as np
from scipy.fft import irfft

import covariance_functions as covs


def determine_smallest_factor(is_indef, upper, tol=1e-1, maxIter=6):

    if is_indef(upper):
        raise ValueError(
            "Initial factor is not sufficient for positive definiteness")

    lower = 1.
    mid = upper

    iterations = 0

    while upper > 1. + tol and iterations < maxIter:

        mid = 0.5 * (lower + upper)

        midIndef = is_indef(mid)

        if midIndef:
            lower = mid
        else:
            upper = mid

        iterations += 1

    factor = mid

    return factor, iterations


def embedding_is_indefinite(covEval):

    embedding = np.concatenate((covEval, covEval[-2:-1:-1]))

    eigVals = irfft(embedding)

    if np.any(np.abs(eigVals.imag) > 1e-6):
        print("imaginary eigenvalues")
        return True

    if np.any(eigVals.real < 0):
        return True

    else:
        return False


def determine_smallest_embedding_factor(cov_fcn, nTgtGrid, maxFactor=1000.):

    def is_indefinite(factor):

        nEmbGrid = int(np.rint(factor * nTgtGrid))

        embeddingGrid = np.linspace(0., factor, nEmbGrid, endpoint=True)

        covEval = cov_fcn(embeddingGrid)

        if embedding_is_indefinite(covEval):
            return True
        else:
            return False

    print("determine a valid initial factor")
    factor = 1.

    while is_indefinite(factor) and factor < maxFactor:
        factor *= 2.

    print(f"initial factor: {factor}")

    print("start bisection")

    minFactor = determine_smallest_factor(is_indefinite, factor)

    print(f"smallest factor: {minFactor[0]}")

    return minFactor


if __name__ == "__main__":

    ell = 0.2
    nu = 3.0

    nGrid = 500

    def cov_fcn(grid): return np.array(
        [covs.matern_covariance_ptw(x, ell, nu) for x in grid])

    minFactor = determine_smallest_embedding_factor(cov_fcn, nGrid)

    print(f"Smallest embedding for ell={ell} and nu={nu}: {minFactor}")
