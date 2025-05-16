from numpy import zeros

from yagregrf.sampling.interface import SamplingEngine
from yagregrf.sampling.spde import SPDEEngine2D, SPDEEngine2DDeep


class DNASPDEEngine2D(SamplingEngine):

    def __init__(self, variance, corrLength, nu, nVertPerDim,
                 alpha, cacheFactorisation=True):

        self._nGrid = nVertPerDim
        self._alpha = alpha

        self._engines = [
            SPDEEngine2D(variance, corrLength, nu, nVertPerDim, alpha,
                         [False, False], cacheFactorisation),
            SPDEEngine2D(variance, corrLength, nu, nVertPerDim, alpha,
                         [False, True], cacheFactorisation),
            SPDEEngine2D(variance, corrLength, nu, nVertPerDim, alpha,
                         [True, False], cacheFactorisation),
            SPDEEngine2D(variance, corrLength, nu, nVertPerDim, alpha,
                         [True, True], cacheFactorisation)]

    @property
    def mesh(self):
        return self._engines[0].mesh

    @property
    def nDof(self):
        return self._engines[0].nDof

    def generate_realisation(self):

        realisation = zeros((self._nGrid, self._nGrid))

        for spdeEngine in self._engines:
            realisation += spdeEngine.generate_realisation()

        return 0.5 * realisation


class DNASPDEEngine2DDeep(DNASPDEEngine2D):

    def __init__(self, variance, correlation, nu,
                 nVertPerDim, alpha, cacheFactorisation=True):

        self._nGrid = nVertPerDim
        self._alpha = alpha

        self._engines = [
            SPDEEngine2DDeep(variance, correlation, nu, nVertPerDim, alpha,
                             [False, False], cacheFactorisation),
            SPDEEngine2DDeep(variance, correlation, nu, nVertPerDim, alpha,
                             [False, True], cacheFactorisation),
            SPDEEngine2DDeep(variance, correlation, nu, nVertPerDim, alpha,
                             [True, False], cacheFactorisation),
            SPDEEngine2DDeep(variance, correlation, nu, nVertPerDim, alpha,
                             [True, True], cacheFactorisation)]

    @property
    def coefficient(self):
        return self._engines[0].coefficient

    @coefficient.setter
    def coefficient(self, coeff):

        for i in range(len(self._engines)):
            self._engines[i].coefficient = coeff

    @property
    def anisotropy(self):
        return self._engines[0].anisotropy

    @anisotropy.setter
    def anisotropy(self, K):

        for i in range(len(self._engines)):
            self._engines[i].anisotropy = K
