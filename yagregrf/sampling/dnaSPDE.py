from numpy import zeros

from yagregrf.sampling.interface import SamplingEngine
from yagregrf.sampling.spde import SPDEEngine2d


class DNASPDEEngine2d(SamplingEngine):

    def __init__(self, variance, corrLength, nu, nVertPerDim, alpha):

        self._nGrid = nVertPerDim
        self._alpha = alpha

        self._engines = [
            SPDEEngine2d(variance, corrLength, nu, nVertPerDim, alpha,
                [False, False]),
            SPDEEngine2d(variance, corrLength, nu, nVertPerDim, alpha,
                [False, True]),
            SPDEEngine2d(variance, corrLength, nu, nVertPerDim, alpha,
                [True, False]),
            SPDEEngine2d(variance, corrLength, nu, nVertPerDim, alpha,
                [True, True])]

    @property
    def mesh(self):
        return self._engines[0].mesh

    def generate_realisation(self):

        realisation = zeros((self._nGrid, self._nGrid))

        for spdeEngine in self._engines:
            realisation += spdeEngine.generate_realisation()

        return 0.5 * realisation
