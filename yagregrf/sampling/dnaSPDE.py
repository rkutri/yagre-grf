from numpy import zeros

from sampling.interface import SamplingEngine
from sampling.spde import SPDEEngine2d


class DNASPDEEngine2d(SamplingEngine):

    def __init__(self, corrLength, nu, nVertPerDim, alpha):

        self._nGrid = nVertPerDim
        self._alpha = alpha
            
        self._engines = [SPDEEngine2d(corrLength, nu, nVertPerDim, alpha, [False, False]),
                         SPDEEngine2d(corrLength, nu, nVertPerDim, alpha, [False, True]),
                         SPDEEngine2d(corrLength, nu, nVertPerDim, alpha, [True, False]),
                         SPDEEngine2d(corrLength, nu, nVertPerDim, alpha, [True, True])]


    def generate_realisation(self):

        realisation = zeros((self._nGrid, self._nGrid))

        for spdeEngine in self._engines:
            realisation += spdeEngine.generate_realisation()

        return 0.5 * realisation
