import numpy as np
import yagregrf.utility.series as srs

from enum import Enum
from scipy.fft import dct, dst
from numpy.random import standard_normal

from yagregrf.sampling.randomField import RandomField
from yagregrf.sampling.interface import SamplingEngine
from yagregrf.utility.evaluation import norm


class DNAFourierEngine1D(SamplingEngine):

    def __init__(self, cov_fourier_callable, nVertices, scaling=1.0):

        if nVertices < 3:
            raise ValueError("Grid must contain at least 3 points")

        self._nGrid = nVertices
        self._alpha = scaling
        self._detCoeff = self.compute_coefficient(cov_fourier_callable)

    def compute_coefficient(self, ftrans_fcn):
        return np.sqrt([ftrans_fcn([m / (2. * self._alpha)])
                        for m in range(self._nGrid - 1)])

    def generate_realisation(self):

        # inner degrees of freedom (two boundary vertices)
        nDofInner = self._nGrid - 2

        dirEval = srs.sin_series(
            standard_normal(
                nDofInner +
                1) *
            self._detCoeff)
        neuEval = srs.cos_series(
            standard_normal(
                nDofInner +
                1) *
            self._detCoeff)

        return (dirEval + neuEval) / np.sqrt(2. * self._alpha)


class BC(Enum):

    NONE = -1
    SINE = 0
    COSINE = 1


class DNAFourierEngine2D(SamplingEngine):

    def __init__(self, cov_fourier_callable, vertPerDim, scaling=1.0):

        if vertPerDim < 3:
            raise ValueError(
                "Grid must contain at least 3 points in each direction")

        self._alpha = scaling

        self._nGrid = int(np.ceil(scaling * vertPerDim))
        self._nCrop = int(self._nGrid / scaling)

        self._detCoeff = self.compute_coefficient(cov_fourier_callable)

        self._bc = [(BC.SINE, BC.SINE),
                    (BC.SINE, BC.COSINE),
                    (BC.COSINE, BC.SINE),
                    (BC.COSINE, BC.COSINE)]

    def compute_coefficient(self, ftrans_fcn):

        # inner degrees of freedom per dimension
        nDofInner = self._nGrid - 2

        fourierEval = np.zeros((nDofInner + 1, nDofInner + 1))

        for i in range(nDofInner + 1):
            for j in range(nDofInner + 1):
                s = [i / (2. * self._alpha), j / (2. * self._alpha)]
                fourierEval[i, j] = ftrans_fcn(s)

        return np.sqrt(fourierEval) / self._alpha

    def generate_realisation(self):

        realisation = np.zeros((self._nGrid, self._nGrid))

        # inner degrees of freedom per dimension
        n = self._nGrid - 2

        for rowBC, colBC in self._bc:

            coeff = standard_normal((n + 1, n + 1)) * self._detCoeff

            rowEval = srs.sin_series_rows if rowBC == BC.SINE else srs.cos_series_rows
            colEval = srs.sin_series_cols if colBC == BC.SINE else srs.cos_series_cols

            R = rowEval(coeff)
            realisation += colEval(R)

        return 0.5 * realisation[:self._nCrop, :self._nCrop]
