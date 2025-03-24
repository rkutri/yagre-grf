from numpy import sqrt, zeros
from numpy.random import standard_normal

from yagregrf.sampling.randomField import RandomField
from yagregrf.sampling.interface import SamplingEngine
from yagregrf.utility.series import sin_series, cos_series
from yagregrf.utility.evaluation import norm


class DNAFourierEngine1d(SamplingEngine):

    def __init__(self, cov_fourier_callable, nVertices, scaling=1.0):

        if nVertices < 3:
            raise ValueError("Grid must contain at least 3 points")

        self._nGrid = nVertices

        self._alpha = scaling
        self._detCoeff = self.compute_coefficient(cov_fourier_callable)

    def compute_coefficient(self, ftrans_fcn):
        return sqrt([ftrans_fcn(m / (2. * self._alpha))
                     for m in range(self._nGrid - 1)])

    def generate_realisation(self):

        # inner degrees of freedom (two boundary vertices)
        nDofInner = self._nGrid - 2

        dirEval = sin_series(standard_normal(nDofInner + 1) * self._detCoeff)
        neuEval = cos_series(standard_normal(nDofInner + 1) * self._detCoeff)

        return (dirEval + neuEval) / sqrt(2. * self._alpha)


class DNAFourierEngine2d(SamplingEngine):

    def __init__(self, cov_fourier_callable, vertPerDim, scaling=1.0):

        if vertPerDim < 3:
            raise ValueError(
                "Grid must contain at least 3 points in each direction")

        self._nGrid = vertPerDim
        self._alpha = scaling

        self._detCoeff = self.compute_coefficient(cov_fourier_callable)

        self._bc = [(sin_series, sin_series),
                    (sin_series, cos_series),
                    (cos_series, sin_series),
                    (cos_series, cos_series)]

    def compute_coefficient(self, ftrans_fcn):

        # inner degrees of freedom per dimension
        nDofInner = self._nGrid - 2

        fourierEval = zeros((nDofInner + 1, nDofInner + 1))

        for i in range(nDofInner):
            for j in range(nDofInner):
                fourierEval[i, j] = ftrans_fcn(
                    norm([i, j]) / (2. * self._alpha))

        return sqrt(fourierEval)

    def generate_realisation(self):

        # inner degrees of freedom per dimension
        nDofInner = self._nGrid - 2

        realisation = zeros((nDofInner + 2, nDofInner + 2))

        for row_transform_fcn, col_transform_fcn in self._bc:

            rfEval = zeros((nDofInner + 2, nDofInner + 2))
            coeff = standard_normal(
                (nDofInner + 1, nDofInner + 1)) * self._detCoeff

            for row in range(nDofInner + 1):
                rfEval[row, :] = row_transform_fcn(coeff[row, :])
            for col in range(nDofInner + 2):
                rfEval[:, col] = col_transform_fcn(rfEval[:-1, col])

            realisation += rfEval

        return realisation / (2. * self._alpha)
