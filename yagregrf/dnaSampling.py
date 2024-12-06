import logging
import utility

from numpy import sqrt, zeros
from numpy.random import standard_normal
from interface import GaussianRandomField
from series import sin_series, cos_series


# set up logger
dnaGRFLogger = logging.getLogger(__name__)
dnaGRFLogger.setLevel(logging.INFO)

consoleHandler = logging.StreamHandler()

formatter = logging.Formatter('%(levelname)s: %(message)s')
consoleHandler.setFormatter(formatter)

dnaGRFLogger.addHandler(consoleHandler)


class DNAGaussianRandomField1d(GaussianRandomField):

    def __init__(self, cov_ftrans, nGrid):

        if nGrid < 3:
            raise ValueError("1D grid must contain at least 3 points")

        self._nDof = nGrid - 2

        self._detCoeff = sqrt([cov_ftrans(0.5 * m) for m in range(nGrid - 1)])

    def generate(self, nSamp, nPrintIntervals=10):

        # print info every pInterval samples
        pInterval = nSamp // nPrintIntervals

        realisations = []

        for n in range(nSamp):

            if n == 0:
                dnaGRFLogger.info(
                    f"Start generating {nSamp} 1d CRF realisations")

            # log information on number of generated samples
            if nSamp > nPrintIntervals:
                if n % pInterval == 0 and n != 0:
                    dnaGRFLogger.info(f"{n} realisations generated")

            dirEval = sin_series(
                standard_normal(self._nDof + 1) * self._detCoeff)
            neuEval = cos_series(
                standard_normal(self._nDof + 1) * self._detCoeff)

            realisations += [(dirEval + neuEval) / sqrt(2.)]

        return realisations


class DNAGaussianRandomField2d(GaussianRandomField):

    def __init__(self, cov_ftrans, nGridPerDim):

        if nGridPerDim < 3:
            raise ValueError("1D grid must contain at least 3 points")

        self._nDof = nGridPerDim - 2

        fourierEval = zeros((self._nDof + 1, self._nDof + 1))

        for i in range(self._nDof):
            for j in range(self._nDof):
                fourierEval[i, j] = cov_ftrans(0.5 * utility.norm([i, j]))

        self._detCoeff = sqrt(fourierEval)

    def generate(self, nSamp, nPrintIntervals=10):

        # print info every pInterval samples
        pInterval = nSamp // nPrintIntervals

        realisations = []

        for n in range(nSamp):

            if n == 0:
                dnaGRFLogger.info(
                    f"Start generating {nSamp} 2d CRF realisations")

            # log information on number of generated samples
            if nSamp > nPrintIntervals:
                if n % pInterval == 0 and n != 0:
                    dnaGRFLogger.info(f"{n} realisations generated")

            # Dirichlet - Dirichlet
            ddEval = zeros((self._nDof + 2, self._nDof + 2))
            ddCoeff = standard_normal(
                (self._nDof + 1, self._nDof + 1)) * self._detCoeff

            for row in range(self._nDof + 1):
                ddEval[row, :] = sin_series(ddCoeff[row, :])
            for col in range(self._nDof + 2):
                ddEval[:, col] = sin_series(ddEval[:-1, col])

            # Dirichlet - Neumann
            dnEval = zeros((self._nDof + 2, self._nDof + 2))
            dnCoeff = standard_normal(
                (self._nDof + 1, self._nDof + 1)) * self._detCoeff

            for row in range(self._nDof + 1):
                dnEval[row, :] = sin_series(dnCoeff[row, :])
            for col in range(self._nDof + 2):
                dnEval[:, col] = cos_series(dnEval[:-1, col])

            # Neumann - Dirichlet
            ndEval = zeros((self._nDof + 2, self._nDof + 2))
            ndCoeff = standard_normal(
                (self._nDof + 1, self._nDof + 1)) * self._detCoeff

            for row in range(self._nDof + 1):
                ndEval[row, :] = cos_series(ndCoeff[row, :])
            for col in range(self._nDof + 2):
                ndEval[:, col] = sin_series(ndEval[:-1, col])

            # Neumann - Neumann
            nnEval = zeros((self._nDof + 2, self._nDof + 2))
            nnCoeff = standard_normal(
                (self._nDof + 1, self._nDof + 1)) * self._detCoeff

            for row in range(self._nDof + 1):
                nnEval[row, :] = cos_series(nnCoeff[row, :])
            for col in range(self._nDof + 2):
                nnEval[:, col] = cos_series(nnEval[:-1, col])

            realisations.append(0.5 * (ddEval + dnEval + ndEval + nnEval))

        return realisations
