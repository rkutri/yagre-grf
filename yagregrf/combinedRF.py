import logging

from numpy import sqrt
from numpy.random import standard_normal
from interface import GaussianRandomField
from series import sin_series, cos_series


# set up logger
crf1dLogger = logging.getLogger(__name__)
crf1dLogger.setLevel(logging.INFO)

consoleHandler = logging.StreamHandler()

formatter = logging.Formatter('%(levelname)s: %(message)s')
consoleHandler.setFormatter(formatter)

crf1dLogger.addHandler(consoleHandler)


class CombinedRandomField1d(GaussianRandomField):

    def __init__(self, cov_ftrans, nGrid):

        if nGrid < 3:
            raise ValueError("1D grid must contain at least 3 points")

        self._nDof = nGrid - 2

        self._detCoeff = sqrt([cov_ftrans(0.5 * m) for m in range(nDof + 1)])

    def generate(self, nSamp, nPrintIntervals=10):

        # print info every pInterval samples
        pInterval = nSamp // nPrintIntervals

        samples = []

        for n in range(nSamp):

            if n == 0:
                crf1dLogger.info(
                    f"Start generating {nSamp} 1d CRF realisations")

            # log information on number of generated samples
            if nSamp > printIntervals:
                if n % pInterval == 0:
                    crf1dLogger.info(f"{n} realisations generated")

            dirEval = sin_series(
                standard_normal(
                    self._nDof +
                    1) *
                self._detCoeff)
            neuEval = cos_series(
                standard_normal(
                    self._nDof +
                    1) *
                self._detCoeff)

            samples += [(dirEval + neuEval) / sqrt(2.)]

        return samples
