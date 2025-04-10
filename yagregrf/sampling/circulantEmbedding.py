import numpy as np
from scipy.fft import fft2, ifft2, fftshift
from numpy.random import standard_normal
import matplotlib.pyplot as plt
from yagregrf.sampling.interface import SamplingEngine


# Adapted from Algorithm 7.6 of
# "An Introduction to Computational Stochastic PDEs"
# by G.J.Lord, C.E. Powell and T. Shardlow
class CirculantEmbedding2DEngine(SamplingEngine):

    def __init__(self, cov_fourier_callable, vertPerDim,
                 domExt=1., autotunePadding=True):

        self._n = vertPerDim
        self._h = domExt / vertPerDim

        self._N = None
        self._coeff = None

        self._performAutotune = autotunePadding
        self._maxPadding = 8 * vertPerDim

    def _compute_reduced_covariance(self, cov_fourier_callable):

        Nred = 2. * self._n - 1.

        self._reducedCov = np.zeros((Nred, Nred))

        for i in range(Nred):
            for j in range(Nred):

                hx = (i - (self._n - 1)) * self._h
                hy = (j - (self._n - 1)) * self._h

                self._reducedCov[i, j] = cov_fourier_callable(hx, hy)

        return

    def _embedding_is_positive_definite(self, freqs, tol=1e-12):

        d = freqs.ravel()
        dNeg = np.maximum(-d, 0.)

        return np.max(dNeg) > tol

    def _determine_valid_padding(self, redCov):
        """
        assuming the embedding without padding is not positive definite
        """

        isPosDef = False

        padding = 0
        nextPadding = 2

        while not isPosDef and nextPadding <= self._maxPadding:

            padding = nextPadding

            nExt = self._n + padding
            NExt = nExt**2

            redCovTilde = np.zeros((2 * nExt, 2 * nExt))
            redCovTilde[1:2 * nExt, 1:2 * nExt] = redCov
            redCovTilde = fftshift(redCovTilde)

            Lambda = NExt * ifft2(redCovTilde)
            isPosDef = self._embedding_is_positive_definite(Lambda)

            nextPadding = padding * 2

        if not isPosDef:
            raise RuntimeError(
                f"Exceeded maximal padding of {self._maxPadding}")

        return Lambda, padding

    def _determine_minimal_padding(redCov, initPadding, maxBisections=4):
        """
        assumes that the initial padding leads to positive definiteness
        """

        upper = initPadding

        lower = 0
        mid = 0

        iteration = 0

        while upper > lower and iteration < maxBisections:

            mid = int(0.5 * (lower + upper))

            nExt = self._n + mid
            NExt = nExt**2

            midRedCov = np.zeros((2 * nExt, 2 * nExt))
            midRedCov[1:2 * nExt, 1:2 * Next] = redCov
            midRedCov = fftshift(midRedCov)

            Lambda = NExt * ifft2(midRedCov)
            isPosDef = self._embedding_is_positive_definite(Lambda)

            if isPosDef:
                upper = mid
            else:
                lower = mid

            iteration += 1

        return Lambda, upper

    def _compute_fft_frequencies(self, cov_fourier_callable):

        redCov = self._compute_reduced_covariance(cov_fourier_callable)

        self._N = self._n**2
        Lambda = self._N * ifft2(redCov)

        if not self._embedding_is_positive_definite(Lambda):

            Lambda, padding = self._determine_valid_padding(redCov)

            # use bisection reduce the padding while remaining positive
            # definite
            if self._performAutotune:
                Lambda, padding = self._determine_minimal_padding(
                    redCov, padding)

        self._N = (self._n + padding)**2
        self._coeff = np.sqrt(Lambda)
        return

    def generate_realisation(self):

        xi = standard_normal((self._n, self._n)) \
            + 1.j * standard_normal((self._n, self._n))
        V = self._coeff * xi

        z = fft2(V) / np.sqrt(self._N)
        z = z.ravel()

        x = np.real(z)
        y = np.imag(z)

        return x, y


class ApproximateCirculantEmbeddingEngine(CirculantEmbeddingEngine):
    pass
