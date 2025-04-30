import numpy as np
from scipy.fft import fft2, ifft2, fftshift
from numpy.random import standard_normal

from yagregrf.sampling.interface import SamplingEngine
from yagregrf.utility.evaluation import norm


class CirculantEmbedding2DEngine(SamplingEngine):

    def __init__(self, cov_ptw_callable, vertPerDim,
                 domExt=1., autotunePadding=True, maxPadding=512, tol=1e-12):

        self._nGrid = vertPerDim  # matches n1 = n2
        self._n = vertPerDim + 1
        self._h = domExt / vertPerDim
        self._padding = 0
        self._tol = tol

        self._performAutotune = autotunePadding
        self._maxPadding = maxPadding

        if self._padding > self._maxPadding:
            raise RuntimeError(
                "initial CE padding larger than maximal allowed padding")

        self._coeff = self._compute_fft_coefficient(cov_ptw_callable)

    def _compute_reduced_covariance(self, cov_ptw_callable):

        nExt = self._n + self._padding
        nRed = 2 * nExt - 1

        redCov = np.zeros((nRed, nRed))

        for i in range(nRed):
            for j in range(nRed):
                x = (i - (nExt - 1)) * self._h
                y = (j - (nExt - 1)) * self._h
                redCov[i, j] = cov_ptw_callable(norm([x, y]))

        return redCov

    def _compute_lambda(self, cov_ptw_callable):

        redCovExt = self._compute_reduced_covariance(cov_ptw_callable)

        nExt = self._n + self._padding
        redCovTilde = np.zeros((2 * nExt, 2 * nExt))
        redCovTilde[1:2 * nExt, 1:2 * nExt] = redCovExt
        redCovTilde = fftshift(redCovTilde)

        NExt = (2 * nExt)**2
        return NExt * ifft2(redCovTilde)

    def _embedding_is_positive_definite(self, freqs):

        d = freqs.ravel()

        dReal = np.real(d)
        dImag = np.imag(d)

        return np.all(dReal >= 0.) and np.all(np.abs(dImag) < self._tol)

    def _determine_valid_lambda(self, cov_ptw_callable):

        isPosDef = False
        nextPadding = 2

        while not isPosDef and nextPadding <= self._maxPadding:
            print(f"check definiteness for padding={nextPadding}")
            self._padding = nextPadding

            Lambda = self._compute_lambda(cov_ptw_callable)
            isPosDef = self._embedding_is_positive_definite(Lambda)

            if not isPosDef:
                print("...still indefinite")

            nextPadding *= 2

        if not isPosDef:
            raise RuntimeError(
                f"Exceeded maximal padding of {self._maxPadding}")

        return Lambda

    def _determine_minimal_padding(self, cov_ptw_callable, maxBisections=4):

        print("determining minimal padding")

        initPadding = self._padding

        upper = self._padding
        lower = 0
        iteration = 0

        while upper > lower and iteration < maxBisections:

            mid = int(np.rint(0.5 * (lower + upper)))
            self._padding = mid

            Lambda = self._compute_lambda(cov_ptw_callable)
            isPosDef = self._embedding_is_positive_definite(Lambda)

            if isPosDef:
                upper = mid
            else:
                lower = mid
                self._padding = upper

            iteration += 1

        if isPosDef:
            return Lambda

        else:

            print("no smaller padding possible")

            self._padding = initPadding
            Lambda = self._compute_lambda(cov_ptw_callable)

            return Lambda

    def _compute_fft_coefficient(self, cov_ptw_callable):

        Lambda = self._compute_lambda(cov_ptw_callable)

        if not self._embedding_is_positive_definite(Lambda):

            print("initial embedding is not positive definite.")

            Lambda = self._determine_valid_lambda(cov_ptw_callable)

            if self._performAutotune:
                Lambda = self._determine_minimal_padding(cov_ptw_callable)

        print(f"Padding used: {self._padding}")

        return np.sqrt(np.real(Lambda))

    def generate_realisation(self):

        nExt = self._coeff.shape[0]
        NExt = nExt ** 2

        xi = standard_normal((nExt, nExt)) + 1.j * \
            standard_normal((nExt, nExt))
        V = self._coeff * xi

        z = fft2(V) / np.sqrt(NExt)

        x = np.real(z)[:self._nGrid, :self._nGrid]
        y = np.imag(z)[:self._nGrid, :self._nGrid]

        return x, y


class ApproximateCirculantEmbedding2DEngine(CirculantEmbedding2DEngine):

    def __init__(self, cov_ptw_callable, vertPerDim, domExt=1.):

        super().__init__(cov_ptw_callable, vertPerDim, domExt, autotunePadding=False)

    def _determine_valid_lambda(self, cov_ptw_callable):

        Lambda = self._compute_lambda(cov_ptw_callable)

        LambdaReal = np.real(Lambda)
        LambdaReal[LambdaReal < self._tol] = 0.

        return LambdaReal
