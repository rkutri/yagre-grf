import numpy as np


class MarginalVarianceAccumulator:

    def __init__(self, dim):

        self._n = 0
        self._mean = np.zeros(dim)
        self._m2 = np.zeros(dim)
        self._dim = dim

    @property
    def marginalVariance(self):

        if self._n == 0:
            return np.zeros(self._dim)
        return self._m2 / self._n

    def update(self, x):
        self._n += 1
        delta = x - self._mean
        self._mean += delta / self._n
        self._m2 += delta * (x - self._mean)

    def clear(self):
        self._n = 0
        self._mean = np.zeros(self._dim)
        self._m2 = np.zeros(self._dim)


class CovarianceAccumulator:

    def __init__(self, dim):

        self._n = 0
        self._mean = np.zeros(dim)
        self._cov = np.zeros((dim, dim))

        self._dim = dim

    @property
    def covariance(self):
        return self._cov

    def update(self, x):

        delta = x - self._mean

        self._n += 1
        self._mean += delta / self._n
        self._cov += (np.outer(delta, x - self._mean) - self._cov) / self._n

    def clear(self):

        self._n = 0
        self._mean = np.zeros(self._dim)
        self._cov = np.zeros((self._dim, self._dim))


# Test
if __name__ == "__main__":

    print("testing marginal variance accumulation")

    DIM = 21

    np.random.seed(42)
    samples = np.random.randn(10000, DIM)

    varAcc = MarginalVarianceAccumulator(DIM)

    for sample in samples:
        varAcc.update(sample)

    trueVar = np.var(samples, axis=0, ddof=0)

    assert np.allclose(varAcc.marginalVariance, trueVar,
                       atol=1e-8), "Variance mismatch!"

    print("test passed")

    print("testing covariance accumulation")

    DIM = 21

    samples = np.random.randn(10000, DIM)  # Generate data

    cov = CovarianceAccumulator(DIM)

    for sample in samples:
        cov.update(sample)

    trueCov = np.cov(samples, rowvar=False, bias=True)

    assert np.allclose(cov.covariance, trueCov,
                       atol=1e-8), "Covariance mismatch!"

    print("test passed")
