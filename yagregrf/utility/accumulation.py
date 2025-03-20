import numpy as np


class CovarianceAccumulator:

    def __init__(self, dim):

        self._n = 0
        self._mean = np.zeros(dim)
        self._cov = np.zeros((dim, dim))

    @property
    def covariance(self):
        return self._cov

    def update(self, x):

        delta = x - self._mean

        self._n += 1
        self._mean += delta / self._n
        self._cov += (np.outer(delta, x - self._mean) - self._cov) / self._n


# Test
if __name__ == "__main__":

    print("testing covariance accumulation")

    DIM = 21

    np.random.seed(42)
    samples = np.random.randn(10000, DIM)  # Generate data

    cov = CovarianceAccumulator(DIM)

    for sample in samples:
        cov.update(sample)

    trueCov = np.cov(samples, rowvar=False, bias=True)

    assert np.allclose(cov.covariance, trueCov,
                       atol=1e-8), "Covariance mismatch!"

    print("test passed")
