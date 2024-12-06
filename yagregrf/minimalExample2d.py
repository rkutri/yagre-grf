from dnaSampling import DNAGaussianRandomField2d
from covariances import matern_fourier_ptw

import matplotlib.pyplot as plt

corrLength = 0.2
smoothness = 4.5


def cov_ftrans(s):
    return matern_fourier_ptw(s, corrLength, smoothness, 2)


dnaRF = DNAGaussianRandomField2d(cov_ftrans, 100)

s = dnaRF.generate(10)

fig, ax = plt.subplots(2, 2)

ax[0, 0].imshow(s[0])
ax[0, 1].imshow(s[1])
ax[1, 0].imshow(s[2])
ax[1, 1].imshow(s[3])

plt.show()
