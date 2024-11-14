import os
import csv

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from numpy.random import standard_normal
from scipy.fft import dct, dctn, dst, dstn

import covariance_functions as covs
import utility


filenameID = "01"


# domain
nDof = 1500

grid = np.linspace(0., 1., nDof + 2, endpoint=True)

# sampling
nSamp = int(1e5)

corrLength = [0.025, 0.05, 0.1, 0.2]

print("\n- grid points: " + str(nDof + 2))
print(f"sequence of correlation Lengths: {corrLength}")

smoothness = [0.5, 2, 8]

dataDir = 'data'
dataSubDir = 'cov'

outDir = os.path.join(dataDir, dataSubDir)

os.makedirs(outDir, exist_ok=True)

for alpha in [1., 2.]:

    print(f"ALPHA = {alpha}")
    print("-----------")

    filename = os.path.join(
        outDir,
        f"cov_error_data_n{nDof}_{nSamp // 1000}k_alpha{int(alpha)}_{filenameID}.csv")

    with open(filename, 'w') as file:

        writer = csv.writer(file)

        rowHeaders = ['covariance']

        for ell in corrLength:
            rowHeaders.append(f"ell={ell}")

        writer.writerow(rowHeaders)

        for nu in smoothness:

            csvRow = [f"nu={nu}"]

            for ell in corrLength:

                print(f"\nRunning for ell = {ell}, nu = {nu}\n")

                def cov_fcn(x):
                    return covs.matern_covariance_ptw(x, ell, nu)

                def cov_ftrans(s):
                    return covs.matern_fourier_ptw(s, ell, nu, 1)

                coeff = np.sqrt([cov_ftrans(0.5 * m) for m in range(nDof + 1)])

                sample = []

                for n in range(nSamp):

                    if (n % 10000 == 0):
                        print(str(n) + " realisations computed")

                    dirEval = utility.sin_series(
                        standard_normal(nDof + 1) * coeff)
                    neuEval = utility.cos_series(
                        standard_normal(nDof + 1) * coeff)

                    sample += [(neuEval + dirEval) / np.sqrt(2.)]

                print("finished sampling, computing statistics")

                trueCov = utility.evaluate_isotropic_covariance_1d(
                    cov_fcn, grid)
                sampCov = np.cov(sample, rowvar=False)

                csvRow.append(np.max(np.abs(trueCov - sampCov)))

            writer.writerow(csvRow)

        csvRow = ['gaussian']

        for ell in corrLength:

            print(f"\nRunning Gaussian covariance with ell = {ell}")

            def cov_fcn(x): return covs.gaussian_ptw(x, ell)
            def cov_ftrans(s): return covs.gaussian_fourier_ptw(s, ell)

            coeff = np.sqrt([cov_ftrans(0.5 * m) for m in range(nDof + 1)])

            sample = []

            for n in range(nSamp):

                if (n % 10000 == 0):
                    print(str(n) + " realisations computed")

                dirEval = utility.sin_series(
                    standard_normal(nDof + 1) * coeff)
                neuEval = utility.cos_series(
                    standard_normal(nDof + 1) * coeff)

                sample += [(neuEval + dirEval) / np.sqrt(2.)]

            print("finished sampling, computing statistics")

            trueCov = utility.evaluate_isotropic_covariance_1d(
                cov_fcn, grid)
            sampCov = np.cov(sample, rowvar=False)

            csvRow.append(np.max(np.abs(trueCov - sampCov)))

            print("success\n\n\n")

        writer.writerow(csvRow)

        csvRow = ['cauchy']

        for ell in corrLength:

            print(f"\nRunning Cauchy covariance with ell = {ell}")

            def cov_fcn(x): return covs.cauchy_ptw(x, ell)
            def cov_ftrans(s): return covs.cauchy_fourier_ptw(s, ell)

            coeff = np.sqrt([cov_ftrans(0.5 * m) for m in range(nDof + 1)])

            sample = []

            for n in range(nSamp):

                if (n % 10000 == 0):
                    print(str(n) + " realisations computed")

                dirEval = utility.sin_series(
                    standard_normal(nDof + 1) * coeff)
                neuEval = utility.cos_series(
                    standard_normal(nDof + 1) * coeff)

                sample += [(neuEval + dirEval) / np.sqrt(2.)]

            print("finished sampling, computing statistics")

            trueCov = utility.evaluate_isotropic_covariance_1d(
                cov_fcn, grid)
            sampCov = np.cov(sample, rowvar=False)

            csvRow.append(np.max(np.abs(trueCov - sampCov)))

            print("success\n\n\n")

        writer.writerow(csvRow)

        file.close()
