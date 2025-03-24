import os
import csv

import numpy as np

import experiments.scriptUtility as util

from yagregrf.sampling.randomField import RandomField
from yagregrf.sampling.spde import SPDEEngine2d
from yagregrf.sampling.dnaSPDE import DNASPDEEngine2d
from yagregrf.sampling.dnaFourier import DNAFourierEngine2d
from yagregrf.utility.covariances import matern_fourier_ptw
from yagregrf.utility.accumulation import CovarianceAccumulator


filenameID = "01"


def print_sampling_progress(n, nSamp, nUpdates=5):

    assert nSamp > nUpdates

    if n % (nSamp // nUpdates) == 0:
        if n == 0:
            print("Start sampling")
        else:
            print(f"{n} realisations computed")


DIM = 2
ell = 0.2
nu = 1.


def cov_ftrans_callable(s):
    return matern_fourier_ptw(s, ell, nu, DIM)


dofPerDim = [8, 16, 32]
oversampling = [1., 1.5]

kappa = np.sqrt(2 * nu) / ell
beta = 0.5 * (1 + nu)

nSamp = int(1e3)

dataDir = 'data'
dataSubDir = 'oversampling'

outDir = os.path.join(dataDir, dataSubDir)
os.makedirs(outDir, exist_ok=True)

filename = os.path.join(
    outDir,
    f"os_ell{int(100*ell)}_nu{int(nu)}_{nSamp // 1000}k_{filenameID}.csv")

maxErrorDNAFourier = []
maxErrorDNASPDE = []
maxErrorSPDE = [[] for _ in oversampling]

for nDof in dofPerDim:
    batchsize = 1
    print(f"\n\nRunning experiments with {nDof} dofs per dimension")

    print(f"\n\t- Running DNA Sampling in Fourier basis")
    dnaFourierRF = RandomField(DNAFourierEngine2d(cov_ftrans_callable, nDof))
    dnaFourierCov = CovarianceAccumulator(nDof)

    for n in range(nSamp):
        print_sampling_progress(n, nSamp)
        realisation = dnaFourierRF.generate(batchsize)[0]
        diagSlice = util.extract_diagonal_slice(realisation)
        dnaFourierCov.update(diagSlice)

    print(f"\n\t- Running DNA Sampling using SPDE approach")
    dnaSPDERF = RandomField(DNASPDEEngine2d(ell, nu, nDof, 1.))
    dnaSPDECov = CovarianceAccumulator(nDof)

    for n in range(nSamp):
        print_sampling_progress(n, nSamp)
        realisation = dnaSPDERF.generate(batchsize)[0]
        diagSlice = util.extract_diagonal_slice(realisation)
        dnaSPDECov.update(diagSlice)

    print(f"\n\t- Running Sampling using SPDE approach with oversampling")
    spdeCovList = [CovarianceAccumulator(nDof) for _ in oversampling]

    for alphaIdx, alpha in enumerate(oversampling):
        print(f"\t\t- oversampling with alpha = {alpha}")
        dofFactor = 2 * alpha - 1
        nOsDof = int(np.rint(dofFactor * nDof))
        osWidth = nOsDof - nDof // 2
        print(f"\t\t- oversampling width: {osWidth} degrees of freedom\n")

        spdeRF = RandomField(
            SPDEEngine2d(
                ell, nu, nOsDof, alpha, useDirBC=[False, False]
            )
        )
        spdeCov = spdeCovList[alphaIdx]

        for n in range(nSamp):
            print_sampling_progress(n, nSamp)
            realisation = spdeRF.generate(batchsize)[0]
            diagSlice = util.extract_diagonal_slice(realisation)
            if osWidth > 0:
                diagSlice = diagSlice[osWidth:-osWidth]
            spdeCov.update(diagSlice)

    def cov_fcn(r):
        return matern_covariance_ptw(r, ell, nu)

    diagonalGrid = util.extract_diagonal_from_mesh(dnaSPDERF.mesh)
    trueCov = evaluate_isotropic_covariance_1d(cov_fcn, diagonalGrid)

    maxErrorDNAFourier.append(
        np.max(
            np.abs(
                trueCov -
                dnaFourierCov.covariance)))
    maxErrorDNASPDE.append(np.max(np.abs(trueCov - dnaSPDECov.covariance)))

    for i, spdeCov in enumerate(spdeCovList):
        maxErrorSPDE[i].append(np.max(np.abs(trueCov - spdeCov.covariance)))

with open(filename, mode='w', newline='') as file:

    writer = csv.writer(file)

    # Header row
    writer.writerow(["Method"] + [str(nDof) for nDof in dofPerDim])

    # Data rows
    writer.writerow(["DNA-Fourier"] + maxErrorDNAFourier)
    writer.writerow(["DNA-SPDE"] + maxErrorDNASPDE)

    for i, alpha in enumerate(oversampling):
        writer.writerow([f"SPDE-alpha-{alpha}"] + maxErrorSPDE[i])
