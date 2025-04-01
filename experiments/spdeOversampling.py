import os
import csv
import sys

import numpy as np

import experiments.scriptUtility as util

from yagregrf.sampling.randomField import RandomField
from yagregrf.sampling.spde import SPDEEngine2d
from yagregrf.sampling.dnaSPDE import DNASPDEEngine2d
from yagregrf.sampling.dnaFourier import DNAFourierEngine2d
from yagregrf.utility.covariances import matern_ptw, matern_fourier_ptw
from yagregrf.utility.accumulation import CovarianceAccumulator
from yagregrf.utility.evaluation import evaluate_isotropic_covariance_1d
from filename import create_data_string

# Parameters
DIM = 2
ell = 0.2
nu = 1.
nSamp = int(1e5)


# Check if an argument was provided
if len(sys.argv) < 2:
    print("Usage: python3 script.py <filenameID> (must be two digits)")
    sys.exit(1)

filenameID = sys.argv[1]

# Ensure it's exactly two digits
if not (filenameID.isdigit() and len(filenameID) == 2):
    print("Error: filenameID must be exactly two digits (e.g., '02', '15').")
    sys.exit(1)

print(f"Filename ID set to: '{filenameID}'")


def print_sampling_progress(n, nSamp, nUpdates=5):

    assert nSamp > nUpdates

    if n % (nSamp // nUpdates) == 0:
        if n == 0:
            print("Start sampling")
        else:
            print(f"{n} realisations computed")


def cov_ftrans_callable(s):
    return matern_fourier_ptw(s, ell, nu, DIM)


dofPerDim = [8, 16, 32, 64]
oversampling = [1., 1.2, 1.4, 1.6, 1.8]

kappa = np.sqrt(2 * nu) / ell
beta = 0.5 * (1 + nu)


dataDir = 'data'
dataSubDir = 'oversampling'

outDir = os.path.join(dataDir, dataSubDir)
os.makedirs(outDir, exist_ok=True)

filename = os.path.join(
    outDir,
    create_data_string(ell, nu, nSamp, "os") + f"_{filenameID}.csv")

maxErrorDNAFourier = []
maxErrorDNASPDE = []
maxErrorSPDE = [[] for _ in oversampling]

for nDof in dofPerDim:
    batchsize = 1
    print(f"\n\nRunning experiments with {nDof} dofs per dimension")

    print(f"\n\n- Running DNA Sampling in Fourier basis")
    dnaFourierRF = RandomField(DNAFourierEngine2d(cov_ftrans_callable, nDof))
    dnaFourierCov = CovarianceAccumulator(nDof)

    for n in range(nSamp):

        print_sampling_progress(n, nSamp)

        realisation = dnaFourierRF.generate(batchsize)[0]
        diagSlice = util.extract_diagonal_slice(realisation)

        dnaFourierCov.update(diagSlice)

    print(f"\n\n- Running DNA Sampling using SPDE approach")
    dnaSPDERF = RandomField(DNASPDEEngine2d(ell, nu, nDof, 1.))
    dnaSPDECov = CovarianceAccumulator(nDof)

    for n in range(nSamp):

        print_sampling_progress(n, nSamp)

        realisation = dnaSPDERF.generate(batchsize)[0]
        diagSlice = util.extract_diagonal_slice(realisation)

        dnaSPDECov.update(diagSlice)

    print(f"\n\n- Sampling using SPDE approach with oversampling")
    spdeCovList = [CovarianceAccumulator(nDof) for _ in oversampling]

    for alphaIdx, alpha in enumerate(oversampling):

        print(f"\noversampling with alpha = {alpha}")

        nOsDof = int(np.ceil(alpha * nDof))
        if nOsDof % 2 == 1:
            nOsDof += 1
        print(nOsDof)
        osWidth = (nOsDof - nDof) // 2

        print(f"oversampling width: {osWidth} degrees of freedom\n")

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
        return matern_ptw(r, ell, nu)

    diagonalGrid = util.extract_diagonal_from_mesh(dnaSPDERF.engine.mesh)
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
    writer.writerow(["DNA_fourier"] + maxErrorDNAFourier)
    writer.writerow(["DNA_spde"] + maxErrorDNASPDE)

    for i, alpha in enumerate(oversampling):
        writer.writerow([f"spde_alpha{int(10*alpha)}"] + maxErrorSPDE[i])
