import numpy as np
import scipy.sparse as sp
import fenics as fncs
import covariance_functions as covs

from numpy.random import standard_normal
from sksparse.cholmod import cholesky
from scipy.special import gamma
from scipy.sparse.linalg import spsolve
from dolfin.fem.norms import errornorm


def on_dirichlet_boundary(x, dir0, dir1, delta, TOL=1e-7):

    if (dir0):
        if (x[0] + delta < TOL or x[0] > 1.0 + delta - TOL):
            return True

    if (dir1):
        if (x[1] + delta < TOL or x[1] > 1.0 + delta - TOL):
            return True

    return False


def white_noise_factor(M, V):

    H = np.zeros(V.dim())
    H[0] = np.sum(M.getrow(0)[1])
    H[V.dim() - 1] = np.sum(M.getrow(V.dim() - 1)[1])

    for i in range(1, V.dim() - 1):
        H[i] = np.sum(M.getrow(i)[1])

    return 1. / np.sqrt(H)


def extract_cropped_diagonal_from_mesh(fncsMesh, osWidth):

    coords = fncsMesh.coordinates()

    def is_on_diagonal(point):
        return np.isclose(point[0], point[1], atol=1e-10)

    diagonalPoints = np.array([pt for pt in coords if is_on_diagonal(pt)])
    diagonalPoints = diagonalPoints[diagonalPoints[:, 0].argsort()]

    diagonal = np.linalg.norm(
        diagonalPoints[1:] - diagonalPoints[:-1], axis=1).cumsum()
    diagonal = np.insert(diagonal, 0, 0)

    if osWidth == 0:
        return diagonal
    else:
        return diagonal[osWidth:-osWidth]
