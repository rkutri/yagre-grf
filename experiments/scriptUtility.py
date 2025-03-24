import numpy as np


def extract_diagonal_from_mesh(fncsMesh, aTol=1e-8):

    coords = fncsMesh.coordinates()

    def is_on_diagonal(point):
        return np.isclose(point[0], point[1], atol=aTol)

    diagonalPoints = np.array([pt for pt in coords if is_on_diagonal(pt)])
    diagonalPoints = diagonalPoints[diagonalPoints[:, 0].argsort()]

    diagonal = np.linalg.norm(
        diagonalPoints[1:] - diagonalPoints[:-1], axis=1).cumsum()
    diagonal = np.insert(diagonal, 0, 0)

    return diagonal


def extract_diagonal_slice(matrix):

    n = matrix.shape[0]
    assert matrix.shape[1] == n

    return np.array([matrix[i, i] for i in range(n)])
