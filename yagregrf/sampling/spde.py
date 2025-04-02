import numpy as np
import fenics as fncs

from scipy.special import gamma
from numpy.random import standard_normal

from yagregrf.sampling.interface import SamplingEngine


def on_dirichlet_boundary_2d(x, dir0, dir1, delta, TOL=1e-8):

    if (dir0):
        if (x[0] + delta < TOL or x[0] > 1. + delta - TOL):
            return True

    if (dir1):
        if (x[1] + delta < TOL or x[1] > 1. + delta - TOL):
            return True

    return False


def white_noise_factor(M, V):

    H = np.zeros(V.dim())
    H[0] = np.sum(M.getrow(0)[1])
    H[V.dim() - 1] = np.sum(M.getrow(V.dim() - 1)[1])

    for i in range(1, V.dim() - 1):
        H[i] = np.sum(M.getrow(i)[1])

    return 1. / np.sqrt(H)


class SPDEEngine2d(SamplingEngine):

    DIM = 2

    def __init__(self, corrLength, nu, nVertPerDim,
                 alpha, useDirBC=[False, False]):

        fncs.set_log_level(fncs.LogLevel.ERROR)

        self._nGrid = nVertPerDim

        beta = 0.5 * nu + 0.25 * SPDEEngine2d.DIM
        self._betaInt = int(np.rint(0.5 * nu + 0.25 * SPDEEngine2d.DIM))

        if np.abs(beta - self._betaInt) > 1e-8:
            raise NotImplementedError(
                f"Only integer values of beta are supported. Received: {beta}")
        if corrLength < 1e-12:
            raise ValueError(
                f"Received invalid correlation length: {corrLength}")
        if nu < 0.5:
            raise ValueError(
                f"Received invalid smoothness parameter: {nu}")

        kappa = np.sqrt(2. * nu) / corrLength

        # oversamling width
        delta = 0.5 * (alpha - 1.)

        lowerLeft = fncs.Point(-delta, -delta)
        upperRight = fncs.Point(1. + delta, 1. + delta)

        self._mesh = fncs.RectangleMesh(
            lowerLeft, upperRight, nVertPerDim - 1, nVertPerDim - 1)
        self._varScaling = np.sqrt(
            4. * np.pi * gamma(1. + nu) / gamma(nu) * np.power(kappa, 2. * nu))

        self._V = fncs.FunctionSpace(self._mesh, "Lagrange", 1)

        u = fncs.TrialFunction(self._V)
        v = fncs.TestFunction(self._V)
        R = fncs.Constant(kappa * kappa)

        a = fncs.inner(fncs.grad(u), fncs.grad(v)) * \
            fncs.dx + R * u * v * fncs.dx

        dirBCVal = fncs.Constant(0.)

        self._bc = fncs.DirichletBC(
            self._V, dirBCVal, lambda x: on_dirichlet_boundary_2d(
                x, useDirBC[0], useDirBC[1], delta))

        A = fncs.assemble(a)
        self._bc.apply(A)

        self._solver = fncs.LUSolver(A, 'umfpack')
        self._solver.parameters['symmetric'] = True

        m = u * v * fncs.dx

        M = fncs.assemble(m)
        self._bc.apply(M)

        self._H = white_noise_factor(M, self._V)

    @property
    def mesh(self):
        return self._mesh

    def generate_realisation(self):

        fncs.set_log_level(fncs.LogLevel.ERROR)

        v = fncs.TestFunction(self._V)

        f = fncs.Function(self._V)
        u = fncs.Function(self._V)

        fDofs = self._varScaling * self._H * standard_normal(self._V.dim())
        f.vector()[:] = fDofs

        l = f * v * fncs.dx
        b = fncs.assemble(l)

        self._solver.solve(u.vector(), b)

        biCounter = self._betaInt

        while (biCounter > 1):

            rhs = fncs.Function(self._V)
            rhs.assign(u)

            l = rhs * v * fncs.dx
            b = fncs.assemble(l)
            self._bc.apply(b)

            self._solver.solve(u.vector(), b)

            biCounter -= 1

        return u.compute_vertex_values(
            self._mesh).reshape(self._nGrid, self._nGrid)
