import numpy as np
import dolfin as df
from scipy.special import gamma
from numpy.random import standard_normal
from yagregrf.sampling.interface import SamplingEngine


def on_dirichlet_boundary_2d(x, dir0, dir1, delta, TOL=1e-8):
    if dir0:
        if x[0] + delta < TOL or x[0] > 1. + delta - TOL:
            return True
    if dir1:
        if x[1] + delta < TOL or x[1] > 1. + delta - TOL:
            return True
    return False


def white_noise_factor(M, V):
    H = np.zeros(V.dim())
    for i in range(V.dim()):
        H[i] = np.sum(M.getrow(i)[1])
    return 1. / np.sqrt(H)


class SPDEEngine2d(SamplingEngine):

    DIM = 2

    def __init__(self, variance, corrLength, nu, nVertPerDim, alpha,
                 useDirBC=[False, False], cacheFactorisation=True, useDirectSolver=True):

        df.set_log_level(df.LogLevel.ERROR)

        self._sd = np.sqrt(variance)
        self._nGrid = nVertPerDim
        self._cacheFactor = cacheFactorisation

        beta = 0.5 * nu + 0.25 * SPDEEngine2d.DIM
        self._betaInt = int(np.rint(beta))

        if np.abs(beta - self._betaInt) > 1e-8:
            raise NotImplementedError(
                f"Only integer values of beta are supported. Received: {beta}")
        if corrLength < 1e-12:
            raise ValueError(
                f"Received invalid correlation length: {corrLength}")
        if nu < 0.5:
            raise ValueError(f"Received invalid smoothness parameter: {nu}")

        kappa = np.sqrt(2. * nu) / corrLength
        delta = 0.5 * (alpha - 1.)

        lowerLeft = df.Point(-delta, -delta)
        upperRight = df.Point(1. + delta, 1. + delta)

        self._mesh = df.RectangleMesh(
            lowerLeft,
            upperRight,
            nVertPerDim - 1,
            nVertPerDim - 1)
        self._varScaling = np.sqrt(
            4. * np.pi * gamma(1. + nu) / gamma(nu) * np.power(kappa, 2. * nu)
        )

        self._V = df.FunctionSpace(self._mesh, "Lagrange", 1)

        u = df.TrialFunction(self._V)
        v = df.TestFunction(self._V)
        R = df.Constant(kappa * kappa)

        self._a = df.inner(df.grad(u), df.grad(v)) * df.dx + R * u * v * df.dx
        dirBCVal = df.Constant(0.)

        self._bc = df.DirichletBC(
            self._V,
            dirBCVal,
            lambda x: on_dirichlet_boundary_2d(
                x, useDirBC[0], useDirBC[1], delta)
        )

        if useDirectSolver:
            self._solver = df.LUSolver("umfpack")
        else:
            self._solver = df.KrylovSolver("cg", "hypre_amg")

        if self._cacheFactor:

            self._A = df.assemble(self._a)
            self._A.ident_zeros()
            self._bc.apply(self._A)

            self._solver.set_operator(self._A)

        else:
            self._A = None

        m = u * v * df.dx
        M = df.assemble(m)
        self._H = white_noise_factor(M, self._V)

        self._sol = df.Function(self._V)
        self._f = df.Function(self._V)
        self._l = self._f * v * df.dx
        self._b = df.Vector()
        df.assemble(self._l, tensor=self._b)

    @property
    def mesh(self):
        return self._mesh

    def generate_realisation(self):

        if not self._cacheFactor:

            self._A = df.assemble(self._a)
            self._A.ident_zeros()
            self._bc.apply(self._A)

            self._solver.set_operator(self._A)

        fDofs = self._f.vector().get_local()
        fDofs[:] = self._sd * self._varScaling * \
            self._H * standard_normal(self._V.dim())
        self._f.vector().set_local(fDofs)
        self._f.vector().apply("insert")

        df.assemble(self._l, tensor=self._b)
        self._bc.apply(self._b)

        self._solver.solve(self._sol.vector(), self._b)

        biCounter = self._betaInt
        while biCounter > 1:

            self._f.assign(self._sol)

            df.assemble(self._l, tensor=self._b)
            self._bc.apply(self._b)

            self._solver.solve(self._sol.vector(), self._b)
            biCounter -= 1

        return self._sol.compute_vertex_values(
            self._mesh).reshape(self._nGrid, self._nGrid)
