import numpy as np
import dolfin as df
from scipy.special import gamma
from numpy.random import standard_normal
from yagregrf.sampling.interface import SamplingEngine


import matplotlib.pyplot as plt
import fenics as fncs


def on_dirichlet_boundary_2D(x, dir0, dir1, delta, TOL=1e-8):
    if dir0 and (x[0] + delta < TOL or x[0] > 1. + delta - TOL):
        return True
    if dir1 and (x[1] + delta < TOL or x[1] > 1. + delta - TOL):
        return True
    return False


def white_noise_factor(M, V):
    H = np.zeros(V.dim())
    for i in range(V.dim()):
        H[i] = np.sum(M.getrow(i)[1])
    return 1. / np.sqrt(H)


class SPDEEngine2D(SamplingEngine):

    DIM = 2

    def __init__(self, variance, corrLength, nu, nVertPerDim, alpha,
                 useDirBC=[False, False], cacheFactorisation=True, useDirectSolver=True):

        df.set_log_level(df.LogLevel.ERROR)

        self._sd = np.sqrt(variance)
        self._nGrid = nVertPerDim
        self._cacheFactor = cacheFactorisation

        beta = 0.5 * nu + 0.25 * SPDEEngine2D.DIM
        self._betaInt = int(np.rint(beta))

        if np.abs(beta - self._betaInt) > 1e-8:
            raise NotImplementedError(
                f"Only integer values of beta are supported. Received: {beta}")
        if corrLength < 1e-12:
            raise ValueError(
                f"Received invalid correlation length: {corrLength}")
        if nu < 0.5:
            raise ValueError(f"Received invalid smoothness parameter: {nu}")

        self._kappa = np.sqrt(2. * nu) / corrLength
        delta = 0.5 * (alpha - 1.)

        lowerLeft = df.Point(-delta, -delta)
        upperRight = df.Point(1. + delta, 1. + delta)

        self._mesh = df.RectangleMesh(
            lowerLeft,
            upperRight,
            nVertPerDim - 1,
            nVertPerDim - 1)
        self._varScaling = np.sqrt(
            4. *
            np.pi *
            gamma(
                1. +
                nu) /
            gamma(nu) *
            np.power(
                self._kappa,
                2. *
                nu))

        self._V = df.FunctionSpace(self._mesh, "Lagrange", 1)
        self._u = df.TrialFunction(self._V)
        self._v = df.TestFunction(self._V)

        self._a = self._assemble_bilinear_form()

        self._bc = df.DirichletBC(
            self._V,
            df.Constant(0.),
            lambda x: on_dirichlet_boundary_2D(
                x, useDirBC[0], useDirBC[1], delta)
        )

        self._solver = df.LUSolver(
            "umfpack") if useDirectSolver else df.KrylovSolver("cg", "hypre_amg")

        if self._cacheFactor:
            self._A = df.assemble(self._a)
            self._A.ident_zeros()
            self._bc.apply(self._A)
            self._solver.set_operator(self._A)
        else:
            self._A = None

        m = self._u * self._v * df.dx
        M = df.assemble(m)
        self._H = white_noise_factor(M, self._V)

        self._sol = df.Function(self._V)
        self._f = df.Function(self._V)
        self._l = self._f * self._v * df.dx
        self._b = df.Vector()
        df.assemble(self._l, tensor=self._b)

    @property
    def mesh(self):
        return self._mesh

    @property
    def nDof(self):
        return self._V.dim()

    def _assemble_bilinear_form(self):
        R = df.Constant(self._kappa * self._kappa)
        return df.inner(df.grad(self._u), df.grad(self._v)) * \
            df.dx + R * self._u * self._v * df.dx

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

        for _ in range(self._betaInt - 1):
            self._f.assign(self._sol)
            df.assemble(self._l, tensor=self._b)
            self._bc.apply(self._b)
            self._solver.solve(self._sol.vector(), self._b)

        return self._sol.compute_vertex_values(
            self._mesh).reshape(self._nGrid, self._nGrid)


class SPDEEngine2DDeep(SPDEEngine2D):

    def __init__(self, variance, correlation, nu, nVertPerDim, alpha,
                 useDirBC=[False, False], cacheFactorisation=True, useDirectSolver=True):

        # Do not call super().__init__ immediately
        df.set_log_level(df.LogLevel.ERROR)

        self._sd = np.sqrt(variance)
        self._nGrid = nVertPerDim
        self._cacheFactor = cacheFactorisation

        beta = 0.5 * nu + 0.25 * SPDEEngine2D.DIM
        self._betaInt = int(np.rint(beta))

        if np.abs(beta - self._betaInt) > 1e-8:
            raise NotImplementedError(
                f"Only integer values of beta are supported. Received: {beta}")
        if nu < 0.5:
            raise ValueError(f"Received invalid smoothness parameter: {nu}")

        self._kappa = np.sqrt(2. * nu)
        delta = 0.5 * (alpha - 1.)

        lowerLeft = df.Point(-delta, -delta)
        upperRight = df.Point(1. + delta, 1. + delta)

        self._mesh = df.RectangleMesh(
            lowerLeft,
            upperRight,
            nVertPerDim - 1,
            nVertPerDim - 1)
        self._varScaling = np.sqrt(
            4. * np.pi * gamma(1. + nu) / gamma(nu) *
            np.power(self._kappa, 2. * nu)
        )

        self._V = df.FunctionSpace(self._mesh, "Lagrange", 1)
        self._coeff = correlation

        self._K = df.Constant(((1., 0.), (0., 1.)))
        self._a = self._assemble_bilinear_form()

        dirBCVal = df.Constant(0.)
        self._bc = df.DirichletBC(
            self._V,
            dirBCVal,
            lambda x: on_dirichlet_boundary_2D(
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

        m = df.TrialFunction(self._V) * df.TestFunction(self._V) * df.dx
        M = df.assemble(m)
        self._H = white_noise_factor(M, self._V)

        self._sol = df.Function(self._V)
        self._f = df.Function(self._V)
        v = df.TestFunction(self._V)
        self._l = self._f * v * df.dx
        self._b = df.Vector()
        df.assemble(self._l, tensor=self._b)

    @property
    def coefficient(self):
        return self._coeff

    @coefficient.setter
    def coefficient(self, coeff):
        self._coeff = coeff
        self._a = self._assemble_bilinear_form()

        if self._cacheFactor:
            self._A = df.assemble(self._a)
            self._A.ident_zeros()
            self._bc.apply(self._A)
            self._solver.set_operator(self._A)

    @property
    def anisotropy(self):
        return self._K

    @anisotropy.setter
    def anisotropy(self, K):

        self._K = df.Constant(((K[0, 0], K[0, 1]), (K[1, 0], K[1, 1]))) 
        self._a = self._assemble_bilinear_form()

        if self._cacheFactor:
            self._A = df.assemble(self._a)
            self._A.ident_zeros()
            self._bc.apply(self._A)
            self._solver.set_operator(self._A)


    def _interpolate_coefficient(self):

        dofCoords = self._V.tabulate_dof_coordinates()

        # FIXME: This only works for alpha == 1 !
        dofValues = np.empty(self._V.dim())
        N = self._nGrid - 1
        for i, (x, y) in enumerate(dofCoords):

            # translate coordinate to array index
            ix = max(int(x * N), 0)
            iy = N - max(int(y * N), 0)

            dofValues[i] = self._coeff.T[ix, iy]

        R = df.Function(self._V)
        R.vector().set_local(dofValues)
        R.vector().apply("insert")

        return R

    def _assemble_bilinear_form(self):

        u = df.TrialFunction(self._V)
        v = df.TestFunction(self._V)

        R = self._interpolate_coefficient()

        return df.inner(self._K * df.grad(u), df.grad(v)) * df.dx + R * u * v * df.dx
