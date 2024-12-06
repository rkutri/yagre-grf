import matplotlib.ticker as ticker
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps
import fenics as fncs
import mshr as mg
import covariance_functions as covs

from numpy.random import standard_normal
from sksparse.cholmod import cholesky
from scipy.special import gamma
from scipy.sparse.linalg import spsolve

import utility

nSamp = 10
nMesh = 50

cb = True


def on_dirichlet_outer(x, useDir, tol, outerRadius=1.):

    if (useDir):
        if (np.sqrt(np.dot(x, x)) > outerRadius - tol):
            return True
    else:
        return False


def on_dirichlet_inner(x, useDir, tol, innerRadius):

    if (useDir):
        if (np.sqrt(np.dot(x, x)) < innerRadius + tol):
            return True
    else:
        return False


def on_dirichlet_disc(x, useDir, tol, radius=1.):

    if (useDir):
        return on_dirichlet_outer(x, useDir, tol, radius)


def on_dirichlet_annulus(x, useDir, tol, innerRadius, outerRadius):

    return (on_dirichlet_outer(x, useDir, tol, outerRadius)
            or on_dirichlet_inner(x, useDir, tol, innerRadius))


def on_dirichlet_ship(x, useDir, tol):

    if (useDir):

        # upper and lower edges
        if (x[1] < tol or x[1] > 1. - tol):
            return True

        # left edges
        if (x[0] < 0.25 + tol):
            return True

        q1 = np.array([x[0], x[1] - 0.5])

        # left arc (inwards)
        if (np.sqrt(np.dot(q1, q1)) < 0.5 + tol):
            return True

        q2 = np.array([x[0] - 2., x[1] - 0.5])

        # right arc (outwards)
        if (x[0] > 2.):
            if (np.sqrt(np.dot(q2, q2)) > 0.5 - tol):
                return True

        return False

    else:
        return False


def white_noise_factor(M, V):

    H = np.zeros(V.dim())
    H[0] = np.sum(M.getrow(0)[1])
    H[V.dim() - 1] = np.sum(M.getrow(V.dim() - 1)[1])

    for i in range(1, V.dim() - 1):
        H[i] = np.sum(M.getrow(i)[1])

    return 1. / np.sqrt(H)


fncs.set_log_level(fncs.LogLevel.ERROR)

ell = 0.2
nu = 1
useCholesky = False

# Matern parameters
kappa = np.sqrt(2. * nu) / ell
beta = 0.5 * (nu + 1.)

Cnu = 4. * np.pi * gamma(nu + 1.) / gamma(nu)
varScaling = np.sqrt(Cnu) * np.power(kappa, nu)


print("setting up fenics solver for 2d problem")

# mesh generation
ANNULUS = 0
DISC = 1
SHIP = 2

# meshType = ANNULUS
# meshType = DISC
meshType = SHIP
radius0 = 1.
radius1 = 0.5


if (meshType == ANNULUS):
    mesh = mg.generate_mesh(mg.Circle(fncs.Point(0, 0), radius0)
                            - mg.Circle(fncs.Point(0, 0), radius1), nMesh)
elif (meshType == DISC):
    mesh = fncs.UnitDiscMesh.create(fncs.MPI.comm_world, nMesh, 1, 2)
elif (meshType == SHIP):
    shape = mg.Rectangle(fncs.Point(0, 0), fncs.Point(
        2, 1)) - mg.Circle(fncs.Point(0, 0.5), 0.5)
    shape = shape + mg.Circle(fncs.Point(2, 0.5), 0.5)
    shape = shape - mg.Rectangle(fncs.Point(0, 0), fncs.Point(0.25, 1))
    mesh = mg.generate_mesh(shape, nMesh)

# Finite-Element space
V = fncs.FunctionSpace(mesh, "Lagrange", 1)
print("  - degrees of freedom: " + str(V.dim()))
print(f"number of cells: {mesh.num_cells()}")

# variational formulation
u = fncs.TrialFunction(V)
v = fncs.TestFunction(V)
R = fncs.Constant(kappa * kappa)

a = fncs.inner(fncs.grad(u), fncs.grad(v)) * fncs.dx + R * u * v * fncs.dx
m = u * v * fncs.dx

# boundary conditions
dirBCVal = fncs.Constant(0.)

TOL = 1e-9
meshTol = 1. / (nMesh + TOL)

if (meshType == ANNULUS):

    bc_dir = fncs.DirichletBC(
        V, dirBCVal, lambda x: on_dirichlet_annulus(
            x, True, meshTol, radius1, radius0))
    bc_neu = fncs.DirichletBC(
        V, dirBCVal, lambda x: on_dirichlet_annulus(
            x, False, meshTol, radius1, radius0))

elif (meshType == DISC):

    bc_dir = fncs.DirichletBC(
        V, dirBCVal, lambda x: on_dirichlet_disc(
            x, True, meshTol, radius0))
    bc_neu = fncs.DirichletBC(
        V, dirBCVal, lambda x: on_dirichlet_disc(
            x, False, meshTol, radius0))

else:
    bc_dir = fncs.DirichletBC(
        V, dirBCVal, lambda x: on_dirichlet_ship(x, True, meshTol))
    bc_neu = fncs.DirichletBC(
        V, dirBCVal, lambda x: on_dirichlet_ship(x, False, meshTol))


print("assemble matrices")

# assembly
A_dir = fncs.assemble(a)
A_neu = A_dir.copy()

M_dir = fncs.assemble(m)
M_neu = M_dir.copy()

bc_dir.apply(A_dir)
bc_neu.apply(A_neu)

bc_dir.apply(M_dir)
bc_neu.apply(M_neu)

print("set up solver")

# options are: “umfpack” “mumps” “superlu” “superlu_dist” “petsc”
solver_dir = fncs.LUSolver(A_dir, 'umfpack')
solver_neu = fncs.LUSolver(A_neu, 'umfpack')
solver_dir.parameters['symmetric'] = True
solver_neu.parameters['symmetric'] = True

print("factorise white noise")

if (useCholesky):

    raise "current version does not support cholesky white noise factorisation"

    print("  - using incomplete Cholesky decomposition")

    # convert to scipy sparse matrix
    M = fncs.as_backend_type(M).mat()
    M = sps.csr_matrix(M.getValuesCSR()[::-1], shape=M.size)

    # use csc as this is the native format for cholesky
    M = M.tocsc()

    # factorise mass matrix
    H = cholesky(M)

else:  # use mass lumping

    print("  - using mass lumping")

    H_dir = white_noise_factor(M_dir, V)
    H_neu = white_noise_factor(M_neu, V)

nBC = 2

H = [H_dir, H_neu]
bc = [bc_dir, bc_neu]
solver = [solver_dir, solver_neu]


# determine how often to solve in SPDE
nuInt = np.rint(nu)
assert np.abs(nu - nuInt) < TOL
assert nuInt % 2 == 1

nSolve = int(np.rint(0.5 * (nu + 1.)) + TOL)
print("performing " + str(nSolve) + " solve(s) per realisation")

genericSol_dir = fncs.Function(V)
genericSol_neu = fncs.Function(V)
genericSol_avg = fncs.Function(V)

genericSol = [genericSol_dir, genericSol_neu, genericSol_avg]

nodalValues_dir = []
nodalValues_neu = []
nodalValues_avg = []

nodalValues = [nodalValues_dir, nodalValues_neu, nodalValues_avg]

for n in range(nSamp):

    if (n % 5000 == 0):
        if (n == 0):
            print("start sampling")
        else:
            print(str(n) + " realisations computed")

    sol = []

    f = fncs.Function(V)
    u = fncs.Function(V)

    for i in range(nBC):

        fDofs = varScaling * H[i] * standard_normal(V.dim())

        f.vector()[:] = fDofs

        l = f * v * fncs.dx

        b = fncs.assemble(l)

        bc[i].apply(b)

        solver[i].solve(u.vector(), b)

        for k in range(1, nSolve):

            l = u * v * fncs.dx
            b = fncs.assemble(l)
            bc[i].apply(b)

            solver[i].solve(u.vector(), b)

        if (n == 0):
            genericSol[i].vector()[:] = u.vector()[:]

        nodalValues[i] += [u.vector()[:]]

    if (n == 0):
        genericSol[2].vector()[:] = (genericSol[0].vector()[:] +
                                     genericSol[1].vector()[:]) / np.sqrt(2.)

    nodalValues[2] += [(nodalValues[0][n] + nodalValues[1][n]) / np.sqrt(2.)]


print("computing marginal variance")


margVarDofs_dir = np.var(np.array(nodalValues[0]), axis=0)
margVarDofs_neu = np.var(np.array(nodalValues[1]), axis=0)
margVarDofs_avg = np.var(np.array(nodalValues[2]), axis=0)

marginalVariance_dir = fncs.Function(V)
marginalVariance_neu = fncs.Function(V)
marginalVariance_avg = fncs.Function(V)

if cb:
    margVarDofs_avg[0] = 0.
    margVarDofs_avg[1] = 6.

marginalVariance_dir.vector()[:] = margVarDofs_dir
marginalVariance_neu.vector()[:] = margVarDofs_neu
marginalVariance_avg.vector()[:] = margVarDofs_avg

print("plotting")


# Adjust line width and font size
lw = 3
fontsize = 12

# Colormap and color range settings
cmap = "turbo"
vmin, vmax = 0.0, 4.0  # Data values mapped to colors
colorbar_range = [0, 1, 2, 3, 4, 5, 6]

# Create figure with specific aspect ratio
fig_width = 8  # Width of the figure
fig_height = 6  # Height of the figure
plt.figure(figsize=(fig_width, fig_height))

# Plot the image with colormap and limits
mvPlt = fncs.plot(marginalVariance_avg)
mvPlt.set_cmap(cmap)
mvPlt.set_clim(vmin, vmax)

plt.gca().set_aspect("equal")
plt.gca().tick_params(which='both', size=0, labelsize=0)

if cb:
    cbar = plt.colorbar(mvPlt, ticks=colorbar_range, extend='both')
    cbar.set_ticks(colorbar_range)
    cbar.ax.tick_params(labelsize=fontsize)

# Remove the box around the plot
plt.axis('off')

plt.tight_layout()
plt.savefig(
    'margVar2d_ship_colorbar.png',
    bbox_inches='tight',
    format='png',
    dpi=800)
plt.show()
