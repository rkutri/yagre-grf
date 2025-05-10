import matplotlib.ticker as ticker
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps
import fenics as fncs
import mshr as mg
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase

from numpy.random import standard_normal
from sksparse.cholmod import cholesky
from scipy.special import gamma
from scipy.sparse.linalg import spsolve

from yagregrf.utility.accumulation import MarginalVarianceAccumulator


nSamp = 50000
nMesh = 50

plotCB = False


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


def on_dirichlet_ship_full(x, useDir, tol):

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

def on_dirichlet_ship_lr(x, tol):

    # upper and lower edges
    if (x[1] < tol or x[1] > 1. - tol):
        return False

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

def on_dirichlet_ship_tb(x, tol):

    # upper and lower edges
    if (x[1] < tol or x[1] > 1. - tol):
        return True
    
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

TOL = 1e-2
meshTol = 1. / (nMesh + TOL)

if meshType == ANNULUS:

    bc_dir = fncs.DirichletBC(
        V, dirBCVal, lambda x: on_dirichlet_annulus(
            x, True, meshTol, radius1, radius0))
    bc_neu = fncs.DirichletBC(
        V, dirBCVal, lambda x: on_dirichlet_annulus(
            x, False, meshTol, radius1, radius0))

elif meshType == DISC:

    bc_dir = fncs.DirichletBC(
        V, dirBCVal, lambda x: on_dirichlet_disc(
            x, True, meshTol, radius0))
    bc_neu = fncs.DirichletBC(
        V, dirBCVal, lambda x: on_dirichlet_disc(
            x, False, meshTol, radius0))

elif meshType == SHIP:

    bc_dir_full = fncs.DirichletBC(
        V, dirBCVal, lambda x: on_dirichlet_ship_full(x, True, meshTol))
    bc_neu_full = fncs.DirichletBC(
        V, dirBCVal, lambda x: on_dirichlet_ship_full(x, False, meshTol))
    bc_dir_lr = fncs.DirichletBC(
        V, dirBCVal, lambda x: on_dirichlet_ship_lr(x, meshTol))
    bc_dir_tb = fncs.DirichletBC(
        V, dirBCVal, lambda x: on_dirichlet_ship_tb(x, meshTol))


print("assemble matrices")

# assembly
A_neu_full = fncs.assemble(a)
A_dir_full = A_neu_full.copy()
A_dir_lr = A_neu_full.copy()
A_dir_tb = A_neu_full.copy()

M_neu_full = fncs.assemble(m)
M_dir_full = M_neu_full.copy()
M_dir_lr = M_neu_full.copy()
M_dir_tb = M_neu_full.copy()

bc_dir_full.apply(A_dir_full)
bc_neu_full.apply(A_neu_full)
bc_dir_lr.apply(A_dir_lr)
bc_dir_tb.apply(A_dir_tb)

bc_dir_full.apply(M_dir_full)
bc_neu_full.apply(M_neu_full)
bc_dir_lr.apply(M_dir_lr)
bc_dir_tb.apply(M_dir_tb)

print("set up solver")

# options are: “umfpack” “mumps” “superlu” “superlu_dist” “petsc”
solver_dir_full = fncs.LUSolver(A_dir_full, 'umfpack')
solver_neu_full = fncs.LUSolver(A_neu_full, 'umfpack')
solver_dir_lr = fncs.LUSolver(A_dir_lr, 'umfpack')
solver_dir_tb = fncs.LUSolver(A_dir_tb, 'umfpack')

solver_dir_full.parameters['symmetric'] = True
solver_neu_full.parameters['symmetric'] = True
solver_dir_lr.parameters['symmetric'] = True
solver_dir_tb.parameters['symmetric'] = True

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

    H_dir_full = white_noise_factor(M_dir_full, V)
    H_neu_full = white_noise_factor(M_neu_full, V)
    H_dir_lr = white_noise_factor(M_dir_lr, V)
    H_dir_tb = white_noise_factor(M_dir_tb, V)

nBC = 4

H = [H_dir_full, H_neu_full, H_dir_lr, H_dir_tb]
bc = [bc_dir_full, bc_neu_full, bc_dir_lr, bc_dir_tb]
solver = [solver_dir_full, solver_neu_full, solver_dir_lr, solver_dir_tb]


# determine how often to solve in SPDE
nuInt = np.rint(nu)
assert np.abs(nu - nuInt) < TOL
assert nuInt % 2 == 1

nSolve = int(np.rint(0.5 * (nu + 1.)) + TOL)
print("performing " + str(nSolve) + " solve(s) per realisation")

genericSol_dir_full = fncs.Function(V)
genericSol_neu_full = fncs.Function(V)
genericSol_dir_lr = fncs.Function(V)
genericSol_dir_tb = fncs.Function(V)

genericSol = [genericSol_dir_full, genericSol_neu_full, genericSol_dir_lr, genericSol_dir_tb]

nodalValues_dir_full = []
nodalValues_neu_full = []
nodalValues_dir_lr = []
nodalValues_dir_tb = []

nodalValues = [nodalValues_dir_full, nodalValues_neu_full, nodalValues_dir_lr, nodalValues_dir_tb]

avgMV = MarginalVarianceAccumulator(V.dim())

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

        nodalValues[i] = u.vector()[:]

    avgSol = nodalValues[0]
    for ii in range(1, nBC):
        avgSol += nodalValues[ii]

    avgMV.update(0.5 * avgSol)

mvFcn = fncs.Function(V)
mvFcn.vector()[:] = avgMV.marginalVariance


print("computing marginal variance")


print("plotting")


# Plotting settings
fontsize = 12
cmap_name = "turbo"
vmin, vmax = 0.0, 4.0
ticks = [0, 1, 2, 3, 4]

# Create figure
fig = plt.figure(figsize=(8, 6))
ax = plt.gca()

# Plot the function (FEniCS uses current axis)
mvPlt = fncs.plot(mvFcn)
mvPlt.set_cmap(plt.colormaps[cmap_name])
mvPlt.set_clim(vmin, vmax)

# Clean plot appearance
ax.set_aspect("equal")
ax.axis("off")

# Colorbar with fixed ticks and no label
if plotCB:
    cbar = plt.colorbar(mvPlt, ax=ax, orientation='vertical', extend='both')
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([str(t) for t in ticks])  # Force exact tick labels
    cbar.ax.tick_params(labelsize=fontsize)

# Final layout and save
plt.tight_layout()
plt.savefig("margVar2d_ship.png", bbox_inches='tight', dpi=800)
plt.show()

