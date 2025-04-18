import matplotlib.pyplot as plt
import numpy as np

import covariance_functions as covs
import embedding1d as ce


corrLength = np.linspace(0.05, 0.3, 10, endpoint=True)
smoothness = [1., 1.5, 2., 2.5, 3]


nGrid = 1500
grid = np.linspace(0., 1., nGrid + 2, endpoint=True)

crfDofFraction = []

nuIdx = 0

for nu in smoothness:

    crfDofFraction.append([])

    print(f"\n\n\nnu = {nu}")

    for ell in corrLength:

        print(f"\nell = {ell}")

        def grid_cov_fcn(g): return np.array(
            [covs.matern_covariance_ptw(r, ell, nu) for r in g])

        def cov_fcn_ptw(x): return covs.matern_covariance_ptw(x, ell, nu)
        def cov_ft_ptw(s): return covs.maternn_fourier_ptw(s, ell, nu, 1)

        try:
            minFactor, nIter = ce.determine_smallest_embedding_factor(
                grid_cov_fcn, nGrid, 200.)

            crfDofFraction[nuIdx].append(2. * minFactor)

        except BaseException:

            print("embedding factor can't be chosen large enough")
            crfDofFraction[nuIdx].append(0.)

    nuIdx += 1

print(crfDofFraction)


crfDofFraction = np.array(crfDofFraction)


# Assuming crfDofFraction, smoothness, and corrLength arrays have been
# defined as in your code

# Plot setup
plt.figure(figsize=(4, 2))
colors = plt.get_cmap("tab10")

for idx, nu in enumerate(smoothness):
    # Retrieve DOF fractions and correlation lengths for this smoothness
    yVals = crfDofFraction[idx]
    xVals = corrLength

    # Find indices where DOF fraction jumps to 0
    zeroJumpIdx = np.where((yVals[:-1] > 0) & (yVals[1:] == 0))[0]

    if not zeroJumpIdx:
        plt.plot(xVals, yVals,
                 color=colors(idx), label=f'ν = {nu}', linewidth=1.5, marker='o')
    else:

        jumpIdx = zeroJumpIdx[0]
        plt.plot(xVals[:jumpIdx + 1],
                 yVals[:jumpIdx + 1],
                 color=colors(idx),
                 label=f'ν = {nu}',
                 linewidth=1.5,
                 marker='o')


plt.plot(
    np.array(corrLength),
    np.zeros_like(
        np.array(corrLength)),
    color='k',
    linewidth=1)

# Plot formatting with font size adjustments
plt.xlabel(r"correlation length $\ell$", fontsize=12)
plt.ylabel(r"overhead $\tau$", fontsize=12)
plt.legend(
    title="smoothness",
    fontsize=10,
    title_fontsize=12,
    bbox_to_anchor=(
        1,
        1),
    frameon=False)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xlim([np.min(corrLength), np.max(corrLength)])
plt.ylim([0, 1.1 * np.max(crfDofFraction)])  # Slight margin on top for clarity

plt.savefig("dof_overhead.pdf", format="pdf", bbox_inches="tight", dpi=600)

plt.show()
