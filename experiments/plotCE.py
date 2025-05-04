import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties

from experiments.readCEData import read_averaged_data

# l.rcParams['text.usetex'] = True

methods = ["dna", "ce", "aCE"]
covs = [r'Gaussian', r'Matérn, $\nu = 5$', r'exponential']

colors = {
    "dna": "tab:green",
    "ce": "tab:purple",
    "aCE": "tab:orange"
}
# LEGEND
methodLabels = {
    "dna": "DNA",
    "ce": "Circulant Embedding",
    "aCE": "approximate CE"}
linestyles = ["-", "--", ":", "-."]
markers = {"dna": 'o', "ce": '^', "aCE": 's'}
zorder = {"dna": 7, "ce": 9, "aCE": 8}

markerSize = 5
markerSizes = {"dna": markerSize, "ce": markerSize - 1, "aCE": markerSize + 1}


baseDir = os.path.join("experiments", "publicationData", "circulantEmbedding")
errorType = "maxError"
nBatch = 5

averagedData = read_averaged_data(baseDir, nBatch)

fig, axes = plt.subplots(1, 2, figsize=(6, 2.2))

lineWidth = 1.5
fontSizeXLabel = 9
fontSizeYLabel = 9
fontSizeTicks = 8
tickLabelSize = 6
fontSizeLegend = 8
legendMarkerSize = 6
lineAlpha = 0.9
circleSize = 225

# === 1. Cost vs Mesh Width ===
ax = axes[0]
meshWidths = averagedData["cost"]["meshWidths"]
problemSize = [1. / h**2 for h in meshWidths]

for i, method in enumerate(averagedData["cost"]["yData"]):
    for j, modelCov in enumerate(averagedData["cost"]["yData"][method]):

        if method == "ce":

            cost = averagedData["cost"]["yData"][method][modelCov]
            costPlt = averagedData["cost"]["yData"]["aCE"][modelCov]

            # add circle, if embedding was not possible
            if np.any(np.isinf(cost)):

                lastIdx = np.where(np.isfinite(cost))[0][-1]
                ax.scatter([problemSize[lastIdx]], costPlt[lastIdx], color=colors[method],
                           s=circleSize, facecolors='none', linewidths=1.5 * lineWidth, alpha=0.9,
                           linestyle=linestyles[j])
        else:
            if j > 0:
                continue

            cost = averagedData["cost"]["yData"][method][modelCov]

            ax.plot(problemSize, cost,
                    marker=markers[method],
                    markersize=markerSizes[method],
                    markeredgewidth=1.5,
                    color=colors[method],
                    linestyle=linestyles[j],
                    linewidth=lineWidth,
                    alpha=lineAlpha,
                    zorder=zorder[method],
                    label=methodLabels[method])

ax.set_xlabel("problem size (dofs)", fontsize=fontSizeXLabel, labelpad=1)
ax.set_ylabel("runtime (s)", fontsize=fontSizeYLabel, labelpad=1)
ax.set_xscale("log")
ax.set_yscale("log")
ax.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.6, zorder=1)
ax.tick_params(axis='both', which='both', labelsize=tickLabelSize)

# === 2. Memory vs Mesh Width ===
ax = axes[1]
meshWidths = averagedData["memory"]["meshWidths"]
problemSize = [1. / h**2 for h in meshWidths]

for i, method in enumerate(averagedData["memory"]["yData"]):
    for j, modelCov in enumerate(averagedData["memory"]["yData"][method]):

        if method == "ce":

            memory = averagedData["memory"]["yData"][method][modelCov]
            memPlt = averagedData["memory"]["yData"]["aCE"][modelCov]

            # add circle, if embedding was not possible
            if np.any(np.isinf(memory)):

                lastIdx = np.where(np.isfinite(memory))[0][-1]
                ax.scatter(problemSize[lastIdx], memPlt[lastIdx], color=colors[method],
                           s=circleSize, facecolors='none', linewidths=1.5 * lineWidth, alpha=0.9,
                           linestyle=linestyles[j])

        else:
            if j > 0:
                continue

            memory = averagedData["memory"]["yData"][method][modelCov]

            ax.plot(problemSize, memory,
                    marker=markers[method],
                    markersize=markerSizes[method],
                    markeredgewidth=1.5,
                    color=colors[method],
                    linestyle=linestyles[j],
                    linewidth=lineWidth,
                    alpha=lineAlpha,
                    zorder=zorder[method],
                    label=methodLabels[method])


ax.set_xlabel("problem size (dofs)", fontsize=fontSizeXLabel, labelpad=1)
ax.set_ylabel("peak memory (MB)", fontsize=fontSizeYLabel, labelpad=1)
ax.set_xscale("log")
ax.set_yscale("log")
ax.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.6, zorder=1)
ax.tick_params(axis='both', which='both', labelsize=tickLabelSize)

fig.subplots_adjust(left=0.08, right=0.75, top=0.97, bottom=0.16, wspace=0.3)


handles, _ = axes[0].get_legend_handles_labels()
# methodHandles = [
#    Line2D(
#        [0],
#        [0],
#        color=colors[method],
#        linestyle='None',
#        linewidth=lineWidth,
#        marker=markers[method],
#        markersize=markerSize + 2,
#        label=methodLabels[method])
#    for method in methods
# ]

covHandles = [
    Line2D(
        [0],
        [0],
        color=colors["ce"],
        linestyle=linestyles[i],
        label=covs[i])
    for i in range(len(covs) - 1)
]

methodLegend = plt.legend(
    handles=handles,
    title=r'Method',
    title_fontproperties=FontProperties(weight='bold'),
    fontsize=fontSizeLegend,
    bbox_to_anchor=(1.03, 1.04),
    frameon=True,
    framealpha=0.9,
    markerscale=legendMarkerSize / markerSize
)

plt.gca().add_artist(methodLegend)

plt.legend(
    handles=covHandles,
    title=r'Last Embedding',
    title_fontproperties=FontProperties(weight='bold'),
    fontsize=fontSizeLegend,
    bbox_to_anchor=(1.03, 0.52),
    frameon=True,
    framealpha=0.9,
    markerscale=legendMarkerSize / markerSize
)

# Save to PDF
plt.savefig('./ce_comparison.pdf', dpi=300, format='pdf')

plt.show()
