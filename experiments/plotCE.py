import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties

from experiments.readCEData import read_averaged_data

# l.rcParams['text.usetex'] = True

methods = ["dna", "ce", "aCE"]
covs = [r'gaussian', r'Matérn', r'exponential']

colors = {
    "dna": "tab:green",
    "ce": "tab:purple",
    "aCE": "tab:orange"
}
linestyles = ["-", "--", ":", "-."]
markers = {"dna": 'o', "ce": '^', "aCE": 's'}
zorder = {"dna": 7, "ce": 9, "aCE": 8}

markerSize = 5
markerSizes = {"dna": markerSize, "ce": markerSize - 1, "aCE": markerSize + 1}


baseDir = os.path.join("experiments", "publicationData", "circulantEmbedding")
errorType = "maxError"
nBatch = 15

averagedData = read_averaged_data(baseDir, nBatch)

fig, axes = plt.subplots(1, 3, figsize=(9, 2.2))

lineWidth = 1.5
fontSizeXLabel = 11
fontSizeYLabel = 11
fontSizeTicks = 9
tickLabelSize = 6
fontSizeLegend = 10
legendMarkerSize = 4
lineAlpha = 0.9

# === 1. Cost vs Mesh Width ===
ax = axes[0]
meshWidths = averagedData["cost"]["meshWidths"]

for i, method in enumerate(averagedData["cost"]["yData"]):
    for j, modelCov in enumerate(averagedData["cost"]["yData"][method]):

        cost = averagedData["cost"]["yData"][method][modelCov]

        ax.plot(meshWidths, cost,
                marker=markers[method],
                markersize=markerSizes[method],
                markeredgewidth=1.5,
                color=colors[method],
                linestyle=linestyles[j],
                linewidth=lineWidth,
                alpha=lineAlpha,
                zorder=zorder[method])

        # add circle, if embedding was not possible
        if np.any(np.isinf(cost)):

            lastIdx = np.where(np.isfinite(cost))[0][-1]
            ax.scatter([meshWidths[lastIdx]], [cost[lastIdx]], color=colors[method],
                       s=125, facecolors='none', linewidths=lineWidth, alpha=0.9)

ax.set_xlabel("mesh width h", fontsize=fontSizeXLabel, labelpad=1)
ax.set_ylabel("runtime (s)", fontsize=fontSizeYLabel, labelpad=1)
ax.set_xscale("log")
ax.set_yscale("log")
ax.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.6, zorder=1)
ax.tick_params(axis='both', which='both', labelsize=tickLabelSize)

# === 2. Memory vs Mesh Width ===
ax = axes[1]
meshWidths = averagedData["memory"]["meshWidths"]

for i, method in enumerate(averagedData["memory"]["yData"]):
    for j, modelCov in enumerate(averagedData["memory"]["yData"][method]):

        memory = averagedData["memory"]["yData"][method][modelCov]

        ax.plot(meshWidths, memory,
                marker=markers[method],
                markersize=markerSizes[method],
                markeredgewidth=1.5,
                color=colors[method],
                linestyle=linestyles[j],
                linewidth=lineWidth,
                alpha=lineAlpha,
                zorder=zorder[method])

        # add circle, if embedding was not possible
        if np.any(np.isinf(memory)):

            lastIdx = np.where(np.isfinite(memory))[0][-1]
            ax.scatter(meshWidths[lastIdx], memory[lastIdx], color=colors[method],
                       s=125, facecolors='none', linewidths=lineWidth, alpha=0.9)

ax.set_xlabel("mesh width h", fontsize=fontSizeXLabel, labelpad=1)
ax.set_ylabel("peak memory (MB)", fontsize=fontSizeYLabel, labelpad=1)
ax.set_xscale("log")
ax.set_yscale("log")
ax.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.6, zorder=1)
ax.tick_params(axis='both', which='both', labelsize=tickLabelSize)

# === 3. Error vs Cost (dna + aCE only) ===
ax = axes[2]

for i, method in enumerate(averagedData["error"]["yData"]):
    for j, modelCov in enumerate(averagedData["error"]["yData"][method]):

        cost = averagedData["error"]["xData"][method][modelCov]
        error = averagedData["error"]["yData"][method][modelCov]
        errorBars = averagedData["error"]["errorBars"][method][modelCov]

        ax.errorbar(cost, error,
                    yerr=errorBars,
                    elinewidth=0.75 * lineWidth,
                    capsize=3,
                    capthick=0.5 * lineWidth,
                    marker=markers[method],
                    markersize=markerSizes[method],
                    markeredgewidth=1.5,
                    color=colors[method],
                    linestyle=linestyles[j],
                    linewidth=lineWidth,
                    alpha=lineAlpha,
                    zorder=zorder[method])

ax.set_xlabel("runtime (s)", fontsize=fontSizeXLabel, labelpad=1)
ax.set_ylabel("est. covariance error", fontsize=fontSizeYLabel, labelpad=1)
ax.set_xscale("log")
ax.set_yscale("log")

ax.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.6, zorder=1)
ax.tick_params(axis='both', which='both', labelsize=tickLabelSize)
ax.tick_params(axis='both', which='minor', labelsize=tickLabelSize // 2)

fig.subplots_adjust(left=0.055, right=0.765, top=0.97, bottom=0.16, wspace=0.3)

# LEGEND
methodLabels = {
    "dna": "DNA",
    "ce": "Circulant Embedding",
    "aCE": "approximate CE"}
methodHandles = [
    Line2D(
        [0],
        [0],
        color=colors[method],
        linestyle='None',
        linewidth=lineWidth,
        marker=markers[method],
        markersize=markerSize + 2,
        label=methodLabels[method])
    for method in methods
]

covHandles = [
    Line2D([0], [0], color='black', linestyle=linestyles[i], label=covs[i])
    for i in range(len(covs))
]

methodLegend = plt.legend(
    handles=methodHandles,
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
    title=r'Covariance',
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
