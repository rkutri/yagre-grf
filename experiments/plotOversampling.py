import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogFormatterMathtext
from experiments.filename import create_data_string

# =============================================================================
# Parameters for Filename Generation
# =============================================================================

DIM = 2
var = 0.1
ell = 0.05
nu = 1.0
nSampBatch = int(5e4)
nBatch = 6


baseDir = 'data'

errorType = "froError"

fileStr = create_data_string(DIM, var, ell, nu, nSampBatch, "os_ACCUMULATED") \
    + f"_{nBatch}batches_" + errorType + ".csv"
fileName = os.path.join(baseDir, fileStr)

# =============================================================================
# Plot Appearance Settings (variables)
# =============================================================================

lineWidth = 0.8
markerSize = 6
fontSizeLabel = 12
fontSizeTicks = 10
tickLabelSize = 8
fontSizeLegend = 8
legendMarkerSize = 4

figWidth = 5.0  # inches
figHeight = 5.0  # inches

# =============================================================================
# Read Data from CSV
# =============================================================================

methods = []
meshWidths = []
errors = {}
errorBars = {}

isErrorBar = False

with open(fileName, mode='r') as file:

    reader = csv.reader(file)
    rows = list(reader)

    # first row contains mesh widths
    meshWidths = [float(x) for x in rows[0][1:]]

    for i in range(1, len(rows)):

        label = rows[i][0]
        print(label)

        if label.count('_') > 1:

            isErrorBar = True
            method = label.rsplit('_', 1)[0]

        else:
            isErrorBar = False
            method = label

        if method not in methods:
            methods.append(method)

        if isErrorBar:
            if method not in errorBars:
                errorBars[method] = [float(x) for x in rows[i][1:]]
        else:
            if method not in errors:
                errors[method] = [float(x) for x in rows[i][1:]]


# =============================================================================
# Define Plot Styles
# =============================================================================

dnaMethods = ["DNA_fourier", "DNA_spde"]
spdeMethods = [m for m in methods if "SPDE_alpha" in m]

dnaColors = {'DNA_fourier': 'tab:green', 'DNA_spde': 'tab:red'}
spdeColor = 'tab:blue'

linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (3, 2, 1, 2))]

markers = ['o', 's', 'D', '^', 'p', 'v']

# =============================================================================
# Create Figure and Axes
# =============================================================================
fig, ax = plt.subplots(figsize=(figWidth, figHeight))

# =============================================================================
# Plot Data
# =============================================================================

# Plot SPDE methods
for i, method in enumerate(spdeMethods):
    linestyle = linestyles[i]
    marker = markers[i]
    ax.errorbar(
        meshWidths,
        errors[method],
        yerr=errorBars[method],
        fmt=marker,
        linestyle=linestyle,
        label=method.replace("_", " "),
        markersize=markerSize,
        markeredgewidth=1.5,
        elinewidth=lineWidth,
        capsize=5,
        color=spdeColor,
        alpha=0.9
    )

# Plot DNA methods
for method in dnaMethods:
    if method in errors:
        ax.errorbar(
            meshWidths,
            errors[method],
            yerr=errorBars[method],
            fmt=markers[0],
            linestyle=linestyles[0],
            label=method.replace("_", " "),
            markersize=markerSize,
            markeredgewidth=1.5,
            elinewidth=lineWidth,
            capsize=5,
            color=dnaColors[method],
            alpha=0.9
        )

# =============================================================================
# Draw Rate Indication Triangle
# =============================================================================
rate = 0.5
base = 0.025
height = rate * base
xTriangle = [0.05, 0.05 + base]
yTriangle = [0.02, 0.02 + height]

ax.plot([xTriangle[0], xTriangle[1]], [yTriangle[0],
                                       yTriangle[0]], color='black', lw=lineWidth)
ax.plot([xTriangle[1], xTriangle[1]], [yTriangle[0],
                                       yTriangle[1]], color='black', lw=lineWidth)
ax.plot([xTriangle[0], xTriangle[1]], [yTriangle[0],
                                       yTriangle[1]], color='black', lw=lineWidth)

ax.text(
    xTriangle[0] +
    0.5 *
    base,
    yTriangle[0] -
    0.006,
    '1',
    fontsize=fontSizeTicks,
    ha='center')
ax.text(
    xTriangle[1] +
    0.003,
    yTriangle[1] -
    0.6 *
    height,
    '1/2',
    fontsize=fontSizeTicks,
    va='center')

# =============================================================================
# Customize Axes
# =============================================================================
ax.set_xlabel('Mesh width $h$', fontsize=fontSizeLabel)
ax.set_ylabel('Error estimate', fontsize=fontSizeLabel)
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Set tick label sizes for both major and minor ticks on both axes
ax.tick_params(axis='both', which='both', labelsize=tickLabelSize)


# =============================================================================
# Legend Setup
# =============================================================================
# Compute marker scale for legend relative to markerSize using
# legendMarkerSize variable
markerScaleValue = legendMarkerSize / markerSize

handles, labels = ax.get_legend_handles_labels()
legend = ax.legend(
    handles,
    labels,
    fontsize=fontSizeLegend,
    loc='upper left',
    bbox_to_anchor=(1.05, 1),
    frameon=True,
    framealpha=0.9,
    markerscale=markerScaleValue
)
for text in legend.get_texts():
    text.set_alpha(1)
for line in legend.get_lines():
    line.set_alpha(1)

# =============================================================================
# Final Layout Adjustments and Save Figure
# =============================================================================
plt.tight_layout(rect=[0, 0, 0.85, 1])  # Reserve space for the legend
ax.set_aspect('auto', adjustable='box')

plt.savefig(
    './spde_oversampling.pdf',
    format='pdf',
    dpi=300,
    bbox_inches='tight')
plt.show()
