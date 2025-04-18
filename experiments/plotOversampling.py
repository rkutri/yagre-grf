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
ell = 0.2
nu = 1.0
nSampBatch = int(5e4)
nBatch = 5


# baseDir = 'data'
baseDir = os.path.join('experiments', 'publicationData')


errorTypes = ["maxError", "froError"]
errorType = errorTypes[0]

fileStr = create_data_string(DIM, var, ell, nu, nSampBatch, "os_ACCUMULATED") \
    + f"_{nBatch}batches_" + errorType + ".csv"
fileName = os.path.join(baseDir, fileStr)

# =============================================================================
# Plot Appearance Settings (variables)
# =============================================================================

lineWidth = 0.8
markerSize = 6
fontSizeLabel = 10
fontSizeTicks = 10
tickLabelSize = 6
fontSizeLegend = 6
legendMarkerSize = 6

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
spdeColors = ['tab:orange', 'tab:blue', 'tab:purple']

linestyles = ['-', '--', '-.', ':']

markers = ['o', 's', 'D', '^', 'p']

# =============================================================================
# Create Figure and Axes
# =============================================================================
fig, ax = plt.subplots(figsize=(figWidth, figHeight))

# =============================================================================
# Plot Data
# =============================================================================

# Plot SPDE methods
for i, method in enumerate(spdeMethods):
    linestyle = linestyles[0]
    marker = markers[0]
    color = spdeColors[i]
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
        color=color,
        alpha=0.9
    )

# Plot DNA methods
for i, method in enumerate(dnaMethods):
    linestyle = linestyles[0]
    marker = markers[0]
    color = dnaColors[method]
    if method in errors:
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
            color=color,
            alpha=0.9
        )
    else:
        raise RuntimeError(f"No error data for method: {method}")

# =============================================================================
# Draw Rate Indication Triangle
# =============================================================================
baseScale = 1.5
x0 = 0.03
x1 = x0 * baseScale

rate = 1.
y0 = 0.003
y1 = y0 * np.power(baseScale, rate)

ax.plot([x0, x1], [y0, y1], color='black', lw=lineWidth)
ax.plot([x0, x1], [y0, y0], color='black', lw=lineWidth)
ax.plot([x1, x1], [y0, y1], color='black', lw=lineWidth)

ax.text(x1 + 0.005, 1.1 * y0, f"{rate}", color='k',
        horizontalalignment='center', verticalalignment='bottom', fontsize=0.5 * fontSizeTicks)

# =============================================================================
# Customize Axes
# =============================================================================

ax.set_title(f"d = {DIM}, ell = {ell}, nu = {nu}")

errorLabel = "maximal error" if errorType == "maxError" else "relative Frobenius Error"

ax.set_xlabel('Mesh width $h$', fontsize=fontSizeLabel)
ax.set_ylabel(errorLabel, fontsize=fontSizeLabel)
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

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
ax.set_aspect('equal', adjustable='box')

plt.savefig(
    './spde_oversampling.pdf',
    format='pdf',
    dpi=300,
    bbox_inches='tight')
plt.show()
