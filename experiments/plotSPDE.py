import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogFormatterMathtext
from experiments.filename import create_data_string
from collections import OrderedDict

# =============================================================================
# Parameters for Filename Generation
# =============================================================================

DIM = 2
var = 0.1
ell = 0.25
nu = 1.0
nSampBatch = int(5e4)
nBatch = 5

variables = [("oversampling", "os"), ("memory", "mem"), ("cost", "cost")]

errorTypes = ["maxError", "froError"]
errorType = errorTypes[0]

legendLabels = [
        r'DNA - Fourier',
        r'DNA - Finite Element',
        r'SPDE - vanilla',
        r'SPDE - heuristic'
    ]

yLabel = r'est. of maximal error'
xLabels = {
        "oversampling": r'mesh width h',
        "memory": r'peak memory (MB)',
        "cost": r'runtime (s)'
    }

# Create figure with 3 subplots
fig, axs = plt.subplots(1, 3, figsize=(9, 2.6))

for i, (variable, prefix) in enumerate(variables):

    print(f"generating plot for {variable}")

    #    baseDir = 'data'
    baseDir = os.path.join('experiments', 'publicationData')
    fileStr = create_data_string(DIM, var, ell, nu, nSampBatch,
                                 prefix + "_ACCUMULATED") \
        + f"_{nBatch}batches_" + errorType + ".csv"
    fileName = os.path.join(baseDir, fileStr)

    # =============================================================================
    # Plot Appearance Settings (variables)
    # =============================================================================

    lineWidth = 1.2
    markerSize = 6
    fontSizeLabel = 10
    fontSizeTicks = 10
    tickLabelSize = 6
    fontSizeLegend = 8
    legendMarkerSize = 4
    lineAlpha = 0.9

    # =============================================================================
    # Read Data from CSV
    # =============================================================================

    meshWidths = []
    methods = []
    variables = {}
    errors = {}
    errorBars = {}

    with open(fileName, mode='r') as file:

        reader = csv.reader(file)
        rows = list(reader)

        for ii, row in enumerate(rows):

            if variable == "oversampling":

                isErrorBar = False

                if ii == 0:
                    meshWidths = [float(x) for x in row[1:]]

                label = row[0]

                if label.count('_') > 1:
                    isErrorBar = True
                    method = label.rsplit('_', 1)[0]

                else:
                    isErrorBar = False
                    method = label

            else:

                label = row[0].rsplit('_', 1)

                method = label[0]
                variableName = label[1]

                isErrorBar = (variableName == "bars")

            print(f"\nreading data for {method}")

            if method not in methods:
                methods.append(method)

            if isErrorBar:
                if method not in errorBars:
                    errorBars[method] = [float(x) for x in row[1:]]
            else:

                if variable == "oversampling":
                    if method not in errors:
                        errors[method] = [float(x) for x in row[1:]]
                else:
                    if variableName == errorType:
                        if method not in errors:
                            errors[method] = [float(x) for x in row[1:]]

                    elif variableName == variable:
                        if method not in variables:
                            variables[method] = [float(x) for x in row[1:]]

                    else:
                        raise RuntimeError(
                            f"unknown variable name: {variableName}")

    # =============================================================================
    # Define Plot Styles
    # =============================================================================

    colors = {
        'SPDE': ['tab:purple', 'tab:blue', 'tab:orange'],
        'DNA_fourier': 'tab:green',
        'DNA_spde': 'tab:red'
    }

    linestyles = ['-', '--', '-.', ':']

    markers = ['o', 's', 'D', '^', 'p']

    # =============================================================================
    # Plot Data
    # =============================================================================

    ax = axs[i]  # Select subplot axis

    # Plot SPDE methods
    for iii, method in enumerate(methods):

        if method == "meshWidths":
            continue

        linestyle = linestyles[0]
        marker = markers[iii]

        if method == "SPDE_alpha100":
            color = colors['SPDE'][0]

        elif "SPDE" in method:
            color = colors['SPDE'][1]
        else:
            color = colors[method]

        if variable == "oversampling":
            xData = meshWidths
        else:
            xData = variables[method]

        yData = errors[method]

        ax.errorbar(
            xData,
            yData,
            yerr=errorBars[method],
            fmt=marker,
            linestyle=linestyle,
            label=method.replace("_", " "),
            markersize=markerSize,
            markeredgewidth=1.5,
            elinewidth=lineWidth,
            capsize=5,
            color=color,
            alpha=lineAlpha
        )

    # =============================================================================
    # Draw Rate Indication Triangle
    # =============================================================================

    if variable == "memory":

        baseScale = 1.5
        rate = -1.

        x0 = 0.03
        y0 = 0.004

        labelOffset = 0.01

    elif variable == "cost":

        baseScale = 1.4
        rate = -1.

        x0 = 0.005
        y0 = 0.007

        labelOffset = 0.001

    elif variable == "oversampling":

        baseScale = 2.0
        rate = -2.

        x0 = 0.02
        y0 = 0.01

        labelOffset = 0.02

    else:
        raise RuntimeError("")

    x1 = x0 * baseScale
    y1 = y0 * np.power(baseScale, rate)

    ax.plot([x0, x1], [y0, y1], color='black', lw=lineWidth)
    ax.plot([x0, x1], [y1, y1], color='black', lw=lineWidth)
    ax.plot([x0, x0], [y0, y1], color='black', lw=lineWidth)

    ax.text(x0 - labelOffset, 1.1 * y1, f"{rate}", color='k',
            horizontalalignment='center', verticalalignment='bottom', fontsize=0.5 * fontSizeTicks)

    # =============================================================================
    # Customize Axes
    # =============================================================================

    errorLabel = "maximal error" if errorType == "maxError" else "relative Frobenius error"

    if i == 0:
        ax.set_ylabel(yLabel, fontsize=fontSizeLabel)

    ax.set_xlabel(xLabels[variable], fontsize=fontSizeLabel)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)

    # Set tick label sizes for both major and minor ticks on both axes
    ax.tick_params(axis='both', which='both', labelsize=tickLabelSize)

    # =============================================================================
    # Legend Setup
    # =============================================================================

handles, _ = axs[0].get_legend_handles_labels() 


fig.legend(
    handles,
    legendLabels,
    fontsize=fontSizeLegend,
    loc='center right',
    bbox_to_anchor=(0.99, 0.8),
    frameon=True,
    framealpha=0.9,
    markerscale=legendMarkerSize / markerSize
)


# =============================================================================
# Final Layout Adjustments and Save Figure
# =============================================================================

fig.subplots_adjust(left=0.08, right=0.8, top=0.95, bottom=0.18, wspace=0.24)
fig.savefig(
    './spde_comparison.pdf',
    format='pdf',
    dpi=300)
# plt.show()
