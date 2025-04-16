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
nBatch = 5

variables = [("memory", "mem"), ("cost", "cost")]

errorTypes = ["maxError", "froError"]
errorType = errorTypes[1]

for variable, prefix in variables:

    #    baseDir = 'data'
    baseDir = os.path.join('experiments', 'publicationData')
    fileStr = create_data_string(DIM, var, ell, nu, nSampBatch,
                                 prefix + "_ACCUMULATED") \
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

    variables = {}
    errors = {}
    errorBars = {}

    with open(fileName, mode='r') as file:

        reader = csv.reader(file)
        rows = list(reader)

        for row in rows:

            label = row[0].rsplit('_', 1)

            method = label[0]

            print(method)

            if method not in methods:
                methods.append(method)

            variableName = label[1]

            print(variableName + "\n")

            if variableName == variable:
                if method not in variables:
                    variables[method] = [float(x) for x in row[1:]]

            if variableName == errorType:
                if method not in errors:
                    errors[method] = [float(x) for x in row[1:]]

            if variableName == "bars":
                if method not in errorBars:
                    errorBars[method] = [float(x) for x in row[1:]]

    # =============================================================================
    # Define Plot Styles
    # =============================================================================

    colors = {'SPDE_osFix': 'tab:blue',
              'DNA_fourier': 'tab:green',
              'DNA_spde': 'tab:red'}

    linestyles = ['-', '--', '-.', ':']

    markers = ['o', 's', 'D', '^', 'p']

    # =============================================================================
    fig, ax = plt.subplots(figsize=(figWidth, figHeight))

    # =============================================================================
    # Plot Data
    # =============================================================================

    # Plot SPDE methods
    for i, method in enumerate(methods):

        linestyle = linestyles[0]
        marker = markers[0]
        color = colors[method]

        ax.errorbar(
            variables[method],
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

    ax.set_title(f"d = {DIM}, ell = {ell}, nu = {nu}")

    errorLabel = "maximal error" if errorType == "maxError" else "relative Frobenius error"

    ax.set_xlabel(variable, fontsize=fontSizeLabel)
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
    ax.set_aspect('auto', adjustable='box')

    plt.savefig(
        f"./spde_comparison_{variable}.pdf",
        format='pdf',
        dpi=300,
        bbox_inches='tight')
    plt.show()
