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
nSampBatch = int(5e1)
nBatch = 5

variables = [("memory", "mem"), ("cost", "cost")]

errorType = "maxError"

for variable, prefix in variables:

    baseDir = 'data'
    fileStr = create_data_string(DIM, var, ell, nu, nSampBatch,
                                 prefix + "_ACCUMULATED") \
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

    colors = {'SPDE_nFix': 'tab:blue', 'SPDE_osFix': 'tab:purple',
              'DNA_fourier': 'tab:green', 'DNA_spde': 'tab:red'}

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
        marker = markers[i]
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
    ax.set_xlabel(variable, fontsize=fontSizeLabel)
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
        f"./spde_comparison_{variable}.pdf",
        format='pdf',
        dpi=300,
        bbox_inches='tight')
    plt.show()
