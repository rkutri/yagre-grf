import os
import numpy as np
import matplotlib.pyplot as plt
import csv
from experiments.filename import create_data_string

# Set parameters for filename generation
ell = 0.2
nu = 1.
nSampBatch = int(1e5)
nBatch = 4

# Construct filename
baseDir = 'data'
fileStr = create_data_string(
    ell, nu, nSampBatch, "ACCUMULATED_ERRORS_OVERSAMPLING"
) + f"_{nBatch}batches.csv"
filename = os.path.join(baseDir, fileStr)

# Read data from CSV
mesh_widths = []
methods = []
errors = {}
errors_std = {}

with open(filename, mode='r') as file:
    reader = csv.reader(file)
    rows = list(reader)

    # Read mesh widths dynamically from first row
    mesh_widths = [float(value) for value in rows[0][1:]]

    # Read method names and values
    i = 1
    while i < len(rows):
        method = rows[i][0]
        methods.append(method)
        errors[method] = [float(value) for value in rows[i][1:]]
        
        # Read corresponding standard deviation (if available)
        if i + 1 < len(rows) and "std" in rows[i + 1][0].lower():
            errors_std[method] = [float(value) for value in rows[i + 1][1:]]
            i += 2  # Move to the next method
        else:
            errors_std[method] = None
            i += 1

# Define plot styles
spde_methods = [m for m in methods if "spde_alpha" in m]
dna_methods = ["DNA_fourier", "DNA_spde"]

linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (3, 2, 1, 2))]
spde_color = 'tab:blue'
dna_colors = {'DNA_fourier': 'tab:green', 'DNA_spde': 'tab:red'}

# Define plot parameters
line_width = 1.5  # Thinner lines
marker_size = 8  # Smaller marker size for dots
icon_size = 4  # Smaller icon size for markers
font_size_label = 12  # Axis labels font size
font_size_ticks = 8  # Ticks font size
font_size_legend = 6  # Legend font size
legend_marker_scale = 0.5

# Set up figure size for three figures per row in 11pt DIN A4 LaTeX
fig_width = 5.0  # Narrower width to reduce horizontal stretch
fig_height = 4.0  # Square shape, balanced aspect ratio
fig, ax = plt.subplots(figsize=(fig_width, fig_height))

# Plot SPDE methods (same color, different linestyles)
for i, method in enumerate(spde_methods):
    linestyle = linestyles[i % len(linestyles)]
    ax.errorbar(
        mesh_widths,
        errors[method],
        yerr=errors_std[method] if errors_std[method] else None,
        fmt='o',
        linestyle=linestyle,
        label=method.replace("_", " "),
        markersize=marker_size,
        markeredgewidth=1.,
        elinewidth=1.,
        capsize=4,
        color=spde_color,
        alpha=0.9
    )

# Plot DNA methods (unique colors)
for method in dna_methods:
    if method in errors:
        ax.errorbar(
            mesh_widths,
            errors[method],
            yerr=errors_std[method] if errors_std[method] else None,
            fmt='s',
            linestyle='-',  # Solid for distinction
            label=method.replace("_", " "),
            markersize=marker_size,
            markeredgewidth=1.,
            elinewidth=1.,
            capsize=4,
            color=dna_colors[method],
            alpha=0.9
        )

# Draw triangle for convergence rate reference
rate = 0.5  # Example rate 1/2
base = 0.025  # Adjusted base (smaller)
height = rate * base

# Shifted triangle position down and to the left
x_triangle = [0.05, 0.05 + base]
y_triangle = [0.02, 0.02 + height]

# Triangle lines (apply line width to all lines)
ax.plot([x_triangle[0], x_triangle[1]], [y_triangle[0], y_triangle[0]], color='black', lw=line_width)  # Base
ax.plot([x_triangle[1], x_triangle[1]], [y_triangle[0], y_triangle[1]], color='black', lw=line_width)  # Height
ax.plot([x_triangle[0], x_triangle[1]], [y_triangle[0], y_triangle[1]], color='black', lw=line_width)  # Hypotenuse

# Annotate catheti lengths
ax.text(x_triangle[0] + 0.5 * base, y_triangle[0] - 0.006, '1', fontsize=font_size_ticks, ha='center')
ax.text(x_triangle[1] + 0.003, y_triangle[1] - 0.6 * height, '1/2', fontsize=font_size_ticks, va='center')

# Customize axes
ax.set_xlabel('Mesh width $h$', fontsize=font_size_label)
ax.set_ylabel('Error estimate', fontsize=font_size_label)
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Add legend outside the figure
handles, labels = ax.get_legend_handles_labels()
legend = ax.legend(
    handles,
    labels,
    fontsize=font_size_legend,  # Reduced font size for legend
    loc='upper left',
    bbox_to_anchor=(1.02, 1),  # Placing outside, to the right
    frameon=True,
    framealpha=0.9,
    markerscale=legend_marker_scale
)

# Set alpha for legend text and markers
for text in legend.get_texts():
    text.set_alpha(1)

for line in legend.get_lines():
    line.set_alpha(1)

# Adjust layout for LaTeX figure arrangement (three figures in one row)
plt.tight_layout(rect=[0, 0, 0.9, 1])  # Make space for the legend outside the plot

# Adjust aspect ratio to more balanced (closer to 1:1)
ax.set_aspect('auto', adjustable='box')  # Make it more balanced and less stretched

# Save and show the plot
plt.savefig('./spde_oversampling.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.show()

