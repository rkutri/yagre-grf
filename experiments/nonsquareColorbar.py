import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Settings
cmap_name = "turbo"
vmin, vmax = 0.0, 4.0
ticks = [0, 1, 2, 3, 4]
fontsize = 28

# Create a dummy ScalarMappable for the colorbar
norm = Normalize(vmin=vmin, vmax=vmax)
cmap = plt.colormaps[cmap_name]
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])  # required in older matplotlib versions

# Create a figure with no axes, only the colorbar
fig, ax = plt.subplots(figsize=(1.0, 4.0))  # Tall narrow figure for vertical colorbar
fig.subplots_adjust(left=0.4, right=0.6, top=0.95, bottom=0.05)

# Add the colorbar
cbar = fig.colorbar(sm, cax=ax, orientation='vertical', ticks=ticks, extend='both')
cbar.set_ticklabels([str(t) for t in ticks])
cbar.ax.tick_params(labelsize=fontsize)

# Save the colorbar
plt.savefig("standalone_colorbar.png", dpi=800, bbox_inches='tight')
plt.show()

