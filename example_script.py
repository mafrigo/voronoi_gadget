import pygadmpa as pygad
import voronoi_gadget as vg

# Load a gadget snapshot file into a snap object with pygad
snap, = pygad.prepare_zoom(filename)

# Create a Voronoi grid for this snapshot with extent 4 kpc
grid = vg.VoronoiGrid(snap, 4., npixel_per_side=50, nspaxels=100)

# Plot the desired quantity (line of sight velocity) on the grid
vg.makevoronoimap("vel", grid, cmap='sauron')
# Note: coordinate 1 (intermediate axis if oriented edge-on) is used as "line of sight"

# Plot another quantity (metallicity) on the same grid
vg.makevoronoimap("ZH", grid, cmap='viridis')

# Outputs will be saved in vel4map.png and ZH2map.png

# Note: you can also do everything in one step with:
# vg.voronoimap(snap, "vel", extent=4., npixel_per_side=50, nspaxels=100)
