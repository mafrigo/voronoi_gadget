# voronoi_gadget
Matteo Frigo (Max Planck Institute for Astrophysics, 2016 - 2022)

![Alt text](example_output.png?raw=true "Galaxy Kinematics example")

## Description
Creates mock integral field unit (IFU) galaxy images from galaxy simulations, like the one shown above. 
Each Voronoi bin (spaxel) contains roughly the same number of simulation particles in projection. 
This simulates the signal-to-noise criterion used in real IFU maps to determine the bin shape and size.
For a given Voronoi grid any quantity stored or derivable from the snapshot (line-of-sight velocity, metallicity, stellar age) can be plotted.
In the case of velocity maps, it calculates the higher order moments h_3 and h_4 with a Gauss-Hermite fit, in addition to mean and dispersion.

The code uses the [pygad](https://bitbucket.org/broett/pygad) library to open simulation snapshots, 
meaning it supports natively Gadget and Arepo simulations, but it can also be used on 
other formats with some tweaking.

It also uses the [vorbin](https://pypi.org/project/vorbin/) package by Michele Cappellari for creating the Voronoi grid, 
therefore any paper using this should cite [Cappellari & Copin 2003](http://adsabs.harvard.edu/abs/2003MNRAS.342..345C).

## Usage example:
```python
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
```
The outputs of this script will be saved in vel4map.png and ZH2map.png.
Note that you can also do everything in one step with:
```python
vg.voronoimap(snap, "vel", extent=4., npixel_per_side=50, nspaxels=100)
```

## Output examples:
See figure above, or figures 4,5,6,10,11 in Frigo et al. 2019 (https://arxiv.org/pdf/1811.11059.pdf).
