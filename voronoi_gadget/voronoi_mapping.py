__all__ = ["voronoimap", "makevoronoimap"]

import matplotlib.pyplot as plt
import pygad
from voronoi_gadget.snapshot_tools import *
from voronoi_gadget.voronoi_grid import *
from voronoi_gadget.defaults import get_plot_config
from voronoi_gadget.plot_maker import makevoronoimap


def voronoimap(snap, qty='vel', grid=None, extent=20, npixel_per_side=200, partperbin=None, nspaxels=500,
               weightqty='mass', sigmapp=-1., npseudoparticles=60, qty_spread=0., force_orient=True, ensure_rotdir=True,
               scalebar=None, cutatmag=None, figureconfig='horizontal', info=None, plotfile=None, savetxt=None,
               savefigure=True, custom_titles=None, style='default'):
    """
    Plots a quantity using a Voronoi-binned grid, calculated so that each cell 
    contains approximately the same mass/light. The 4 default panels show average,
    dispersion, skewness and kurtosis of the given quantity.

    Input parameters:

    snap            : (Sub)snapshot to be analyzed.
    qty             : Quantity to be analyzed and plotted. If 3D quantity, its component along coordinate 1
                      (intermediate axis) is taken. Apart from the standard pygad blocks, also supports:
                      -'ZH' ([Z/H] in solar units (logarithmic))
                      -'alphafe' ([alpha/Fe] in solar units (logarithmic))
                      -'age' (Age in Gyr).
                      -'logage' (Age in Gyr (logarithmic)).
    extent          : Extent of the final image, in kpc; for instance, extent=20 means from -10 kpc to 10 kpc.
    npixel_per_side : npixel_per_side**2 is the number of regular pixels which form the base grid. The voronoi spaxels
                      are constructed by associating these.
    partperbin      : Target number of particles in each voronoi spaxel. If None, nspaxels spaxels are generated.
    nspaxels        : If partperbin is None, number of voronoi spaxels to be generated, otherwise it is unused. If
                      nspaxels==0, no Voronoi binning is performed and a regular 2D grid is used instead.
    weightqty       : Quantity over which the evaluation of "qty" is weighted.
    sigmapp         : If positive, each particle is expanded into a cloud of pseudo-particles distributed according to
                      a 3D gaussian with sigma sigmapp. If negative, no such expansion is performed.
    npseudoparticles: Number of pseudoparticles per original particle. Unused if sigmapp<0.
    force_orient    : Whether to forcefully reorient the snapshot so that the galaxy is seen edge-on (uses pygad's
                      prepare_zoom).
    ensure_rotdir   : If True, the rotation direction will be forced to be always the same by flipping the snapshot if
                      necessary.
    scalebar        : Adds a bar to the bottom left corner of the plot to show the scale of the figure. If
                      scalebar=='reff', the effective radius of the given subsnap will be used.
    cutatmag        : Instead of making the normal voronoiplot, cuts the figure at the given magnitude and saves it
                      without axes.
    figureconfig    : Configuration/orientation of the final figure; 'horizontal', 'vertical' or '22' (square).
    info            : Text that will appear on the left of the plot instead of "kpc".
    savetxt         : File on which to save the numerical data of the plot in text format. Does not do it if
                      savetxt=None.
    savefigure      : Whether to save the final figure.
    plotfile        : Name of the final plot file. Standard is qty+str(npanels)+'map' (e.g. "vel4map.png").
    style           : Plot style, as defined in config/style_config.yaml.
    """
    # Loading default parameters
    def_titles, cmap, cmaplimits, statsmode, addlambdar, centeriszero = get_plot_config(qty)
    if custom_titles is not None:
        titles = custom_titles
    else:
        titles = def_titles
    if statsmode == "fit":
        npanels = 4
    else:
        npanels = 2
    if plotfile is None:
        plotfile = qty + str(npanels) + 'map'

    # Loading and preparing snapshot
    if type(snap) == str:
        snap, = pygad.prepare_zoom(snap)
    if force_orient:
        snap = orient_snap(snap, axisorientation=1, ensure_rotdir=ensure_rotdir)
    if scalebar == 'reff':
        scalebar = pygad.analysis.half_mass_radius(snap, proj=1)
    if sigmapp > 0.:
        snap = PseudoSnap(snap, npseudoparticles, sigmapp, qty_spread)

    # Calculating Voronoi grid and statistics in each bin
    if grid is None:
        grid = VoronoiGrid(snap, extent, npixel_per_side=npixel_per_side, partperbin=partperbin, nspaxels=nspaxels)
    plotquantity = grid.get_stats(qty, weightqty=weightqty, mode=statsmode, centeriszero=centeriszero)

    # Make plot
    makevoronoimap(plotquantity, grid, npanels=npanels, fluxqty=weightqty, figureconfig=figureconfig, cmap=cmap,
                   cmaplimits=cmaplimits, titles=titles, titles2=info, plotfile=plotfile, savefigure=savefigure,
                   cutatmag=cutatmag, addlambdar=addlambdar, savetxt=savetxt, scalebar=scalebar, style=style)
    if savefigure:
        plt.close()
