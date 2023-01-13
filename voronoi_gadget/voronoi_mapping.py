__all__ = ["voronoimap", "makevoronoimap"]

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as mplax
import scipy
import pygad
from voronoi_gadget.snapshot_tools import *
from voronoi_gadget.voronoi_grid import *
from voronoi_gadget.sauron_cmap import *
from voronoi_gadget.lambdar import _lambdar
from voronoi_gadget.defaults import getdefaultplotparams, get_style_config


def voronoimap(snap, qty='vel', extent=20,
               npixel_per_side=200, partperbin=None, nspaxels=500,
               statsmode='default', weightqty='mass',
               sigmapp=-1., npseudoparticles=60,
               force_orient=True, ensure_rotdir=True,
               artificial_error=0., measerror=0., scalebar=None,
               cutatmag=None, figureconfig='horizontal',
               info=None, plotfile=None, savetxt=None,
               savefigure=True, custom_titles=None):
    """
    Plots a quantity using a Voronoi-binned grid, calculated so that each cell 
    contains approximately the same mass/light. The 4 default panels show average,
    dispersion, skewness and kurtosis of the given quantity.

    Input parameters:

    snap            : (Sub)snapshot to be analyzed.
    qty             : Quantity to be analyzed and plotted. If 3D quantity, the 
                      z-direction (line-of-sight) is taken. 
                      Apart from the standard pygad blocks, also supports: 
                      -'ZH' ([Z/H] in solar units (logarithmic))
                      -'alphafe' ([alpha/Fe] in solar units (logarithmic))
                      -'age' (Age in Gyr).
                      -'logage' (Age in Gyr (logarithmic)).
                      -'lambdar' lambda_R parameter for every spaxel(1st panel only).
    extent          : Extent of the final image, in kpc; for instance, extent=20 
                      means from -10 kpc to 10 kpc.
    npixel_per_side : npixel_per_side**2 is the number of regular pixels which form 
                      the base grid. The voronoi spaxels are constructed by 
                      associating these.
    partperbin      : Target number of particles in each voronoi spaxel. If None, 
                      nspaxels spaxels are generated.
    nspaxels        : If partperbin is None, number of voronoi spaxels to be 
                      generated, otherwise it is unused. If nspaxels==0, no Voronoi 
                      binning is performed and a regular 2D grid is used instead.
    weightqty       : Quantity over which the evaluation of "qty" is weighted.
    sigmapp         : If positive, each particle is expanded into a cloud of 
                      pseudo-particles distributed according to a 3D gaussian with 
                      sigma sigmapp. If negative, no such expansion is performed.
    npseudoparticles: Number of pseudoparticles per original particle. Unused if 
                      sigmapp<0.
    force_edgeon    : Whether to forcefully reorient the snapshot so that the galaxy 
                      is seen edge-on (uses pygad's prepare_zoom).
    ensure_rotdir   : If True, the rotation direction will be forced to be always
                      the same by flipping the snapshot if necessary.
    artificial_error: If specified, quantity is randomly displaced according to a 
                      gaussian with sigma=artificial_error.
    measerror       : The measurement error of every value; only affects the 
                      calculation if statsmode='likelihood'.
    scalebar        : Adds a bar to the bottom left corner of the plot to show the
		              scale of the figure. If scalebar=='reff', the effective radius
                      of the given subsnap will be used.
    cutatmag        : Instead of making the normal voronoiplot, cuts the figure at 
                      the given magnitude and saves it without axes.
    figureconfig    : Configuration/orientation of the final figure; 'horizontal', 
                      'vertical' or '22' (square).
    info            : Text that will appear on the left of the plot instead of "kpc".
    savetxt         : File on which to save the numerical data of the plot in text 
                      format. Does not do it if savetxt=None.
    savefigure      : Whether to save the final figure.
    plotfile        : Name of the final plot file. Standard is qty+str(npanels)+'map'
                      (e.g. "vel4map.png").
    """
    if force_orient:
        snap = orient_snap(snap, axisorientation=1, ensure_rotdir=ensure_rotdir)
    if scalebar == 'reff':
        scalebar = pygad.analysis.half_mass_radius(snap, proj=1)
    if sigmapp > 0.:
        snap = PseudoSnap(snap, npseudoparticles, sigmapp)

    def_titles, cmap, cmaplimits, statsmode, addlambdar, centeriszero = getdefaultplotparams(qty)
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

    grid = VoronoiGrid(snap, extent, npixel_per_side=npixel_per_side, partperbin=partperbin, nspaxels=nspaxels)
    plotquantity = grid.get_stats(qty, weightqty=weightqty, mode=statsmode,
                                  artificial_error=artificial_error, measerror=measerror, centeriszero=centeriszero)
    makevoronoimap(plotquantity, grid, npanels=npanels, fluxqty=weightqty, figureconfig=figureconfig, cmap=cmap,
                   cmaplimits=cmaplimits, titles=titles, titles2=info, plotfile=plotfile, savefigure=savefigure,
                   cutatmag=cutatmag, addlambdar=addlambdar, savetxt=savetxt, scalebar=scalebar)
    if savefigure:
        plt.close()


def makevoronoimap(plotquantity, grid, npanels=4, fluxcontours='smooth',
                   fluxqty='mass', cmap='sauron',
                   cmaplimits=["deduce", "minmax", "symmetric", "symmetric"],
                   figsize=5, figureconfig='horizontal',
                   titles=['Average', 'Dispersion', r'$h_3$', r'$h_4$'],
                   titles2=None, titles3=r"$\rm kpc$", cutatmag=None,
                   addlambdar=True, cutatrad=None, scalebar=None,
                   savetxt=None, savefigure=True, plotfile='voronoimap',
                   **kwargs):
    """
    Plots a Voronoi-binned image.

    Usage:
    
    plotquantity    : Output of VoronoiGrid.get_stats(). It's the average, dispersion,
                      h3, h4 of every Voronoi spaxel.
    grid            : VoronoiGrid object.
    npanels         : Number of panels/moments to be displayed. npanels=1 means 
                      just the average, npanels=2 average and dispersion, npanels=4 
                      average, dispersion, h3 and h4.
    fluxcontours    : Whether and how to overplot flux contours on the Voronoi figure. 
                      It accepts the values:
                      -'voronoi': calculates the contours on the Voronoi 
                          grid, for a faster calculation (recommended). 
                      -'regular': calculates the flux in every pixel, which takes 
                          longer and can show numerical noise, but is more precise.
                      -'smooth': same as regular but with gaussian smoothing.
                      -None: no contours are plotted.
    fluxqty         : Quantity used to compute the flux contours.
    cmap            : Colormap to be used. If None, uses 'sauron' colormap for 
                      kinematics and '' for age and Z plots.
    cmaplimits      : Colormap limits. Must be in the form [[minavg, maxavg], 
                      [mindisp, maxdisp], [minh3,maxh3], [minh4,maxh4]].
    figsize         : Height (width) of the figure when figureconfig=='horizontal' 
                      (=='vertical').
    figureconfig    : Orientation of the final figure; 'horizontal', 'vertical' or '22'.
    titles          : Titles of the 4 plots.
    titles2         : x/y labels of the plots (for figureconfig=horizontal/vertical). 
                      Can be used to display additional info.
    titles3         : y/x label of the figure (for figureconfig=horizontal/vertical).
    cutatmag        : Instead of making the normal voronoiplot, cuts the figure at the 
                      given magnitude and saves it without axes. 
    addlambdar      : Whether to calculate and display the lambda_r parameter in 
                      the figure on the lower right corner of the first panel.
    cutatrad        : Same as cutatmag, but cutting at a given radius instead.
    scalebar        : If not None, adds a bar of the given length (in kpc) 
		      to the bottom left corner of the plot.
    savetxt         : File on which to save data in text format. Doesn't do it if 
                    savetxt==None.
    savefigure      : If True the final figure is saved in the current directory. 
                      If False it is just shown.
    plotfile        : Name of the output file.
    """
    cfg = get_style_config()

    if titles3 is None:
        titles3 = r"$\rm kpc$"

    if fluxcontours == 'voronoi':
        print("Calculating flux in each spaxel for plotting contours")
        flux = np.zeros(len(set(grid.binNum)))
        fluxsnap = grid._snap[fluxqty]
        for bn in set(grid.binNum):
            # Mass of each spaxel divided by its area
            flux[bn] = np.sum(fluxsnap[grid._spaxelofpart == bn]) / (len(grid.binNum[grid.binNum == bn]))
        xflux = grid.xBar
        yflux = grid.yBar
        fluxvoro = flux
    else:
        fluxvoro = np.zeros(np.shape(grid.xBar))
    if fluxcontours == 'regular' or fluxcontours == 'smooth':
        print("Calculating flux in each pixel for plotting contours")
        flux, xedges, yedges, pixelofpart0 = scipy.stats.binned_statistic_2d(
            x=grid._snap["pos"][:, 0], y=grid._snap["pos"][:, 1], values=grid._snap[fluxqty],
            statistic='sum', bins=grid.npixel_per_side,
            range=[[-grid.extent / 2., grid.extent / 2.], [-grid.extent / 2., grid.extent / 2.]])
        if fluxcontours == 'smooth':
            # gaussian smoothing with sigma 2 pixels
            flux = scipy.ndimage.gaussian_filter(flux, int(grid.npixel_per_side / 100.))  # 1.)
        flux = flux.reshape(grid.npixel_per_side ** 2)
        xflux = grid.xvor
        yflux = grid.yvor
    if fluxcontours is None:
        flux = np.zeros(len(set(grid.binNum)))
        xflux = None
        yflux = None

    print("Preparing plot")
    if figureconfig == 'horizontal':
        plt.figure(figsize=(figsize * npanels, figsize))
    if figureconfig == 'vertical':
        plt.figure(figsize=(figsize, figsize * npanels))
    if figureconfig == '22':
        plt.figure(figsize=(2. * figsize, 2. * figsize))

    if cutatmag is not None:
        for i in np.arange(len(flux)):
            if -2.5 * np.log10(flux[i] / np.max(flux)) >= cutatmag:
                plotquantity[i] = None

    if cutatrad is not None:
        for i in np.arange(len(flux)):
            if grid.xBar[i] ** 2 + grid.yBar[i] ** 2 >= cutatrad ** 2:
                plotquantity[i] = None

    if cmap == 'viridis':
        textcolor = 'w'  # for better visualization
    else:
        textcolor = 'k'

    if (xflux.any() is None and yflux.any() is None):  # added .any
        xflux, yflux = grid.xBar, grid.yBar
    if titles2 is None:
        titles2 = 2 * [" "]

    # Determining colorbar limits
    cmaplimits = deduce_cbar_limits(plotquantity, cmaplimits)

    for i in np.arange(npanels):
        # Determining titles and configuration
        if figureconfig == 'horizontal':
            plt.subplot(int(100 + npanels * 10 + (i + 1)))
            plt.gca().tick_params(axis='x', labelbottom='off')
            if i == 0:
                plt.ylabel(titles3, fontsize=cfg["textsize"])
            else:
                plt.gca().tick_params(labelleft='off')  # left
        if figureconfig == 'vertical':
            plt.subplot(int(npanels * 100 + 10 + (i + 1)))
            plt.ylabel(titles2[i], fontsize=cfg["textsize"], labelpad=-20)
            if i == npanels - 1:
                plt.xlabel(titles3, fontsize=cfg["textsize"])
        if figureconfig == '22':
            plt.subplot(220 + (i + 1))
            if titles2 == [" ", " ", " ", " "]:
                if i == 0 or i == 2:
                    plt.ylabel(r"$\rm kpc$", fontsize=cfg["textsize"], labelpad=-20)
            else:
                plt.ylabel(titles2[i], fontsize=cfg["textsize"], labelpad=-20)
            if i == npanels - 1 or i == npanels - 2:
                plt.xlabel(titles3, fontsize=cfg["textsize"])
        if i == 0:
            minx = -0.5 * grid.extent  # np.min(xvor)
            miny = -0.5 * grid.extent  # np.min(yvor)
            xlength = grid.extent  # np.max(xvor)-np.min(xvor)
            ylength = grid.extent  # np.max(yvor)-np.min(yvor)
            # Custom titles in left side of first panel
            if len(titles2) > 0:  # above and below
                plt.text(minx + 0.03 * xlength, miny + 0.03 * ylength, titles2[0], fontsize=cfg["labelsize"], color=textcolor)
                plt.text(minx + 0.03 * xlength, miny + 0.89 * ylength, titles2[1], fontsize=cfg["labelsize"], color=textcolor)
            else:  # just below
                plt.text(minx + 0.03 * xlength, miny + 0.03 * ylength, titles2, fontsize=cfg["labelsize"], color=textcolor)

            if scalebar is not None:  # scale bar on the left axis
                # plt.axhline(y=np.min(yvor)+0.89*ylength, linewidth=2, color='k',
                # xmin=0.95-scalebar/xlength, xmax=0.95)
                if isinstance(scalebar, float):
                    plt.axvline(x=minx - 0.005 * xlength, linewidth=2, color='k',
                                ymin=0.5, ymax=0.5 + scalebar / ylength, clip_on=False)
                elif len(scalebar) == 2:
                    plt.axvline(x=minx - 0.005 * xlength, linewidth=2, color='k',
                                ymin=0.5, ymax=0.5 + scalebar[0] / ylength, clip_on=False)
                    plt.axvline(x=minx - 0.005 * xlength, linewidth=2, color='k',
                                ymin=0.5 - scalebar[1] / ylength, ymax=0.5, clip_on=False,
                                linestyle='dashed')
                else:
                    print("Warning: Scalebar format not accepted")

            if addlambdar and cutatmag is None:
                reff = pygad.analysis.half_mass_radius(grid._snap, proj=2)
                print("Effective radius: " + str(reff))
                lambdaR = _lambdar(grid, plotquantity, rmax=reff)[-1]
                plt.text(minx + 0.6 * xlength, miny + 0.03 * ylength, r"$\lambda_R = %.2f$" % (lambdaR),
                         fontsize=cfg["labelsize"], color=textcolor)

        # Actually plotting the colors
        img = display_bins(grid.xvor, grid.yvor, grid.binNum, plotquantity[:, i], vmin=cmaplimits[i][0],
                           vmax=cmaplimits[i][1], cmap=cmap, **kwargs)

        ax = plt.gca()
        if cutatmag is None:
            # Plotting titles
            plt.title(titles[i], fontsize=cfg["titlesize"], y=1.04)

            # Flux contours
            if fluxcontours is not None:
                try:
                    mag = -2.5 * np.log10(flux / np.max(flux).ravel())
                    plt.tricontour(xflux, yflux, mag,
                                   levels=np.arange(5), colors='k',  # 20
                                   linewidths=cfg["contourthickness"])
                except:
                    print("Warning: Impossible to print contours; problem with the data")
                    print(xflux, yflux, flux)

            # Adjusting axes ticks
            ax.tick_params(labelsize=cfg["digitsize"])
            ax.tick_params(direction="in", which='both')
            # locs= ax.get_yticks()
            locs = np.round(np.linspace(-0.5 * grid.extent, 0.5 * grid.extent, cfg["naxeslabels"]), 2)
            ax.xaxis.set_ticks(locs)
            ax.yaxis.set_ticks(locs)
            plt.xlim([-0.5 * grid.extent, 0.5 * grid.extent])
            plt.ylim([-0.5 * grid.extent, 0.5 * grid.extent])

            # Determining and plotting colorbar ticks
            divider = mplax.make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            if cfg["ncbarlabels"] is None:
                cb = plt.colorbar(img, cax=cax)
                cb.ax.tick_params(labelsize=cfg["digitsize"])
            else:  # Calculating optimal colormap tick distribution
                cspan = (cmaplimits[i][1] - cmaplimits[i][0]) / (cfg["ncbarlabels"] - 1)
                # cspan: separation between colorbar ticks/labels
                cspanorder = int(np.round(np.log10(cspan)))
                # recalculating cspan in case lower limit needs to be truncated
                if cmaplimits[i][1] != -cmaplimits[i][0]:
                    if cmaplimits[i][0] == 0.:
                        cmaplimits[i][0] = np.min(plotquantity[:, i])
                    # limitorder=int(np.floor(np.log10(abs(cmaplimits[i][0]))))
                    limitorder = int(np.floor(np.log10(abs(cmaplimits[i][1]))))
                    lowerlimit = (10. ** (limitorder - 1)) * np.ceil(cmaplimits[i][0] / (10. ** (limitorder - 1)))
                    cspan = (cmaplimits[i][1] - lowerlimit) / (cfg["ncbarlabels"] - 1)
                    # truncating cspan to avoid lots of decimals
                cspan = (10. ** (cspanorder - 1)) * np.trunc(cspan / (10. ** (cspanorder - 1)))
                if cspanorder > 0:
                    cspan = int(cspan)
                if cmaplimits[i][1] == -cmaplimits[i][0]:
                    cticks = []
                    for itick in np.arange(cfg["ncbarlabels"]):
                        cticks.append(cspan * (itick - ((cfg["ncbarlabels"] - 1) / 2.)))
                else:
                    cticks = []
                    for itick in np.arange(cfg["ncbarlabels"]):
                        cticks.append(lowerlimit + cspan * (itick))
                print("Colorbar ticks (" + str(i) + "): " + str(cticks))
                cb = plt.colorbar(img, cax=cax, ticks=cticks)
                cb.ax.tick_params(labelsize=cfg["digitsize"] * 1.1)

        plt.subplots_adjust(wspace=0.3)

    if savefigure:
        if cutatmag is not None:  # Figure cut at a given isophote, without axes
            plt.axis('off')
            # plt.tight_layout()
            plt.subplots_adjust(wspace=0.01)
            print("Saving cut figure in " + plotfile + ".png")
            plt.savefig(plotfile, bbox_inches='tight', transparent=True)
            return
        else:
            print("Saving figure in " + plotfile + ".png")
            plt.savefig(plotfile, bbox_inches='tight')
    plt.pause(0.01)

    if savetxt is not None:
        np.savetxt(savetxt + '.kin', np.transpose(np.array([plotquantity[:, 0], plotquantity[:, 1],
                                                            plotquantity[:, 2], plotquantity[:, 3],
                                                            fluxvoro, grid.xBar, grid.yBar])))
        np.savetxt(savetxt + '.grid', np.transpose(np.array([grid.xvor, grid.yvor, grid.binNum])))


def deduce_cbar_limits(plotquantity, cmaplimits):
    for ipanel in np.arange(len(cmaplimits)):
        if cmaplimits[ipanel] == 'deduce':
            if np.min(plotquantity[:, ipanel]) * np.max(plotquantity[:, ipanel]) < 0.:
                cmaplimits[ipanel] = 'symmetric'
            else:
                cmaplimits[ipanel] = 'minmax'

        if cmaplimits[ipanel] in ['symmetric', 'minmax']:
            pltqty = plotquantity[:, ipanel][
                np.logical_and(plotquantity[:, ipanel] != -np.inf, plotquantity[:, ipanel] != np.inf)]
            if cmaplimits[ipanel] == 'symmetric':
                cmaplimits[ipanel] = [-np.max(abs(pltqty)), np.max(abs(pltqty))]
            elif cmaplimits[ipanel] == 'minmax':
                cmaplimits[ipanel] = [np.min(pltqty), np.max(pltqty)]
            else:
                pass
    print("Colorbar limits: " + str(cmaplimits))
    return cmaplimits


def display_bins(x, y, binNum, qtyBin, ax=None, cmap='Sauron', plotextent_fac=1., **kwargs):
    """
    Adapted from vorbin (Michele Cappellari)
    """
    from scipy.spatial import distance

    if ax is None:
        ax = plt.gca()
    if cmap == 'sauron':
        cmap = sauron

    if not (x.size == y.size == binNum.size):
        raise ValueError('The vectors (x, y, binNum) must have the same size')

    if np.unique(binNum).size != qtyBin.size:
        raise ValueError('qtyBin size does not match number of bins')

    if np.unique(binNum).size > 1:
        val = qtyBin[binNum]
    else:
        val = np.full(np.shape(x), qtyBin)

    pixelsize = np.min(distance.pdist(np.column_stack([x, y])))

    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    nx = int(round((xmax - xmin) / pixelsize) + 1)
    ny = int(round((ymax - ymin) / pixelsize) + 1)
    j = np.round((x - xmin) / pixelsize).astype(int)
    k = np.round((y - ymin) / pixelsize).astype(int)
    mask = np.ones((nx, ny), dtype=bool)
    img = np.empty((nx, ny))
    mask[j, k] = 0
    img[j, k] = val
    img = np.ma.masked_array(img, mask)
    img = ax.imshow(np.rot90(img), interpolation='none', cmap=cmap,
                    extent=[plotextent_fac * (xmin - pixelsize / 2), plotextent_fac * (xmax + pixelsize / 2),
                            plotextent_fac * (ymin - pixelsize / 2), plotextent_fac * (ymax + pixelsize / 2)],
                    **kwargs)
    ax.minorticks_on()
    ax.tick_params(length=10, width=1, which='major')
    ax.tick_params(length=5, width=1, which='minor')
    ax.tick_params(top=True, right=True, direction='in', which='both')

    return img
