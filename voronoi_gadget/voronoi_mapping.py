"""
by Matteo Frigo (Max Planck Institute for Astrophysics)
Last update: 14 May 2018

Plots voronoi-binned maps of velocity, metallicity and age of Gadget snapshots. 
Can plot average, dispersion, h3 and h4. Can plot separately the maps for particles
selected by a certain property (ex: age larger or smaller than 7 Gyr). Uses voronoi
binning routines by Cappellari & Copin (2003), available at: 

http://www-astro.physics.ox.ac.uk/~mxc/software/voronoi_python_2016-04-12.zip

"""

__all__ = ["voronoimap", "makevoronoimap", "lambdar"]

import pygad
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as mplax
import scipy
from matplotlib import colors
from voronoi_gadget.snapshot_tools import *
from voronoi_gadget.voronoi_grid import *

# Style parameters for the standard Voronoi plot
labelsize = 17  # 20
digitsize = 14  # 18
textsize = 23  # 27
titlesize = 25  # 30
naxeslabels = 5
ncbarlabels = 5
contourthickness = 0.8


def voronoimap(snap, qty='vel', npanels=4, extent=20,
               npixel_per_side=200, partperbin=None, nspaxels=500,
               statsmode='default', weightqty='mass', selectqty=None,
               selectbounds=None, sigmapp=-1., npseudoparticles=60, cmap=None,
               cmaplimits=None, force_edgeon=True, ensure_rotdir=True,
               artificial_error=0., measerror=0., scalebar='reff',
               cutatmag=None, addlambdar='default', figureconfig='horizontal',
               info=None, centeriszero='default', plotfile=None, savetxt=None,
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
    npanels         : Number of panels/moments to be displayed. With npanels=1 only 
                      the average is shown, with npanels=2 average and dispersion, 
                      and with npanels=4 average, dispersion, h3, and h4.
    extent          : Extent of the final image, in kpc; for instance, extent=20 
                      means from -10 kpc to 10 kpc.
    subsnap         : Which subsnap to pick (e.g. 'stars' or 'gas')
    npixel_per_side : npixel_per_side**2 is the number of regular pixels which form 
                      the base grid. The voronoi spaxels are constructed by 
                      associating these.
    partperbin      : Target number of particles in each voronoi spaxel. If None, 
                      nspaxels spaxels are generated.
    nspaxels        : If partperbin is None, number of voronoi spaxels to be 
                      generated, otherwise it is unused. If nspaxels==0, no Voronoi 
                      binning is performed and a regular 2D grid is used instead. 
    statsmode       : Method for calculating average and dispersion, h3 and h4 in 
                      each spaxel, as used by gauss_hermite_fit(). 'default' means 
                      'sample' for most quantities and 'fit' for velocity.
    weightqty       : Quantity over which the evaluation of "qty" is weighted.
    selectqty       : Quantity to be used for slicing the snapshot in different 
                      parts. Can be 'age', 'Z', 'vel', or None. If None, plots all 
                      particles together (no slicing).
    selectbounds    : Boundaries for the various slices. Example: 
                      selectbounds=[0,7,13] produces 2 slices, 0<selectqty<7 and 
                      7<selectqty<13.
    sigmapp         : If positive, each particle is expanded into a cloud of 
                      pseudo-particles distributed according to a 3D gaussian with 
                      sigma sigmapp. If negative, no such expansion is performed.
    npseudoparticles: Number of pseudoparticles per original particle. Unused if 
                      sigmapp<0.
    cmap            : Color map for the plot. In addition to the standard matplotlib 
                      ones, the 'sauron' one (Sauron, Atlas3D,...) is also available.
    cmaplimits      : Color map limits for the 4 panels. Must be in the form 
                      [[minavg, maxavg], [mindisp, maxdisp], [minh3,maxh3], [minh4,
                      maxh4]].
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
    addlambdar      : Whether to calculate and display the lambda_r parameter in 
                      the figure on the lower right corner of the first panel. If 
                      'default', lambda_r is displayed only when plotting velocities.
    figureconfig    : Configuration/orientation of the final figure; 'horizontal', 
                      'vertical' or '22' (square).
    info            : Text that will appear on the left of the plot instead of "kpc".
    centeriszero    : If True, the first panel of the map gets normalized so that
                      the central area has average zero. If 'default', this is True
                      only for velocity maps.
    savetxt         : File on which to save the numerical data of the plot in text 
                      format. Does not do it if savetxt=None.
    savefigure      : Whether to save the final figure.
    plotfile        : Name of the final plot file. Standard is qty+str(npanels)+'map'
                      (e.g. "vel4map.png").
    """

    if force_edgeon:
        snap = orient_snap(snap, axisorientation=1)

    if ensure_rotdir:
        # ensures that the angular momentum vector is always pointing in the same direction
        rangmom = 0.5 * extent
        angmomy = snap["mass"] * (snap["vel"][:, 2] * snap["pos"][:, 1] - snap["vel"][:, 1] * snap["pos"][:, 2])
        if np.mean(angmomy[snap["pos"][:, 0] ** 2 + snap["pos"][:, 1] ** 2 < rangmom ** 2]) > 0:
            snap["pos"] = -snap["pos"]

    def_titles, defaultcmap, defaultcmaplimits = getdefaultplotparams(qty, statsmode,
                                                                      npanels=npanels)
    if custom_titles is not None:
        titles = custom_titles
    else:
        titles = def_titles

    if cmap is None:
        cmap = defaultcmap
    if cmaplimits is None:
        cmaplimits = defaultcmaplimits
    if plotfile is None:
        plotfile = qty + str(npanels) + 'map'

    if statsmode == 'default':
        if npanels > 2:
            statsmode = 'fit'
        else:
            statsmode = 'sample'
    if addlambdar == 'default':
        if qty == 'vel' or qty == 'lambdar':
            addlambdar = True
        else:
            addlambdar = False
    if centeriszero == 'default':
        if qty == 'vel' or qty == 'lambdar':
            centeriszero = True
        else:
            centeriszero = False

    if scalebar == 'reff':
        scalebar = pygad.analysis.half_mass_radius(snap, proj=2)

    if selectqty is None:
        if sigmapp > 0.:
            snap = PseudoSnap(snap, npseudoparticles, sigmapp)
        grid = VoronoiGrid(snap, extent, npixel_per_side=npixel_per_side,
                           partperbin=partperbin, nspaxels=nspaxels)
        plotquantity = grid.get_stats(qty, weightqty=weightqty, mode=statsmode,
                                      artificial_error=artificial_error,
                                      measerror=measerror, centeriszero=centeriszero)
        fluxquantity = weightqty  # snap[weightqty]
        makevoronoimap(plotquantity, grid, npanels=npanels, fluxqty=fluxquantity,
                       figureconfig=figureconfig, cmap=cmap, cmaplimits=cmaplimits,
                       titles=titles, titles2=info, plotfile=plotfile,
                       savefigure=savefigure, cutatmag=cutatmag,
                       addlambdar=addlambdar, ncbarlabels=ncbarlabels,
                       savetxt=savetxt, scalebar=scalebar)

    else:  # for voronoimaps separated by age/Z
        selectquantity = snap[selectqty]
        if selectbounds is None:
            selectbounds = ['min', 'median', 'max']
        selecttitle, selectcmap, selectcmaplimits = getdefaultplotparams(selectqty, select=True)
        for i in np.arange(len(selectbounds)):
            if type(selectbounds[i]) == str:
                if selectbounds[i] == "min":
                    selectbounds[i] = np.min(selectquantity)
                    print("Min " + selectqty + ": " + str(selectbounds[i]))
                if selectbounds[i] == "median":
                    selectbounds[i] = (np.max(selectquantity) - np.min(selectquantity)) / 2.
                    print("Median " + selectqty + ": " + str(selectbounds[i]))
                if selectbounds[i] == "max":
                    selectbounds[i] = np.max(selectquantity)
                    print("Max " + selectqty + ": " + str(selectbounds[i]))
            if i > 0:
                plt.clf()
                print("\n Producing figure with " + str(selectbounds[i - 1]) + " < " + \
                      selectqty + " < " + str(selectbounds[i]))
                if info is None:
                    info = ["", "%.1f" % (selectbounds[i - 1]) + " < " + selecttitle[0] + \
                            " < " + "%.1f" % (selectbounds[i])]
                else:
                    print(selecttitle[0])
                    info[1] = "%.0f" % (selectbounds[i - 1]) + " < " + selecttitle[0] + " < " + "%.0f" % (selectbounds[i])
                selectcond = np.where(np.logical_and(selectbounds[i - 1] < np.array(selectquantity),
                                      np.array(selectquantity) < selectbounds[i]))
                subsnap = snap[selectcond]
                print("Subsnap has " + str(len(subsnap["ID"])) + " particles.")
                if sigmapp > 0.:
                    subsnap = PseudoSnap(subsnap, npseudoparticles, sigmapp)
                grid = VoronoiGrid(subsnap, extent, npixel_per_side=npixel_per_side,
                                   partperbin=partperbin, nspaxels=nspaxels)
                plotquantity = grid.get_stats(qty, weightqty=weightqty,
                                              mode=statsmode,
                                              artificial_error=artificial_error,
                                              measerror=measerror,
                                              centeriszero=centeriszero)
                fluxquantity = weightqty  # snap[weightqty]
                makevoronoimap(plotquantity, grid, npanels=npanels,
                               fluxqty=fluxquantity, figureconfig=figureconfig,
                               cmap=cmap, cmaplimits=cmaplimits, titles=titles,
                               titles2=info, plotfile=plotfile + str(i),
                               savefigure=savefigure, cutatmag=cutatmag,
                               addlambdar=addlambdar, ncbarlabels=ncbarlabels,
                               savetxt=savetxt, scalebar=scalebar)
    if savefigure:
        plt.close()


def getdefaultplotparams(qty, statsmode=None, select=False, npanels=4):
    """
    For a given quantity, returns default subplot titles, colormap, FF
    colormap limit type (symmetric or not).
    """

    if qty == 'vel':
        titles = [r'$V_{avg} (\rm km/s)$', r'$\sigma (\rm km/s)$', r'$h_3$',
                  r'$h_4$']
        cmap = 'sauron'
        cmaplimits = ['symmetric', 'minmax', 'symmetric', 'symmetric']
    elif qty == 'age':
        if not select:
            titles = [r'$\rm Stellar \, age \, (\rm Gyr)$',
                      r'$\sigma_{\rm age} \, (\rm Gyr)$', r'$h_3$', r'$h_4$']
        else:
            titles = [r'$\rm age \, (\rm Gyr)$',
                      r'$\sigma_{\rm age} \, (\rm Gyr)$', r'$h_3$', r'$h_4$']
        cmaplimits = ['minmax', 'minmax', 'symmetric', 'symmetric']
        cmap = 'Spectral_r'
    elif qty == 'logage':
        titles = [r'$\rm log_{10} \, \rm age \, (\rm Gyr)$',
                  r'$\sigma_{\rmlog \, age} (\rm Gyr)$', r'$h_3$', r'$h_4$']
        cmaplimits = ['minmax', 'minmax', 'symmetric', 'symmetric']
        cmap = 'Spectral_r'
    elif qty == 'ZH':
        titles = [r'$\rm log(Z/Z_\odot)$', r'$\sigma_{Z}$', r'$h_3$', r'$h_4$']
        # titles=[r'$[Z/H]$', r'$\sigma_{[Z/H]}$', r'$h_3$', r'$h_4$']
        cmaplimits = ['minmax', 'minmax', 'symmetric', 'symmetric']
        cmap = 'viridis'
    elif qty == 'alphafe':
        titles = [r'$[\alpha/{\rm Fe}]$', r'$\sigma_{[\alpha/{\rm Fe}]}$',
                  r'$h_3$', r'$h_4$']
        cmaplimits = ['minmax', 'minmax', 'symmetric', 'symmetric']
        cmap = 'viridis'
    elif qty == 'OFe':
        titles = ['      ' + r'$[{\rm O}/{\rm Fe}]$',
                  r'$\sigma_{[{\rm O}/{\rm Fe}]}$',
                  r'$h_3$', r'$h_4$']
        cmaplimits = ['minmax', 'minmax', 'symmetric', 'symmetric']
        cmap = 'afmhot'
    elif qty == 'r':
        titles = [r'$r (\rm kpc)$', r'$\sigma_{\rm r} \, (\rm kpc)$', r'$h_3$',
                  r'$h_4$']
        cmaplimits = ['minmax', 'minmax', 'symmetric', 'symmetric']
        cmap = 'hot'
    elif qty == 'temp':
        titles = [r'$T\, (K)$', r'$\sigma_{\rm T} \, (K)$', r'$h_3$', r'$h_4$']
        cmaplimits = ['minmax', 'minmax', 'symmetric', 'symmetric']
        cmap = 'viridis'
    elif qty == 'lambdar':
        titles = [r'$\lambda_R$', r'$\sigma (km/s)$', r'$h_3$', r'$h_4$']
        cmaplimits = ['minmax', 'minmax', 'symmetric', 'symmetric']
        cmap = 'Spectral_r'
    elif qty == 'h3par':
        titles = [r'$K$', r'$\sigma (km/s)$', r'$h_3$', r'$h_4$']
        cmaplimits = ['minmax', 'minmax', 'symmetric', 'symmetric']
        cmap = 'bwr'
    else:
        titles = [r'$' + qty + '$', r'$\sigma_{' + qty + '}$', r'$h_3$', r'$h_4$']
        cmaplimits = ['minmax', 'minmax', 'symmetric', 'symmetric']
        cmap = 'viridis'

    if statsmode == '2gauss':
        if qty in ['vel']:
            cmaplimits = ['symmetric', 'minmax', 'symmetric', 'minmax', [0., 1.]]
        else:
            cmaplimits = ['minmax', 'minmax', 'minmax', 'minmax', [0., 1.]]
        titles = [r'$\mu_1$', r'$\sigma_1$', r'$\mu_2$', r'$\sigma_2$',
                  'fraction']

    if npanels > len(cmaplimits):
        cmaplimits = cmaplimits + ["minmax"] * (npanels - len(cmaplimits))

    return titles, cmap, cmaplimits


def makevoronoimap(plotquantity, grid, npanels=4, fluxcontours='smooth',
                   fluxqty='mass', cmap='sauron',
                   cmaplimits=["deduce", "minmax", "symmetric", "symmetric"],
                   figsize=5, figureconfig='horizontal', ncbarlabels=5,
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
    ncbarlabels     : Number of colorbar labels. The code will automatically calculate 
                      the best ncbarlabels labels for the plot.
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
    maplimits = [-grid.extent / 2, grid.extent / 2, -grid.extent / 2, grid.extent / 2]
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
                plt.ylabel(titles3, fontsize=textsize)  # , labelpad=-20)
            else:
                plt.gca().tick_params(labelleft='off')  # left
        if figureconfig == 'vertical':
            plt.subplot(int(npanels * 100 + 10 + (i + 1)))
            plt.ylabel(titles2[i], fontsize=textsize, labelpad=-20)
            if i == npanels - 1:
                plt.xlabel(titles3, fontsize=textsize)
        if figureconfig == '22':
            plt.subplot(220 + (i + 1))
            if titles2 == [" ", " ", " ", " "]:
                if i == 0 or i == 2:
                    plt.ylabel(r"$\rm kpc$", fontsize=textsize, labelpad=-20)
            else:
                plt.ylabel(titles2[i], fontsize=textsize, labelpad=-20)
            if i == npanels - 1 or i == npanels - 2:
                plt.xlabel(titles3, fontsize=textsize)
        if i == 0:
            minx = -0.5 * grid.extent  # np.min(xvor)
            miny = -0.5 * grid.extent  # np.min(yvor)
            xlength = grid.extent  # np.max(xvor)-np.min(xvor)
            ylength = grid.extent  # np.max(yvor)-np.min(yvor)
            # Custom titles in left side of first panel
            if len(titles2) > 0:  # above and below
                plt.text(minx + 0.03 * xlength, miny + 0.03 * ylength,
                         titles2[0], fontsize=labelsize, color=textcolor)
                plt.text(minx + 0.03 * xlength, miny + 0.89 * ylength,
                         titles2[1], fontsize=labelsize, color=textcolor)
            else:  # just below
                plt.text(minx + 0.03 * xlength, miny + 0.03 * ylength,
                         titles2, fontsize=labelsize, color=textcolor)

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
                plt.text(minx + 0.6 * xlength,  # 0.53
                         miny + 0.03 * ylength,
                         r"$\lambda_R = %.2f$" % (lambdaR), fontsize=labelsize,
                         color=textcolor)

        # Actually plotting the colors
        img = display_bins(grid.xvor, grid.yvor, grid.binNum, plotquantity[:, i], vmin=cmaplimits[i][0],
                           vmax=cmaplimits[i][1], cmap=cmap, **kwargs)

        ax = plt.gca()
        if cutatmag is None:
            # Plotting titles
            plt.title(titles[i], fontsize=titlesize, y=1.04)

            # Flux contours
            if fluxcontours is not None:
                try:
                    mag = -2.5 * np.log10(flux / np.max(flux).ravel())
                    plt.tricontour(xflux, yflux, mag,
                                   levels=np.arange(5), colors='k',  # 20
                                   linewidths=contourthickness)
                except:
                    print("Warning: Impossible to print contours; problem " \
                          + "with the data")
                    print(xflux, yflux, flux)

            # Adjusting axes ticks
            ax.tick_params(labelsize=digitsize)
            ax.tick_params(direction="in", which='both')
            # locs= ax.get_yticks()
            locs = np.round(np.linspace(-0.5 * grid.extent, 0.5 * grid.extent, naxeslabels), 2)
            ax.xaxis.set_ticks(locs)
            ax.yaxis.set_ticks(locs)
            plt.xlim([-0.5 * grid.extent, 0.5 * grid.extent])
            plt.ylim([-0.5 * grid.extent, 0.5 * grid.extent])

            # Determining and plotting colorbar ticks
            divider = mplax.make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            if ncbarlabels is None:
                cb = plt.colorbar(img, cax=cax)
                cb.ax.tick_params(labelsize=digitsize)
            else:  # Calculating optimal colormap tick distribution
                cspan = (cmaplimits[i][1] - cmaplimits[i][0]) / (ncbarlabels - 1)
                # cspan: separation between colorbar ticks/labels
                cspanorder = int(np.round(np.log10(cspan)))
                # recalculating cspan in case lower limit needs to be truncated
                if cmaplimits[i][1] != -cmaplimits[i][0]:
                    if cmaplimits[i][0] == 0.:
                        cmaplimits[i][0] = np.min(plotquantity[:, i])
                    # limitorder=int(np.floor(np.log10(abs(cmaplimits[i][0]))))
                    limitorder = int(np.floor(np.log10(abs(cmaplimits[i][1]))))
                    lowerlimit = (10. ** (limitorder - 1)) * np.ceil(cmaplimits[i][0] / \
                                                                     (10. ** (limitorder - 1)))
                    cspan = (cmaplimits[i][1] - lowerlimit) / (ncbarlabels - 1)
                    # truncating cspan to avoid lots of decimals
                cspan = (10. ** (cspanorder - 1)) * np.trunc(cspan / \
                                                             (10. ** (cspanorder - 1)))
                if cspanorder > 0:
                    cspan = int(cspan)
                if cmaplimits[i][1] == -cmaplimits[i][0]:
                    cticks = []
                    for itick in np.arange(ncbarlabels):
                        cticks.append(cspan * (itick - ((ncbarlabels - 1) / 2.)))
                else:
                    cticks = []
                    for itick in np.arange(ncbarlabels):
                        cticks.append(lowerlimit + cspan * (itick))
                print("Colorbar ticks (" + str(i) + "): " + str(cticks))
                cb = plt.colorbar(img, cax=cax, ticks=cticks)
                cb.ax.tick_params(labelsize=digitsize * 1.1)

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
        np.savetxt(savetxt + '.kin', np.transpose(np.array([plotquantity[:, 0],
                                                            plotquantity[:, 1],
                                                            plotquantity[:, 2],
                                                            plotquantity[:, 3],
                                                            fluxvoro, grid.xBar, grid.yBar])))
        np.savetxt(savetxt + '.grid', np.transpose(np.array([grid.xvor,
                                                             grid.yvor,
                                                             grid.binNum])))


def lambdar(snap, grid=None, qty='vel', weightqty='mass', extent=20, nbin=100,
            pseudoparticles=True, snaporientation=[1, 0, 0], npixel_per_side=200,
            partperbin=None, nspaxels=500, statsmode='sample', forceedgeon=True):
    """
    Calculates radial lambdar profile for galaxy in the given snapshot (needs 
    to be already centered). Returns nbin values of lambdar at different radii 
    between 0 and extent/2. Other parameters are the same as the ones in the 
    other Voronoi binning functions.
    """
    if forceedgeon:
        snap = orient_snap(snap, snaporientation=[1, 0, 0])
    if pseudoparticles == True:
        snap = PseudoSnap(snap)
    if grid == None:
        grid = VoronoiGrid(snap, extent, npixel_per_side=npixel_per_side,
                           partperbin=partperbin, nspaxels=nspaxels)
    avgbin, sigmabin, h3bin, h4bin = grid.get_stats(qty, mode=statsmode)

    weights = snap[weightqty]
    flux = np.zeros(len(set(grid.binNum)))
    for bn in set(grid.binNum):
        # Mass of each spaxel divided by its area
        flux[bn] = np.sum(weights[grid._spaxelofpart == bn]) / (len(grid.binNum[grid.binNum == bn]))

    lambdaR = _lambdar(grid, np.array([avgbin, sigmabin, h3bin, h4bin]),
                       weights=flux, nbin=nbin, quiet=False)
    return lambdaR


def _lambdar(grid, stats, weights=None, nbin=100, quiet=False, rmax=-1,
             **kwargs):
    """
    Computes lambdar for a given set voronoi grid (grid) with already computed 
    average and dispersion per each bin (stats) and given weights (flux).
    """
    xBar = grid.xBar
    yBar = grid.yBar
    if np.shape(weights) == ():
        # checking if weights==None; avoids bug when not using pseudosnap
        weights = np.full(np.shape(grid.xBar), 1.)
    vavg = stats[:, 0]
    sigma = stats[:, 1]
    h3 = stats[:, 2]
    h4 = stats[:, 3]

    def lambdarvalue(xBar, yBar, vavg, sigma, weights, rad=None):
        vrms = np.sqrt(vavg ** 2 + sigma ** 2)
        c1 = weights * np.sqrt(xBar ** 2 + yBar ** 2) * np.abs(vavg)
        c2 = weights * np.sqrt(xBar ** 2 + yBar ** 2) * vrms
        c1 = c1[np.sqrt(xBar ** 2 + yBar ** 2) < rad]
        c2 = c2[np.sqrt(xBar ** 2 + yBar ** 2) < rad]
        lambdaR = np.sum(c1) / np.sum(c2)
        return lambdaR

    lambdaR = []
    if rmax < 0.:
        rmax = np.max(np.sqrt(xBar ** 2 + yBar ** 2))
    step = float(rmax) / float(nbin)
    xaxis = np.linspace(step, rmax, nbin)
    for r in xaxis:
        lambdaR.append(lambdarvalue(xBar, yBar, vavg, sigma, weights, rad=r))
    if not quiet:
        print("LambdaR value at " + str(rmax) + \
              " kpc:" + str(lambdaR[-1]))
    return lambdaR


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

################################################################################
# Following routines are from the display_pixels package of Michele Cappellari, #
# slightly edited by MF to allow for personalized color maps.                   #
# See http://www-astro.physics.ox.ac.uk/~mxc/software/                          #
################################################################################

def display_bins(x, y, binNum, qtyBin, ax=None, **kwargs):
    if not (x.size == y.size == binNum.size):
        raise ValueError('The vectors (x, y, binNum) must have the same size')

    if np.unique(binNum).size != qtyBin.size:
        raise ValueError('qtyBin size does not match number of bins')

    if np.unique(binNum).size > 1:
        val = qtyBin[binNum]
    else:
        val = np.full(np.shape(x), qtyBin)
    img = _display_pixels(x, y, val, ax=ax, **kwargs)

    return img


def _display_pixels(x, y, val, pixelsize=None, angle=None, cmap='sauron',
                    ax=None, **kwargs):
    """
    Display vectors of square pixels at coordinates (x,y) coloured with "val".
    An optional rotation around the origin can be applied to the whole image.
    
    The pixels are assumed to be taken from a regular cartesian grid with 
    constant spacing (like an image), but not all elements of the grid are 
    required (missing data are OK).

    This routine is designed to be fast even with large images and to produce
    minimal file sizes when the output is saved in a vector format like PDF.

    """

    from scipy.spatial import distance
    if pixelsize is None:
        pixelsize = np.min(distance.pdist(np.column_stack([x, y])))

    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    nx = int(round((xmax - xmin) / pixelsize) + 1)
    ny = int(round((ymax - ymin) / pixelsize) + 1)
    j = np.round((x - xmin) / pixelsize).astype(int)
    k = np.round((y - ymin) / pixelsize).astype(int)
    # print(nx, ny)
    mask = np.ones((nx, ny), dtype=bool)
    img = np.empty((nx, ny))
    mask[j, k] = 0
    img[j, k] = val
    img = np.ma.masked_array(img, mask)

    if ax == None:
        ax = plt.gca()
    plotextent = 1.

    if cmap == 'sauron':
        cmap = sauron

    if (angle is None) or (angle == 0):

        img = ax.imshow(np.rot90(img), interpolation='none', cmap=cmap,
                        # cmap=kwargs.get("cmap", sauron),
                        extent=[plotextent * (xmin - pixelsize / 2),
                                plotextent * (xmax + pixelsize / 2),
                                plotextent * (ymin - pixelsize / 2),
                                plotextent * (ymax + pixelsize / 2)],
                        **kwargs)

    else:

        x, y = np.ogrid[xmin - pixelsize / 2: xmax + pixelsize / 2: (nx + 1) * 1j,
               ymin - pixelsize / 2: ymax + pixelsize / 2: (ny + 1) * 1j]
        ang = np.radians(angle)
        x, y = x * np.cos(ang) - y * np.sin(ang), x * np.sin(ang) + y * np.cos(ang)

        mask1 = np.ones_like(x, dtype=bool)
        mask1[:-1, :-1] *= mask  # Flag the four corners of the mesh
        mask1[:-1, 1:] *= mask
        mask1[1:, :-1] *= mask
        mask1[1:, 1:] *= mask
        x = np.ma.masked_array(x, mask1)  # Mask is used for proper plot range
        y = np.ma.masked_array(y, mask1)

        img = ax.pcolormesh(x, y, img, cmap=cmap,  # cmap=kwargs.get("cmap", sauron),
                            edgecolors="face", **kwargs)
        ax.axis('image')

    ax.minorticks_on()
    ax.tick_params(length=10, width=1, which='major')
    ax.tick_params(length=5, width=1, which='minor')
    ax.tick_params(top=True, right=True, direction='in', which='both')

    return img


##############################################################################

# V1.0: SAURON colormap by Michele Cappellari & Eric Emsellem, Leiden, 10 July 2001
#
# Start with these 7 equally spaced coordinates, then add 4 additional points
# x = findgen(7)*255/6. + 1
# 1.0  43.5  86.0  128.5  171.0  213.5  256.0
#
# x = [1.0, 43.5, 86.0, 86.0+20, 128.5-10, 128.5, 128.5+10, 171.0-20, 171.0, 213.5, 256.0]
# red =   [0.0, 0.0, 0.4,  0.5, 0.3, 0.0, 0.7, 1.0, 1.0,  1.0, 0.9]
# green = [0.0, 0.0, 0.85, 1.0, 1.0, 0.9, 1.0, 1.0, 0.85, 0.0, 0.9]
# blue =  [0.0, 1.0, 1.0,  1.0, 0.7, 0.0, 0.0, 0.0, 0.0,  0.0, 0.9]

_cdict = {'red': [(0.000, 0.01, 0.01),
                  (0.170, 0.0, 0.0),
                  (0.336, 0.4, 0.4),
                  (0.414, 0.5, 0.5),
                  (0.463, 0.3, 0.3),
                  (0.502, 0.0, 0.0),
                  (0.541, 0.7, 0.7),
                  (0.590, 1.0, 1.0),
                  (0.668, 1.0, 1.0),
                  (0.834, 1.0, 1.0),
                  (1.000, 0.9, 0.9)],
          'green': [(0.000, 0.01, 0.01),
                    (0.170, 0.0, 0.0),
                    (0.336, 0.85, 0.85),
                    (0.414, 1.0, 1.0),
                    (0.463, 1.0, 1.0),
                    (0.502, 0.9, 0.9),
                    (0.541, 1.0, 1.0),
                    (0.590, 1.0, 1.0),
                    (0.668, 0.85, 0.85),
                    (0.834, 0.0, 0.0),
                    (1.000, 0.9, 0.9)],
          'blue': [(0.000, 0.01, 0.01),
                   (0.170, 1.0, 1.0),
                   (0.336, 1.0, 1.0),
                   (0.414, 1.0, 1.0),
                   (0.463, 0.7, 0.7),
                   (0.502, 0.0, 0.0),
                   (0.541, 0.0, 0.0),
                   (0.590, 0.0, 0.0),
                   (0.668, 0.0, 0.0),
                   (0.834, 0.0, 0.0),
                   (1.000, 0.9, 0.9)]
          }

_rdict = {'red': [(0.000, 0.9, 0.9),
                  (0.170, 1.0, 1.0),
                  (0.336, 1.0, 1.0),
                  (0.414, 1.0, 1.0),
                  (0.463, 0.7, 0.7),
                  (0.502, 0.0, 0.0),
                  (0.541, 0.3, 0.3),
                  (0.590, 0.5, 0.5),
                  (0.668, 0.4, 0.4),
                  (0.834, 0.0, 0.0),
                  (1.000, 0.01, 0.01)],
          'green': [(0.000, 0.9, 0.9),
                    (0.170, 0.0, 0.0),
                    (0.336, 0.85, 0.85),
                    (0.414, 1.0, 1.0),
                    (0.463, 1.0, 1.0),
                    (0.502, 0.9, 0.9),
                    (0.541, 1.0, 1.0),
                    (0.590, 1.0, 1.0),
                    (0.668, 0.85, 0.85),
                    (0.834, 0.0, 0.0),
                    (1.000, 0.01, 0.01)],
          'blue': [(0.000, 0.9, 0.9),
                   (0.170, 0.0, 0.0),
                   (0.336, 0.0, 0.0),
                   (0.414, 0.0, 0.0),
                   (0.463, 0.0, 0.0),
                   (0.502, 0.0, 0.0),
                   (0.541, 0.7, 0.7),
                   (0.590, 1.0, 1.0),
                   (0.668, 1.0, 1.0),
                   (0.834, 1.0, 1.0),
                   (1.000, 0.01, 0.01)]
          }

sauron = colors.LinearSegmentedColormap('sauron', _cdict)
sauron_r = colors.LinearSegmentedColormap('sauron_r', _rdict)
