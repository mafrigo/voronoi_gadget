import numpy as np
from voronoi_gadget.los_statistics import *


class VoronoiGrid(object):
    """
    Creates a grid of Voronoi spaxels in which each spaxel has approximately
    the same S/N ratio (= square root of number of particles).

    Usage:
     snap            : Snapshot to use for creating the grid. Either a real
                       pygad snapshot or one generated by PositionDisplacements.
     extent          : Extent of the figure (in same the units as the snapshot).
     weightqty       : Quantity to use as weight for computing the signal/noise
                       of each bin. Default is "None", meaning that the Voronoi
                       tessellation is determined by the number density of
                       particles,  regardless of their mass/luminosity/etc.
     npixel_per_side : npixel_per_side**2 is the number of regular pixels which
                       form the base grid. The voronoi spaxels are constructed
                       by associating these.
     nspaxels        : If partperbin is None, number of voronoi spaxels to be
                       generated. Otherwise unused. If nspaxel==0, no Voronoi
                       binning is performed, and the regular 2D grid is
                       returned in the same format.
     partperbin      : Target number of particles in each voronoi spaxel.
                       If None, nspaxels spaxels are generated.

    Returns a VoronoiGrid object, to be used in voronoistats() and
    makevoronoimap().
    """

    def __init__(self, snap, extent, weightqty=None, npixel_per_side=200,
                 nspaxels=500, partperbin=None):

        if nspaxels == 0:
            voronoibinning = False
        else:
            voronoibinning = True
        posx = snap["pos"][:, 0]
        posy = snap["pos"][:, 1]
        if weightqty is None:
            weights = np.full(np.shape(posx), 1.)
        else:
            weights = snap[weightqty]
            weights = weights * len(posx) / np.sum(weights)
            # Weighted number of particles

        spaxelofpart, xvor, yvor, xBar, yBar, binNum = _makegrid(posx, posy,
                                                                 weights, extent, voronoibinning=voronoibinning,
                                                                 npixel_per_side=npixel_per_side,
                                                                 partperbin=partperbin, nvoronoibins=nspaxels)
        self._spaxelofpart = spaxelofpart
        self._snap = snap
        self.xBar = xBar
        self.yBar = yBar
        self.xvor = xvor
        self.yvor = yvor
        self.binNum = binNum
        self.extent = extent
        self.npixel_per_side = npixel_per_side
        self.nspaxels = np.max(binNum) + 1

    def qty_in_spaxel(self, qtylabel, ispaxel, qtyarray=None):
        if qtyarray is None:
            qtyarray = self._snap[qtylabel]
        qtyarray = qtyarray[self._spaxelofpart == ispaxel]
        return qtyarray

    def qty_by_spaxel(self, qtylabel):
        qty = self._snap[qtylabel]
        return np.array([self.qty_in_spaxel(qtylabel, ispaxel, qtyarray=qty)
                         for ispaxel in self.binNum])

    def get_stats(self, qty, weightqty='mass', mode='sample',
                     artificial_error=0., measerror=0., quiet=True,
                     centeriszero=False):
        """
        Calculates average, dispersion, h3 and h4 of quantity "qty" for all
        the spaxels of a given voronoigrid.
        mode can be 'sample', 'biweight', 'fit', 'likelihood', '2gauss'.
        """

        if weightqty is not None:
            weights = self._snap[weightqty]
        else:
            weights = np.full(np.shape(qty), 1.)
        if qty not in ["ZH", "alphafe", "logage"]:
            quantity = self._snap[qty]
        else:
            if qty == 'ZH':
                # Z/H in solar units (log); solar values from Asplund,Grevesse,Sauval 2006
                quantity = np.log10(np.clip(self._snap["metals"] / self._snap["H"], 10 ** -8, 1.) / 0.0165)
            if qty == 'alphafe':
                quantity = np.log10(np.clip((self._snap["alpha_el"] + 10 ** -20) / self._snap["Fe"], 10 ** -8, 10.))
            if qty == 'logage':
                quantity = np.log10(self._snap["age"] * 10. ** 9)  # log(age)
        if len(np.shape(quantity)) == 2:
            quantity = quantity[:, 1]  # line-of-sight

        print("Calculating statistics of each bin...")
        stats = np.full((self.nspaxels, 5), 0.)
        if mode == 'likelihood':
            quiet = False  # To have something to see during the calculations!
        if not quiet and mode != '2gauss':
            print("Result of Gauss-Hermite fit:")
            print("Average,            sigma,       h3,          h4:")

        for bn in set(self.binNum):
            qtybin = quantity[self._spaxelofpart == bn]
            weightsbin = weights[self._spaxelofpart == bn]
            if len(qtybin) == 0:
                print("WARNING: No particles in current voronoi bin.")
            else:
                if len(qtybin) < 3:
                    print("Few particles in voronoi bin ( " + str(len(qtybin)) + " )")
                if mode in ["sample", "biweight", "fit", "likelihood"]:
                    stats[bn][:4] = gauss_hermite_fit(qtybin, weights=weightsbin, mode=mode,
                                                      artificial_error=artificial_error, meas_error=measerror)
                if mode == '2gauss':
                    stats[bn] = double_gaussian_fit(qtybin, weights=weightsbin, artificial_error=artificial_error)
            if not quiet:
                print(stats[bn])

        if centeriszero:
            stats[:, 0] = stats[:, 0] - np.mean(stats[:, 0][self.xBar ** 2 + self.yBar ** 2 < 1.])
        return stats


def _makegrid(posx, posy, weights, extent, voronoibinning=True,
              npixel_per_side=200, partperbin=None, nvoronoibins=500):
    import scipy.stats as stats
    print("Binning particles on regular grid... ")
    nimg, xedges, yedges, pixelofpart0 = stats.binned_statistic_2d(x=posx,
                                                                   y=posy, values=weights, statistic='sum',
                                                                   bins=npixel_per_side,
                                                                   range=[[-extent / 2., extent / 2.],
                                                                          [-extent / 2., extent / 2.]])
    pixelsize = float(extent) / npixel_per_side
    x = xedges[:-1] + 0.5 * pixelsize
    y = yedges[:-1] + 0.5 * pixelsize
    xvor, yvor = np.meshgrid(x, y)
    xvor = xvor.reshape((npixel_per_side ** 2))
    yvor = yvor.reshape((npixel_per_side ** 2))

    if voronoibinning:
        from vorbin.voronoi_2d_binning import voronoi_2d_binning
        print("Preparing voronoi binning...")
        if partperbin is None:
            partperbin = np.sum(nimg) / nvoronoibins
        else:
            partperbin *= 0.7  # 0.7 is an empirical factor to get the right output partperbin
        if partperbin > np.sum(nimg):
            print("Warning: too few particles for the selected value " + \
                  "of partperbin; making only one spaxel")
            binNum = np.full(np.shape(xvor), 0).astype(int)
            xBar = np.array([0.])
            yBar = np.array([0.])
        else:
            # (Poissonian) signal-to-noise used to determine the Voronoi grid
            signal_to_noise = np.sqrt(nimg.reshape(npixel_per_side ** 2))
            targetSN = np.sqrt(partperbin)  # target signal-to-noise of every bin

            print("Voronoi binning...")
            binNum, xNode, yNode, xBar, yBar, sn, nvorpixels, scale = \
                voronoi_2d_binning(xvor, yvor, signal_to_noise,
                                   np.full(np.shape(signal_to_noise), 1.),
                                   targetSN, pixelsize=pixelsize, plot=False, quiet=True)
    else:
        binNum = np.arange(len(xvor))
        xBar = xvor
        yBar = yvor
    yind, xind = np.unravel_index(pixelofpart0, (len(xedges) + 1, len(yedges) + 1))
    gridcondition = (xind > 0) * (xind < npixel_per_side + 1) * (yind > 0) * \
                    (yind < npixel_per_side + 1)
    # gridcondition excludes the "pixels" outside of the grid
    pixelofpart = ((xind - 1) + (yind - 1) * npixel_per_side)[gridcondition]
    spaxelofpart = np.full(posx.size, -1)
    spaxelofpart[gridcondition] = binNum[pixelofpart]

    return spaxelofpart, xvor, yvor, xBar, yBar, binNum



