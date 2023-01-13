__all__ = ["lambdar"]


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
