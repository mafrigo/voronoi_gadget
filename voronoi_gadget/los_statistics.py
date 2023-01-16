"""
by Matteo Frigo (Max Planck Institute for Astrophysics)
Last update: 14 May 2018

Module for calculating statistics out of an array of quantities.
Thought for line-of-sight kinematics but can be applied to anything.
Currently allows to fit distributions with a Gauss-Hermite formula or
with the sum of two Gaussians.
"""

__all__ = ["gauss_hermite_func", "normalized_gauss_hermite_func", "gauss_hermite_fit"]

import numpy as np
import scipy.integrate
import scipy.optimize as opt
import scipy.special


# from astropy.stats.funcs import _biweight_location
# from astropy.stats.funcs import _biweight_midvariance


def gauss_hermite_func(x, mu, sigma, h3, h4):
    """
    Returns the value of the Gauss-Hermite function in x:
    
    f(x)=e^-((x-mu)^2/(2*sigma^2))*(1+h3*H3(x)+h4*H4(x))
    
    This is NOT normalized so that the integral is 1 (it is only if h3==0 
    and h4==0), but is close to it.
    """
    wpar = (x - mu) / sigma
    func = np.exp(-0.5 * wpar ** 2) / np.sqrt(2 * np.pi * sigma ** 2)
    H3 = (1. / np.sqrt(3.)) * (2. * wpar ** 3 - 3. * wpar)
    H4 = (1. / np.sqrt(24.)) * (4. * wpar ** 4 - 12. * wpar ** 2 + 3.)
    func = func * (1. + h3 * H3 + h4 * H4)
    func = np.clip(func, 10 ** -10, None)
    return func


def normalized_gauss_hermite_func(x, mu, sigma, h3, h4):
    """
    Same as gauss_hermite_func, but normalized so that the integral from 
    -inf to +inf is 1. Takes much longer.
    """
    ghintegral = scipy.integrate.quad(gauss_hermite_func, -np.inf, np.inf, args=(mu, sigma, h3, h4))[0]
    func = gauss_hermite_func(x, mu, sigma, h3, h4) / ghintegral
    return func


def gauss_hermite_fit(quantity, weights=None, mode='fit', quiet=True, nbins=None):
    """
    Calculates best-fitting Gauss-Hermite parameters (average, dispersion, h3, 
    h4) of an array of real numbers (quantity) with optional weights (weights). 
    
    Input parameters:
    
    quantity        : Array of values of which you want to calculate the 
                      Gauss-Hermite fit. 
    weights         : Weights of "quantity" (must be in the same shape).
    mode            : Determines how to calculate the best-fitting parameters. 
                      Can be:
        - "sample"    : Sample average and dispersion. h3 and h4 come from fitting 
                        the quantity histogram.
        - "biweight"  : Biweight average and dispersion; unweighted. h3 and h4 
                        come from fitting the quantity histogram.
        - "fit"       : Average, dispersion, h3 and h4 come from directly 
                        fitting the histogram of quantity. Recommended.
        - "likelihood": Maximum likelihood estimation of the parameters assuming 
                        a gaussianly distributed measurement error. Takes much 
                        longer and doesn't converve as well as "fit".
    quiet           : If False, prints the fit results.
    nbins           : Number of bins for building the histogram of the "quantity"
                      array which then actually gets fit (when mode=='fit'). 
                      If None, the optimal number of bins is calculated using the 
                      Freedman-Diaconis rule.
    artificial_error: If > 0, displaces the values of quantity gaussianly with 
                      sigma artificial_error.
    meas_error      : The measurement error for the values of quantity. Only used
                      when mode=='likelihood'.
    dispersion_fix  : If dispersion_fix==True and meas_error > 0, meas_error gets
                      subtracted from the best fit velocity dispersion.
    skip_normaliz   : Makes the code a bit faster by skipping the normalization
                      of the Gauss-Hermite function. Should not be used with
                      mode=='likelihood'.
                      
    Output parameters:
    
    avg  : Mean parameter of the Gauss-Hermite distribution (as in the mu parameter
           in gauss_hermite_func, not the actual mean of the distribution).
    disp : Standard deviation of the GH distribution (as in the sigma parameter
           in gauss_hermite_func, not the actual dispersion of the distribution).
    h3   : h3 parameter (skewness) of the GH distribution.
    h4   : h4 parameter (kurtosis) of the GH distribution.
    """
    if np.shape(weights) == ():
        weights = np.full(np.shape(quantity), 1.)
    quantity = np.array(quantity)
    weights = np.array(weights)

    if mode != 'biweight':
        # Calculating sample average
        avg = np.average(quantity, weights=weights)
        disp = np.sqrt(np.average((quantity - avg) ** 2, weights=weights))
    else:
        # Calculating biweight average
        avg = _biweight_location(quantity)
        disp = _biweight_midvariance(quantity)

    if nbins is None:
        # Determining number of bins according to the Freedman-Diaconis rule
        binsize = 2. * (2. * 0.6745 * disp) / (float(len(quantity))) ** (1. / 3.)
        if disp != 0.:
            nbins = int(8. * disp / binsize)
        else:
            print("Warning: trying to fit only " + str(len(quantity)) + " particle(s)")
            nbins = 1
        if nbins < 10:
            print("Warning: too few particles to make a proper histogram.")
            nbins = 10

    # Making a histogram of the quantity and fitting it to Gauss Hermite series
    fitavg, fitdisp, h3, h4 = _histogram_ghfit(quantity, weights, initialguess=[avg, disp, 0., 0.], nbins=nbins)
    if mode == 'fit':
        avg, disp = fitavg, fitdisp

    if not quiet:
        print("%f,\t %f,\t %f,\t %f" % (avg, disp, h3, h4))
    return avg, disp, h3, h4


def _histogram_ghfit(quantity, weights, initialguess=None, nbins=50, returnerr=False,
                     skip_normaliz=True):
    """
    Performs a histogram fit of the distribution in quantity with a 
    Gauss-Hermite series.
    """
    if initialguess is None:
        initialguess = [0., 0., 0., 0.]
    try:
        velhist, histxaxis = np.histogram(quantity, bins=nbins, weights=weights, density=True)
        histxaxis = histxaxis + (histxaxis[1] - histxaxis[0]) / 2.
        histxaxis = histxaxis[:-1]
        if not skip_normaliz:
            popt, pcov = opt.curve_fit(normalized_gauss_hermite_func, histxaxis, velhist, p0=initialguess)
        else:
            popt, pcov = opt.curve_fit(gauss_hermite_func, histxaxis, velhist, p0=initialguess)
        avg, disp, h3, h4 = popt[0], popt[1], popt[2], popt[3]
        avgerr, disperr, h3err, h4err = np.sqrt(np.diag(pcov))
    except (RuntimeError, TypeError, ValueError):
        avg, disp, h3, h4 = initialguess
        avgerr, disperr, h3err, h4err = 0., 0., 0., 0.
        print("Warning: Failure in fitting the spaxel distribution with a GH series.")
        print("Number of particles in failed vorobin: " + str(len(quantity)))
    if returnerr:
        return avg, disp, h3, h4, avgerr, disperr, h3err, h4err
    else:
        return avg, disp, h3, h4
