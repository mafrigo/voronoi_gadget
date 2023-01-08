"""
by Matteo Frigo (Max Planck Institute for Astrophysics)
Last update: 14 May 2018

Module for calculating statistics out of an array of quantities.
Thought for line-of-sight kinematics but can be applied to anything.
Currently allows to fit distributions with a Gauss-Hermite formula or
with the sum of two Gaussians.
"""

__all__ = ["gauss_hermite_func", "normalized_gauss_hermite_func", "gauss_hermite_fit",
           "double_gaussian_fit"]

import numpy as np
import pygad
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


def gauss_hermite_fit(quantity, weights=None, mode='fit', quiet=True,
                      nbins=None, artificial_error=0., meas_error=0.,
                      dispersion_fix=True, skip_normaliz=True):
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

    # Adding artificial error displacement to the variable
    if artificial_error != 0.:
        quantity = quantity + np.random.normal(0., artificial_error, len(quantity))
        # gaussian random error (sigma=addederror) 

    # Assigning measurement error
    try:
        if isinstance(meas_error, pygad.UnitArr):
            meas_error.convert_to(quantity.units)
            meas_error = float(meas_error)
    except(IOError, ModuleNotFoundError, AttributeError): #pygad missing - could not process measerror
        meas_error = float(meas_error)
    error = np.full(np.shape(quantity), meas_error)
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

    if mode != 'likelihood':
        # Making a histogram of the quantity and fitting it to Gauss Hermite series
        fitavg, fitdisp, h3, h4 = _histogram_ghfit(quantity, weights,
                                                   initialguess=[avg, disp, 0., 0.],
                                                   nbins=nbins, skip_normaliz=skip_normaliz)
        if mode == 'fit':
            avg, disp = fitavg, fitdisp
        if dispersion_fix:  # subtracting measurement error from the velocity dispersion
            disp = np.sqrt(disp ** 2 - meas_error ** 2)
    else:
        avg, disp, h3, h4 = _maximum_likelihood_ghfit(quantity, weights, error,
                                                      initialguess=[avg, disp, 0., 0.])

    if not quiet:
        print("%f,\t %f,\t %f,\t %f" % (avg, disp, h3, h4))
    return avg, disp, h3, h4


def _histogram_ghfit(quantity, weights, initialguess=None, nbins=50, returnerr=False,
                     skip_normaliz=True):
    """
    Performs a histogram fit of the distribution in quantity with a 
    Gauss-Hermite series.
    """
    if initialguess == None:
        initialguess = [0., 0., 0., 0.]
    try:
        velhist, histxaxis = np.histogram(quantity, bins=nbins, weights=weights,
                                          density=True)
        histxaxis = histxaxis + (histxaxis[1] - histxaxis[0]) / 2.
        histxaxis = histxaxis[:-1]
        if not skip_normaliz:
            popt, pcov = opt.curve_fit(normalized_gauss_hermite_func, histxaxis,
                                       velhist, p0=initialguess)
        else:
            popt, pcov = opt.curve_fit(gauss_hermite_func, histxaxis,
                                       velhist, p0=initialguess)
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


def _maximum_likelihood_ghfit(quantity, weights, error, initialguess=None,
                              bias_correction=True):
    """
    Performs maximum likelihood fit of the distribution in quantity with a 
    Gauss-Hermite series.
    """
    if initialguess == None:
        initialguess = [0., 0., 0., 0.]
    avg, disp, h3, h4 = initialguess

    result = opt.minimize(_likelihood, x0=[avg, disp, 0., 0.],
                          args=[quantity, weights, error, analyticjacobian],
                          jac=analyticjacobian,
                          bounds=((avg - disp, avg + disp), (0.5 * disp, 2. * disp), (-0.5, 0.5), (-0.5, 0.5)))
    # , tol=0.0000000001)
    avg, disp, h3, h4 = result.x[0], result.x[1], result.x[2], result.x[3]

    if bias_correction:  # see Kenney & Keeping 1951, p. 171
        b = np.sqrt(2. / len(quantity)) * np.exp(scipy.special.gammaln(len(quantity) / 2.) \
                                                 - scipy.special.gammaln((len(quantity) - 1.) / 2.))
        corrinsdisp = (1. - b ** 2) * np.mean(error) ** 2
        disp = np.sqrt(disp ** 2 + corrinsdisp) / b

    return avg, disp, h3, h4


def _likelihood(par, args):
    """
    Analytical solution for the integral of the convolution of the quantity 
    distribution (assumed to be gaussian-hermite) with a measurement error 
    distribution. See appendix A in van de Ven et al. 2006.
    """

    avg, disp, h3, h4 = par[0], par[1], par[2], par[3]
    quantity, weights, error, analyticjacobian = args
    disperr = np.sqrt(error ** 2 + disp ** 2)
    wi = (quantity - avg) / disperr
    diff = error ** 2 - disp ** 2
    rawllh = np.full(len(quantity), 1.)
    # 3rd moment
    rawllh = rawllh + h3 * (2. * (disp * wi) ** 3 + 3. * disp * wi * diff) / (np.sqrt(3.) * disperr ** 3)
    # 4th moment
    rawllh = rawllh + h4 * (4. * (disp * wi) ** 4 + 12. * diff * (disp * wi) ** 2. + 3. * diff ** 2) \
             / (2. * np.sqrt(6.) * disperr ** 4)
    # integral of gh convolution
    rawllh = rawllh * gauss_hermite_func(np.array(quantity), avg, disperr, 0., 0.)
    rawllh = np.clip(rawllh, 10 ** -10, None)
    if not skipnormaliz:
        # integral of GH series for normalizing
        ghintegral = scipy.integrate.quad(gauss_hermite_func,
                                          -np.inf, np.inf, args=(avg, disp, h3, h4))[0]
    else:
        ghintegral = 1.
    # weighted likelihood
    likelihood = -2. * np.sum(weights * np.log(rawllh / ghintegral)) / np.sum(weights)
    return likelihood


def double_gaussian_fit(quantity, weights=None, nbins=None, artificial_error=0.):
    """
    Fits an array with the sum of two gaussians with different 
    averages/dispersions and normalizations.
    
    Parameters:
    
    quantity        : Array of values of which you want to calculate the 
                      Gauss-Hermite fit. 
    weights         : Weights of "quantity" (must be in the same shape).
    nbins           : Number of bins for building the histogram of the "quantity"
                      array which then actually gets fit. If None, the optimal 
                      number of bins is calculated using the Freedman-Diaconis
                      rule.
    artificial_error: If > 0, displaces the values of quantity gaussianly with 
                      sigma artificial_error.
                      
    Output:
    
    avg1  : Mean of the first gaussian.
    disp1 : Standard deviation of the first gaussian.
    avg2  : Mean of the second gaussian.
    disp2 : Standard deviation of the second gaussian.
    frac  : Normalization of the second gaussian divided by normalization of
            the first one (between 0. and 1.).
    """
    if weights == None or np.shape(weights) == ():
        weights = np.full(np.shape(quantity), 1.)

    if artificial_error != 0.:
        error = np.random.normal(0., artificial_error, len(quantity))
    else:
        error = np.zeros(len(quantity))
    quantity = quantity + error

    #   sample average
    avg = np.average(quantity, weights=weights)
    disp = np.sqrt(np.average((quantity - avg) ** 2, weights=weights))

    if nbins == None:
        try:
            binsize = np.amax([25. * np.sqrt(2. * np.pi) * disp / len(quantity),
                               meas_error])
            nbins = int(4. * disp / binsize)
        except:
            print("Warning: Couldn't calculate optimal bin size, " \
                  + "something is fishy! Using nbins=50")
            nbins = 50

    def double_gaussian(x, mu, sigma, mu2, sigma2, frac):
        wpar = (x - mu) / sigma
        wpar2 = (x - mu2) / sigma2
        cost = 1. / ((1. - frac) * np.sqrt(2 * np.pi * sigma ** 2) + frac * np.sqrt(2 * np.pi * sigma2 ** 2))
        gauss = cost * ((1. - frac) * np.exp(-0.5 * wpar ** 2) + frac * np.exp(-0.5 * wpar2 ** 2))
        gauss = np.clip(gauss, 10 ** -10, None)
        if frac <= 0 or sigma < 0. or sigma2 < 0.:
            gauss = gauss * 10. ** 20
        return gauss

    velhist, histxaxis = np.histogram(quantity, bins=nbins, weights=weights,
                                      normed=True)
    histxaxis = histxaxis + (histxaxis[1] - histxaxis[0]) / 2.
    histxaxis = histxaxis[:-1]
    try:
        popt, pcov = opt.curve_fit(double_gaussian, histxaxis, velhist,
                                   p0=[np.sign(avg) * (abs(avg) - 0.3 * disp),
                                       0.8 * disp, np.sign(avg) * (abs(avg) + 0.3 * disp),
                                       0.8 * disp, 0.5])
        avg1 = popt[0]
        disp1 = popt[1]
        avg2 = popt[2]
        disp2 = popt[3]
        frac = popt[4]
        avg1err, disp1err, avg2err, disp2err, fracerr = np.sqrt(np.diag(pcov))
        print(avg1, disp1, avg2, disp2, frac)
    except RuntimeError:
        avg1 = avg
        disp1 = disp
        avg2 = 0.
        disp2 = 0.
        frac = 0.
        print("Warning: Double gaussian fit didn't work. Parameters:")
        print(avg1, disp1, avg2, disp2, frac)
        print("Number of particles in failed vorobin: " + str(len(quantity)))
    return np.array([avg1, disp1, avg2, disp2, frac])
