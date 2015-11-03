# -*- coding: utf-8 -*-

from numpy import sqrt, exp, log, log10, pi
from scipy.special import gamma
from scipy.optimize import minimize

eul = 0.577215664901532

def lnconv(mu_ln, sig_ln, base='e'):
    """
    Convert mean and standard deviation of lognormal distribution
    to mean and standard deviation of the underlying normal distribution.
    """

    if base == 'e':
        sig = sqrt(log(sig_ln**2/exp(2*log(mu_ln))+1))
        mu = log(mu_ln) - sig**2/2
    elif base == 10:
        sig = sqrt(log10(sig_ln**2/10**(2*log10(mu_ln))+1))
        mu = log10(mu_ln) - sig**2/2
    else:
        print('Use e or 10')
    return mu, sig


def frconv(mu, sig):
    """
    Convert mean and standard deviation of Frechet (Type-II GEV) distribution
    to shape and scale parameters for use in input files.
    """

    #beta_gumb = (sqrt(6)/pi)*sig
    #alpha_gumb = mu - eul*beta_gumb
    #alpha = 1/beta_gumb
    #beta = exp(alpha_gumb)
    #return alpha, beta


    def frechet(x):
        alpha, beta = x
        dmu = mu - beta*gamma(1-1/alpha)
        dsig = sig - beta*sqrt(gamma(1-2/alpha) - (gamma(1-1/alpha))**2)
        return dmu**2 + dsig**2

    ans = minimize(frechet, x0=[2,1], method='Nelder-Mead')

    return ans
