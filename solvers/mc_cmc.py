# -*- coding: utf-8 -*-
from __future__ import division # Python 2.x compatibility
from numpy import array, sqrt, eye, isnan, zeros, dot
from numpy.random import RandomState
from scipy.stats import norm


def cmc(g, xdists, u_to_x, T, seed, maxitr):
    """
    Crude Monte Carlo simulation.
    """

    # Seed the random number generator if required
    if seed == -1:
        prng = RandomState()
    else:
        prng = RandomState(seed)
    
    # Generate standard normal samples centered at the origin
    u0 = zeros(len(xdists))
    covmat = eye(len(xdists)) 
    u = prng.multivariate_normal(u0, covmat, size=maxitr).T
    g_mc = g(u_to_x(u, xdists, T))

    # Convert g-function output to pass/fail indicator function and estimate pf
    g_mc[g_mc>0] = 0
    g_mc[g_mc<0] = 1
    mu_pf = g_mc.mean()
    beta = -norm.ppf(mu_pf) if mu_pf < 0.5 else norm.ppf(mu_pf)

    # Convergence metrics (standard deviation, standard error, CoV of s.e.)
    std_pf = g_mc.std(ddof=1) # Calculate sample standard deviation
    se_pf = std_pf/sqrt(maxitr)
    cv_pf = se_pf/mu_pf

    return {'vars': xdists, 'beta': beta, 'Pf': mu_pf, 'stderr': se_pf, 
            'stdcv': cv_pf}
