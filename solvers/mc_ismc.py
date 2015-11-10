# -*- coding: utf-8 -*-
from __future__ import division # Python 2.x compatibility
from numpy import array, sqrt, eye, isnan, zeros, dot
from numpy.random import RandomState 
from scipy.stats import norm, multivariate_normal

#import ftrans as tr
from solvers import slsqp


def ismc(g, xdists, u_to_x, T, inp):
    """
    Importance sampling Monte Carlo.
    """

    # Seed the random number generator if required
    if inp['seed'] == -1:
        prng = RandomState()
    else:
        prng = RandomState(inp['seed'])
    
    # Use FORM to get estimate of u*
    u_beta = slsqp(g, xdists, u_to_x, T, inp)['u_beta']

    # Generate standard normal samples centred at u*
    covmat = eye(len(xdists))
    v = prng.multivariate_normal(u_beta, covmat, size=inp['maxitr']).T
    g_mc = g(u_to_x(v, xdists, T))
    
    # Define importance sampling functions weighting functions
    fv = multivariate_normal(zeros(len(xdists)), covmat).pdf
    hv = multivariate_normal(u_beta, covmat).pdf

    # Convert g-function output to pass/fail indicator function and estimate pf
    g_mc[g_mc>0] = 0
    g_mc[g_mc<0] = 1
    indfunc = g_mc * fv(v.T)/hv(v.T)
    mu_pf = indfunc.mean()
    beta = -norm.ppf(mu_pf) if mu_pf < 0.5 else norm.ppf(mu_pf)

    # Convergence metrics (standard deviation, standard error, CoV of s.e.)
    std_pf = indfunc.std(ddof=1) # Calculate sample standard deviation
    se_pf = std_pf/sqrt(inp['maxitr'])
    cv_pf = se_pf/mu_pf

    return {'vars': xdists, 'beta': beta, 'Pf': mu_pf, 'stderr': se_pf, 
            'stdcv': cv_pf}
