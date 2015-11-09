# -*- coding: utf-8 -*-

from __future__ import division # Python 2.x compatibility
from numpy import array, sqrt, eye, isnan, zeros, dot
from numpy.random import RandomState
from scipy.stats import norm

def cmc(g, xdists, u_to_x, T, inp):
    """
    Crude Monte Carlo simulation.
    """

    # Seed the random number generator if required
    if inp['seed'] == -1:
        prng = RandomState()
    else:
        prng = RandomState(inp['seed'])
    
    u_mc = prng.normal(0, 1, size=(inp['maxitr'], len(inp['vars'])))
    g_mc = g(u_to_x(u_mc.T, xdists, T))
    indfunc = (g_mc <= 0).astype(int)
    mu_pf = indfunc.mean()
    std_pf = indfunc.std(ddof=1) # Calculate sample standard deviation
    se_pf = std_pf/sqrt(inp['maxitr'])
    cv_pf = se_pf/mu_pf
    beta = -norm.ppf(mu_pf) if mu_pf < 0.5 else norm.ppf(mu_pf)
    msgs = []
    return {'vars': xdists, 'beta': beta, 'Pf': mu_pf, 'stderr': se_pf, 
            'stdcv': cv_pf, 'nitr': inp['maxitr'], 'g_beta': -1, 'msgs': msgs}
