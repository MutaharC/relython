# -*- coding: utf-8 -*-
from __future__ import division # Python 2.x compatibility
from numpy import array, sqrt, eye, isnan, zeros, dot
from numpy.random import normal
from scipy.stats import norm

def cmc(g, u_to_x, T, inp):
    """Crude Monte Carlo simulation"""

    u_mc = normal(0, 1, size=(inp['maxitr'], len(inp['vars'])))
    g_mc = g(u_to_x(u_mc.T, inp['vars'], T))                  # Evaluate g-function at U-origin
    Pf = g_mc[g_mc<0].size/inp['maxitr']
    beta = -norm.ppf(Pf) if Pf < 0.5 else norm.ppf(Pf)
    msgs = []
    # if d_beta > inp['tol']:
    #     msgs.append('Warning: beta convergence = {0:.2e} > {1:.2e} after {2:d} iterations'.format(d_beta, inp['tol'], i))
    # if isnan(beta):
        # msgs.append('Warning: failed to converge - beta is nan')
    return {'beta': beta, 'Pf': Pf, 'u_beta': None, 'x_beta': None, 'alpha': None, 'alpha_tr': None, 'nitr': inp['maxitr'], 'g_beta': -1, 
            'msgs': msgs}