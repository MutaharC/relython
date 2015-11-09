# -*- coding: utf-8 -*-
from __future__ import division # Python 2.x compatibility
from numpy import array, sqrt, eye, isnan, zeros, dot
import numpy.linalg as la
from numpy.random import RandomState 
from scipy.stats import norm, chi2


def dgdu(g, u, u_to_x, xdists, T, eps):
    """
    Gradient of g(u_to_x(u)) evaluated numerically using central differences.
    """

    du_mat = eps*eye(u.size)
    grad = array([0.5*(g(u_to_x(u+du_m, xdists, T))-
                       g(u_to_x(u-du_m, xdists, T)))/eps 
                  for du_m in du_mat])   
    return grad


def dsim(g, xdists, u_to_x, T, inp):
    """
    Directional simulation. 
    """

    # Seed the random number generator if required
    if inp['seed'] == -1:
        prng = RandomState()
    else:
        prng = RandomState(inp['seed'])

    ndims = len(inp['vars'])
    rays = prng.normal(0, 1, size=(inp['maxitr'], ndims))
    unit_rays = rays/la.norm(rays)
    pfs = zeros(inp['maxitr'])

    for i, ray in enumerate(unit_rays):
        d_ray = 1
        ray0 = ray 
        while abs(g(u_to_x(ray, xdists, T))) > inp['ftol'] and abs(d_ray) > inp['ftol']:
            # Calculate gradient and directional derivative
            grad = dgdu(g, ray, u_to_x, xdists, T, inp['eps'])
            dderiv = dot(grad, ray/la.norm(ray))
            # delta length of ray is g function / directional derivative
            d_ray = g(u_to_x(ray, xdists, T))/dderiv
            ray = ray - d_ray*ray/la.norm(ray)
        # Only store a non-zero pf if the ray hasn't changed direction 
        pfs[i] = 1 - chi2.cdf(la.norm(ray)**2, ndims) if dot(ray0, ray) > 0 else 0
    pfs[isnan(pfs)] = 0
    mu_pf, std_pf = pfs.mean(), pfs.std()
    se_pf = std_pf/sqrt(inp['maxitr'])
    cv_pf = se_pf/mu_pf
    beta = -norm.ppf(mu_pf) if mu_pf < 0.5 else norm.ppf(mu_pf)
    msgs = []
    return {'vars': xdists, 'beta': beta, 'Pf': mu_pf, 'stderr': se_pf, 
            'stdcv': cv_pf, 'nitr': inp['maxitr'], 'g_beta': -1, 'msgs': msgs}
