# -*- coding: utf-8 -*-

from numpy import dot, array, eye
from numpy.linalg import cholesky
from nataf import nataf

def corrmat_u(cmat_x, xdists):
    """
    Convert X-space correlation matrix to U-space correlation matrix.
    """

    n = len(xdists)
    cmat_u = eye(n)
    if cmat_x.sum() > n: 
        # Generate correlation matrix in standard normal space, if necessary
        for i in range(n):
            for j in range(i+1, n, 1):
                rho_prime = nataf(xdists[i], xdists[j], cmat_x[i, j])['x']
                cmat_u[i, j], cmat_u[j, i] = rho_prime, rho_prime
    return cmat_u


def utrans(cmat_u):
    """
    Generate transformation matrix, L, from standard to correlated N(0,1) 
    multivariate normal space.
    """

    L = cholesky(cmat_u)
    return L


def u_to_x(u, xdists, L):
    """
    Transform a point in standard normal space to basic variable space.
    """
    
    # Transformation u from independent to correlated U-space
    Lu = dot(L, u)
    
    # Transform from U-space to X-space
    x = [xdist.u_to_x(Lu[i]) for i, xdist in enumerate(xdists)]
    return x
