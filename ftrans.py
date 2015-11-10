# -*- coding: utf-8 -*-

from numpy import dot, eye
from numpy.linalg import cholesky
from multiprocessing import Pool, cpu_count
from nataf import nataf

def corrmat_u(cmat_x, xdists):
    """
    Convert X-space correlation matrix to U-space correlation matrix.
    """

    cpus = max(1, cpu_count()-1) # Don't use every core!
    rho_dict = {}

    n = len(xdists)
    cmat_u = eye(n)
    if cmat_x.sum() > n: 
        # Generate correlation matrix in standard normal space, if necessary
        if __name__ == 'ftrans':
            # Parallelise the calculation of equivalent rhos 
            pool = Pool(cpus)
            rho_dict = {(i, j): pool.apply_async(nataf, [xdists[i], xdists[j], cmat_x[i, j]])
                                for i in range(n) for j in range(i+1, n, 1)}
            for ij in rho_dict.keys():
                rho_ij = rho_dict[ij].get(timeout=60)['x']
                cmat_u[ij], cmat_u[ij[::-1]] = rho_ij, rho_ij
            pool.close()
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
