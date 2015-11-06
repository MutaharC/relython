# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import multivariate_normal
from scipy.integrate import simps
from scipy.optimize import minimize


def integrand(rho1, x1x2_norm, uu):
    """
    Integrand in the correlation integral to determine the effective correlation 
    rho_prime between var1 and var2 in U-space giving the target correlation.
    """

    # 2D multivariate normal with correlation rho_prime evaluated on grid pos
    mvn2 = multivariate_normal.pdf(uu, mean=[0, 0], cov=[[1, rho1], [rho1, 1]])
 
    return x1x2_norm * mvn2


def nataf(var1, var2, rho, limu=6, n=100):
    """
    Solve numerically for rho_prime in U-space to get rho in X-space.
    """

    # Generate grid in U-space on which to evaluate correlation integral
    u1 = np.linspace(-limu, limu, n)
    uu1, uu2 = np.meshgrid(u1, u1)
    uu = np.dstack((uu1, uu2))

    # Transform from U- to X-space
    x1 = var1.u_to_x(uu1) 
    x2 = var2.u_to_x(uu2)
    x1x2_norm = (x1-var1.mu)/var1.sig * (x2-var2.mu)/var2.sig
    
    # Find rho_prime such that correlation coeff is rho after u_to_x transform
    def func(rho1): 
        return abs(rho - simps(simps(integrand(rho1, x1x2_norm, uu), u1), u1))

    result = minimize(func, x0=rho, method='Nelder-Mead', 
                      options={'ftol': 1e-6, 'maxiter': 100, 'disp': False})
    return result


def mvn_pdf(x, mu, cov):
    """
    Calculate the multivariate normal pdf

    Keyword arguments:
        x   : numpy array of a "d x 1" sample vector
        mu  : numpy array of a "d x 1" mean vector
        cov : numpy array of a "d x d" covariance matrix
    """

    assert(mu.shape[0] > mu.shape[1]), 'mu must be a row vector'
    assert(x.shape[0] > x.shape[1]), 'x must be a row vector'
    assert(cov.shape[0] == cov.shape[1]), 'covariance matrix must be square'
    assert(mu.shape[0] == cov.shape[0]), 'cov_mat and mu_vec must have same dims'
    assert(mu.shape[0] == x.shape[0]), 'mu and x must have the same dims'
    part1 = 1 / ( ((2* np.pi)**(len(mu)/2)) * (np.linalg.det(cov)**(1/2)))
    part2 = (-1/2) * ((x-mu).T.dot(np.linalg.inv(cov))).dot((x-mu))
    return float(part1 * np.exp(part2))
