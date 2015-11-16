# -*- coding: utf-8 -*-
from numpy import array, sqrt, eye, isnan, zeros
from scipy.stats import norm


def dgdu(g, u, u_to_x, xdists, T, eps):
    """
    Gradient of g(u_to_x(u)) evaluated numerically using central differences.
    """

    du_mat = eps*eye(u.size)
    grad = array([0.5*(g(u_to_x(u+du_m, xdists, T))-
                       g(u_to_x(u-du_m, xdists, T)))/eps 
                  for du_m in du_mat])   
    return grad


def hlrf(g, xdists, u_to_x, T, maxitr, tol, ftol, eps):
    """
    Hasofer-Lind-Rackwitz-Fiessler algorithm, after Melchers (1999), 4.3.6.
    """

    u0 = zeros(len(xdists))
    d_beta = 999
    u = u0
    beta_arr = zeros(maxitr)
    
    for i in range(maxitr):
        beta = sqrt((u*u).sum())
        gu = g(u_to_x(u, xdists, T))

        # Direction cosines of grad(g)
        grad = dgdu(g, u, u_to_x, xdists, T, eps)
        l = sqrt((grad*grad).sum())
        alpha = grad/l

        # Updated estimates of u, x, g(u(x)) and beta
        u = -alpha*(beta + gu/l)
        x = u_to_x(u, xdists, T)
        gu = g(x)
        beta_arr[i] = sqrt((u*u).sum())
        
        # Check convergence criteria
        if i > 0:
            d_beta = beta_arr[i] - beta_arr[i-1]
        if d_beta <= tol and abs(gu) <= ftol:
            break

    # Check if g-function < 0 at U-origin
    if g(u_to_x(u0, xdists, T)) < 0:                                        
        Pf = norm.cdf(beta)
    else:
        Pf = norm.cdf(-beta)

    msgs = []
    if d_beta > tol:
        msgs.append("""Warning: beta convergence = {0:.2e} > {1:.2e} after 
                     {2:d} iterations""".format(d_beta, tol, i))
    if isnan(beta):
        msgs.append('Warning: failed to converge - beta is nan')

    return {'vars': xdists, 'beta': beta, 'Pf': Pf, 'u_beta': u, 'x_beta': x,
            'alpha': alpha, 'nitr': i+1, 'g_beta': gu, 'tol': d_beta, 
            'msgs': msgs}
