# -*- coding: utf-8 -*-

from numpy import array, sqrt, eye, isnan, zeros, dot
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

def hlrf(g, xdists, u_to_x, T, inp):
    """
    Hasofer-Lind-Rackwitz-Fiessler algorithm, after Melchers (1999), 4.3.6.
    """

    u0 = array([0.]*len(inp['vars']))
    eps = inp['eps']
    d_beta = 999
    g_0 = g(u_to_x(u0, xdists, T))                     # Evaluate g-function at U-origin 
    u = u0
    beta_arr = zeros(inp['maxitr'])                    # Iterations of reliability index
    for i in range(inp['maxitr']):
        beta = sqrt((u*u).sum())                       # Calculate beta
        gu = g(u_to_x(u, xdists, T))                   # Evaluate g(x(u))
        dg_by_du = dgdu(g, u, u_to_x, xdists, T, eps)  # Evaluate grad(g(x(u))
        l = sqrt((dg_by_du*dg_by_du).sum())            # Length of gradient vector
        alpha = dg_by_du/l                             # Direction cosines of grad(g)
        u = -alpha*(beta + gu/l)                       # Updated estimate of u
        x = u_to_x(u, xdists, T)                       # Updated estimate of x
        gu = g(x)                                      # Updated estimate of g(x(u))
        beta_arr[i] = sqrt((u*u).sum())                # Updated estimate of beta  
        if i > 0:
            d_beta = beta_arr[i] - beta_arr[i-1]
        if d_beta<=inp['tol'] and abs(gu)<=inp['ftol']:# Check convergence criteria
            break
    u_beta = u                                          # Design point in U space
    x_beta = x                                          # Design point in X space
    g_beta = gu                                         # Value of g-function at design point
    if g_0 < 0:                                         # Check if g-function <0 at U-origin
        Pf = norm.cdf(beta)
    else:
        Pf = norm.cdf(-beta)

    msgs = []
    if d_beta > inp['tol']:
        msgs.append('''Warning: beta convergence = {0:.2e} > {1:.2e} after 
                     {2:d} iterations'''.format(d_beta, inp['tol'], i))
    if isnan(beta):
        msgs.append('Warning: failed to converge - beta is nan')

    return {'vars': xdists, 'beta': beta, 'Pf': Pf, 'u_beta': u_beta, 
            'x_beta': x_beta, 'alpha': alpha, 'alpha_tr': dot(T, u_beta), 
	    'nitr': i+1, 'g_beta': g_beta, 'msgs': msgs}
