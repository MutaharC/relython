# -*- coding: utf-8 -*-
from numpy import sqrt, dot, zeros
from scipy.stats import norm
from scipy.optimize import minimize
from ftrans import u_to_x

def slsqp(g, xdists, u_to_x, T, maxitr, tol, ftol, eps):
    """
    Wrapper for scipy.optimize SLSQP constrained minimisation routine.
    Minimizes beta subject to the constraint g(x)=0. 
    More stable than Rackwitz-Fiessler.
    """

    u0 = zeros(len(xdists))

    # Define constraint function to be applied during minimisation
    g_SLSQP = lambda u: g(u_to_x(u, xdists, T))

    ans = minimize(lambda u:(u*u).sum(), x0=u0, method='SLSQP', tol=tol, 
                   constraints={'type': 'eq', 'fun': g_SLSQP}, 
                   options={'ftol': ftol, 'eps': eps, 
                            'maxiter': maxitr, 'disp': False})
    u_beta = ans['x']
    x_beta = u_to_x(u_beta, xdists, T)
    g_beta = g(x_beta)
    beta = sqrt(ans['fun'])

    # Sensitivity factors
    alpha = u_beta/sqrt((u_beta*u_beta).sum())
    
    # Check if g-function is negative at the origin of U-space
    if g(u_to_x(u0, xdists, T)) < 0: 
        Pf = norm.cdf(beta)
    else:
        Pf = norm.cdf(-beta)
    return {'vars': xdists, 'beta': beta, 'Pf': Pf, 'u_beta': u_beta, 
            'x_beta': x_beta, 'alpha': alpha, 'alpha_tr': dot(T, alpha), 
	    'nitr': ans['nit'], 'g_beta': g_beta, 'tol': tol,
            'msgs': ans['message']}
