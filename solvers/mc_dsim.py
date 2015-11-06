# -*- coding: utf-8 -*-
from __future__ import division # Python 2.x compatibility
from numpy import array, sqrt, eye, isnan, zeros, dot
from numpy.random import normal
from scipy.stats import norm


def dir_sim(g, xdists, itr=1000, convg=1e-4):
    """
    Directional simulation algorithm to estimate reliability index. 
    """

    max_dist = 10.
    n_steps = 20
    ray_segs = linspace(0, max_dist, n_steps)
    d_seg = ray_segs[1] - ray_segs[0]
    n_dims = len(xdists)
    bisect_max = 500
    raw_rays = random.multivariate_normal(zeros(n_dims), cov=eye(n_dims), size=itr)
    norms = linalg.norm(raw_rays, axis=-1)
    rays = (raw_rays.T/norms).T
    Pf = 0.
    am = []
    for ray in rays:
        segments = array([g(*u_to_x(ray*l, xdists)) for l in ray_segs])
        t = (diff(sign(segments)) != 0)
        x0=0; x1=0        
        if any(t):
            x0 = ray*ray_segs[t][0]
            x1 = x0 + d_seg*ray
            for i in xrange(bisect_max):
                m = (x0 + x1)/2
                dx = x1 - x0
                len_dx = sqrt((dx*dx).sum())
                if len_dx/2 < convg:
                    beta = (m*m).sum()
                    Pf = Pf + 1 - chi2.cdf(beta, n_dims)
                    am.append(m)
                    break
                else:
                    if sign(g(*u_to_x(x0, xdists))) == sign(g(*u_to_x(m, xdists))):
                        x0 = m
                    else:
                        x1 = m
    print '%.3e' % (Pf/itr)
    return array(am)
