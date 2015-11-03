# -*- coding: utf-8 -*-

from numpy import exp, log, sqrt, pi
from scipy.stats import norm
from scipy.special import gamma

class frch:
  """
  Class for Frechet (Type 2 GEV) distribution.
  
  alpha - shape parameter
  beta  - scale parameter
  m     - location parameter
  """
  
  def __init__(self, alpha, beta=1, m=0):
    self.alpha = alpha
    self.beta = beta
    self.m = m
    if alpha > 1:
        self.mu = m + beta*gamma(1-1/alpha)
    else:
        self.mu = None
    if alpha > 2:
        self.sig = beta*sqrt(gamma(1-2/alpha) - (gamma(1-1/alpha))**2)
    else:
        self.sig = None

  def u_to_x(self, u):
    return exp(-(1/self.alpha)*log(-log(norm.cdf(u, 0, 1))))*self.beta + self.m
