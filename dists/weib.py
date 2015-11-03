# -*- coding: utf-8 -*-

from numpy import log
from scipy.stats import norm

class weib:
  """TBC"""
  def __init__(self, l, k):
    self.l = l
    self.k = k
    self.mu = l*gamma(1+1/k)
    self.sig = l**2 * (gamma(1+2/k) - gamma(1+1/k)**2)

  def u_to_x(self, u):
    return exp((log(-log(1-norm.cdf(u, 0, 1)))+self.k*log(self.l))/self.k)
