# -*- coding: utf-8 -*-

from numpy import log, sqrt, pi
from scipy.stats import norm

class rayl:
  """TBC"""
  def __init__(self, p1):
    self.p1 = p1
    self.mu = p1*sqrt(pi/2)
    self.sig = p1*sqrt((4-p1)/2)

  def u_to_x(self, u):
    return self.p1*sqrt(-log((1.0 - norm.cdf(u, 0, 1))**2))
