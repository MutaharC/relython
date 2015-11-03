# -*- coding: utf-8 -*-
from numpy import sqrt
from scipy.stats import norm

class unif:
  """TBC"""
  def __init__(self, p1, p2):
    self.p1 = p1
    self.p2 = p2
    self.mu = (p1+p2)/2
    self.sig = 1.0/sqrt(12)*(p2-p1)

  def u_to_x(self, u):
    return self.p1+(self.p2-self.p1)*norm.cdf(u, 0, 1)
