# -*- coding: utf-8 -*-
from numpy import log
from scipy.stats import norm

class expo:
  """TBC"""
  def __init__(self, p1):
    self.p1 = p1
    self.mu = 1.0/p1
    self.sig = 1.0/p1

  def u_to_x(self, u):
    return -log(1.0 - norm.cdf(u, 0, 1))/self.p1
