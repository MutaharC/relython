# -*- coding: utf-8 -*-
from numpy import log, sqrt
from scipy.stats import norm, gamma


class gamm:
  """TBC"""

  def __init__(self, k, th):
    self.k = k
    self.th = th
    self.mu = k*th
    self.sig = sqrt(k)*th

  def u_to_x(self, u):
    return gamma.ppf(norm.cdf(u, 0, 1), a=self.k, scale=self.th)
