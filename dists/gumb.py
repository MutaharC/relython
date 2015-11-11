# -*- coding: utf-8 -*-

from numpy import log, sqrt, pi
from scipy.stats import norm

eul = 0.577215664901532

class gumb:
  """TBC"""
  def __init__(self, loc, beta):
    self.loc = loc
    self.beta = beta
    self.mu = loc+beta*eul
    self.sig = pi*beta/sqrt(6)

  def u_to_x(self, u):
    return -log(-log(norm.cdf(u, 0, 1)))*self.beta + self.loc
