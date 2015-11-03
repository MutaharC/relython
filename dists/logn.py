# -*- coding: utf-8 -*-
from numpy import exp, sqrt

class logn:
  """TBC"""
  def __init__(self, p1, p2):
    self.p1 = p1
    self.p2 = p2
    self.mu = exp(p1 + p2**2/2)
    self.sig = sqrt((exp(p2**2)-1)*exp(2*p1+p2**2))

  def u_to_x(self, u):
    return exp(u*self.p2 + self.p1)
