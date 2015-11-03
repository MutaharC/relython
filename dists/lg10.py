# -*- coding: utf-8 -*-

class lg10:
  """TBC"""
  def __init__(self, p1, p2):
    self.p1 = p1
    self.p2 = p2
    self.mu = 10**(p1 + p2**2/2)
    self.sig = sqrt((10**(p2**2)-1)*10**(2*p1+p2**2))

  def u_to_x(self, u):
    return 10**(u*self.p2 + self.p1)
