# -*- coding: utf-8 -*-

class norm:
  """TBC"""
  def __init__(self, p1, p2):
    self.p1 = p1
    self.p2 = p2
    self.mu = p1
    self.sig = p2

  def u_to_x(self, u):
    return u*self.p2 + self.p1
