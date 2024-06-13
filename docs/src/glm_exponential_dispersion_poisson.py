from glm_exponential_dispersion_parent import exponential_regression_parent

import numpy as np
from math import factorial

class poisson(exponential_regression_parent):

  def __init__(self,seed = 0, **kwargs):
      super().__init__(seed, **kwargs)

  def theta(self, X=None):
        """ As canonical form, theta == link function. 
        So we simply equate θ = g(µ) = η(X)= X'β """
        beta = self.beta
        if X is None:
            X = self.X
        return X.dot(beta)

  def b(self,theta):
      """ b(θ) = exp{θ}: integral of the activation function """
      return np.exp(theta)
  
  def db(self,theta):
      """ b'(θ) = exp{θ}: Activation function is exponential (inverse of the log-link function)"""
      return np.exp(theta)

  def d2b(self,theta):
      """ b''(θ) = exp{θ}: Differential of the activation function """
      return np.exp(theta)
  
  def phi(self):
      """ Not needed - one parameter distribution """
      return 1
  
  def a(self,phi):
      """ Not needed - one parameter distribution """
      return 1
  
  def c(self,y,phi):
      """ Adjustment needed so pdf sums to one. """
      # # Optionally use Stirling's approximation to deal easily with large factorials
      # def approx_ln_factorial(n):
      #     return n * np.log(n) - n
      y = y.flatten()
      return -np.log(np.array([factorial(v) for v in y.tolist()]).astype(float)).reshape(-1,1)