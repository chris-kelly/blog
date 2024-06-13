from glm_exponential_dispersion_parent import exponential_regression_parent

import numpy as np
from math import pi

class gaussian(exponential_regression_parent):

  def __init__(self,seed = 0,**kwargs):
      super().__init__(seed, **kwargs)

  def theta(self, X=None):
        """ As canonical form, theta == link function. 
        So we simply equate θ = g(µ) = η(X)= X'β """
        beta = self.beta
        if X is None:
            X = self.X
        return X.dot(beta)

  def b(self,theta):
      """ b(θ) = 0.5*θ^2: Integral of the activation function """
      return 0.5*theta**2
  
  def db(self,theta):
      """ b'(θ) = θ: Activation function is the identity """
      return theta

  def d2b(self,theta):
      """ b''(θ) = 1: Differential of the activation function is just one """
      return np.ones(theta.shape)

  def phi(self):
      """ φ = σ^2: Dispersion measure is variance
      It is estimated from the residuals, i.e. s = RSS / DoF
      """
      y, n, k = self.y, self.n, self.k
      y_hat = self.predict(self.X)
      var = np.sum( (y - y_hat)**2 ) / n # / ( n - k )
      return var
      
  def a(self,phi):
      """ a(φ) = σ^2: The variance of y """
      return phi

  def c(self,y,phi):
      """ Adjustment needed so pdf sums to one """
      return -0.5 * ( y**2/phi + np.log(2*pi*phi))