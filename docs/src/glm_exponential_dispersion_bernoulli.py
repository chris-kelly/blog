from glm_exponential_dispersion_parent import exponential_regression_parent

import numpy as np

class bernoulli(exponential_regression_parent):

  def __init__(self,seed = 0, ols_beta_init = False, **kwargs):
      """
      If using a pure Netwon method, this can fail to converge if initialized coefficients are too far from the neighbourhood of the global minima.
      One option is to initialize coefficients using OLS (i.e. linear probabiity model, lpm) and optimize from there.
      Otherwise, by default, it uses a more stable calculation for the variance of y.
      """
      if ols_beta_init:
        XtX = X.transpose().dot(X)
        Xty = X.transpose().dot(y2).flatten()
        lpm_beta = np.linalg.pinv(XtX).dot(Xty)
        super().__init__(seed, **kwargs, beta=lpm_beta.reshape(-1,1))
      else:
        super().__init__(seed, **kwargs)

  def theta(self, X=None):
        """ As canonical form, theta == link function. 
        So we simply equate θ = g(µ) = η(X)= X'β """
        beta = self.beta
        if X is None:
            X = self.X
        return X.dot(beta)

  def b(self,theta):
      """ b(θ) = ln[1+exp{θ}]: integral of the activation function """
      return np.log(1 + np.exp(theta))
  
  def db(self,theta):
      """ b'(θ) = Λ(θ) = 1/(1+exp{-θ}): Activation function is logistic (inverse of the logit-link function)"""
      return (1+np.exp(-theta))**-1
      # return np.exp(theta) * (1+np.exp(theta))**-1
      
  def d2b(self,theta):
      """ b''(θ) = Λ(θ)*(1-Λ(θ)): Differential of the activation function is the logistic distribution """
      _db = self.db(theta)
      return _db*(1-_db)
      # return np.exp(theta) * (1 + np.exp(theta))**-2

  def phi(self):
      """ Not needed - one parameter distribution """
      return 1
  
  def a(self,phi):
      """ Not needed - one parameter distribution """
      return 1
  
  def c(self,y,phi):
      """ No adjustment needed (pdf already sums to one) """
      return 0