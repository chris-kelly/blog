import numpy as np

class exponential_regression_parent():

    def __init__(self,seed=0,y=None,X=None,beta=None):
      """
      seed: Scalar sets seed for random coefficient initialization
      y:    Dependent variable. A 1d vector of numeric values. Optional.
      X:    Independent variables. A 2d matrix of numeric values. Optional.
      beta: Initial coefficients. A 1d vector of numeric values. Optional.
      """
      self.seed = seed
      self.y = y
      self.X = X
      self.beta = beta

    def _initialize(self,y=None,X=None):
        if y is None:
            if self.y is None:
                raise ValueError('Please provide y')
            else:
                y = self.y
        if X is None:
            if self.X is None:
                raise ValueError('Please provide X')
            else:
                X = np.array(self.X).reshape(y.shape[0],-1)
        self.y = np.array(y).reshape(-1,1)
        self.X = np.array(X).reshape(y.shape[0],-1)
        self.n, self.k = X.shape
        np.random.seed(self.seed)
        if self.beta is None:
            self.beta = np.random.normal(0, 0.5, self.k).reshape((self.k,1))

    def _theta(self, theta=None):
        """ helper function """
        if theta is None:
            theta = self.theta()
        return theta

    def _phi(self,phi=None):
        """ helper function """
        if phi is None:
            phi = self.phi()
        return phi

    def negative_log_likelihood(self):
        """ Negative log likelihood i.e. the current cost"""
        y = self.y
        theta, phi = self._theta(), self._phi()
        a, b, c = self.a, self.b, self.c
        log_likelihood = ( y * theta - b(theta) ) / a(phi) + c(y,phi)
        L = -1 * np.sum(log_likelihood)
        return L

    def informant(self, theta=None, phi=None):
        """ First derivative of the cost function with respect to theta """
        y, X, a, b, db = self.y, self.X, self.a, self.b, self.db
        theta, phi = self._theta(theta), self._phi(phi)
        dJ = (1/a(phi)) * X.T.dot( db( theta ) - y )
        return dJ
    
    def hessian(self, theta=None, phi=None, use_stable=True):
        """ Second derivative of the cost function with respect to theta """
        X, y = self.X, self.y
        a, d2b = self.a, self.d2b
        theta, phi = self._theta(theta), self._phi(phi)
        if use_stable:
            meat = a(phi)**-2 * np.diagflat((self.db(theta)-y)**2) # using residual variance
        else:
            meat = a(phi)**-1 * np.diagflat(self.d2b(theta)) # using second differential directly
        d2L = X.T.dot(meat).dot(X)
        return d2L

    def update_beta(self, use_stable_hessian=True):
        """ A single step towards optimizing beta """
        X, beta = self.X, self.beta.copy()
        theta, phi = self._theta(), self._phi()
        learning_rate = np.linalg.inv(self.hessian(theta, use_stable=True))
        dL = self.informant(theta, phi)
        beta -= learning_rate.dot(dL)
        return beta

    def fit(self, y=None, X=None, max_iter=100, epsilon=1e-8, use_stable_hessian=True):
        """ Fit the model using Newton Raphson Optimization """
        self._initialize(y,X)
        for i in range(max_iter):
            old_beta = self.beta.copy()
            new_beta = self.update_beta(use_stable_hessian)
            self.beta = new_beta
            if (np.abs(new_beta - old_beta)/(0.1 + np.abs(new_beta)) <= epsilon).all():
                print("Fully converged by iteration " + str(i))
                break
            if (i == max_iter):
                print("Warning - coefficients did not fully converge within " + str(max_iter) + " iterations.")

    def predict(self, X=None):
        """ Predicted value of y uses the activation function, b'(θ) """
        if X is None:
            X = self.X
        y_hat = self.db( self.theta(X=X) )
        return y_hat