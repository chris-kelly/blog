from typing import Tuple
import numpy as np

class _eta(
    self, 
    X: np.ndarray, 
    beta: np.ndarray, 
    y: np.ndarray, 
    type: str="linear",
) -> Tuple(np.ndarray, np.ndarray):
    if type == "linear":
        y_hat = X.dot(beta)
    else:
        raise ValueError("Only linear eta is supported")
    residuals = y - y_hat
    return y_hat, residuals




class reweight(X, y, Omega):
    return X, y, cX, cY

class eta(X, betas):
    returns y_hat, residuals

class derive_coefficients_OLS(X, y, family)
    returns betas

class derive_coefficients_MLE(X, y, family)
    returns betas    

class error_covariance(y, y_hat)
    returns V

class standard_errors(X, y, betas, family)
    returns set

class confidence intervals()
    returns lb, ub






-> class OLS()
    * estimate betas -> OLS or MLE
    * shortcut SE or sandwich estimator for OLS
    * confidence intervals (or use sandwich estimator)
-> class GLM()
    * takes weights (part 2)
    * eta -> y_hat
    * estimate_betas -> MLE
    * glm SE or sandwich estimator for GLM
    * confidence intervals (or use sandwich estimator)
-> class GLS()
    * same as OLS, but takes weights
-> class PLM()




