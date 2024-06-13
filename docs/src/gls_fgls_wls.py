from scipy.sparse import bsr_array
import pandas as pd, numpy as np
from typing import Optional, Tuple

from ols_blue import OLS
from ols_sandwich_estimators import sandwich

class LS(sandwich):

    def __init__(
        self, 
        y: Optional[np.ndarray] = None, 
        X: Optional[np.ndarray] = None,
        omega: Optional[np.ndarray] = None
        ) -> None:
        """Initializes the LS class to run an least-squares regression"""
        super().__init__(y, X)
        self.omega = omega
        self.P = None

    def _estimate_gls_coefs(self, y: np.ndarray, X: np.ndarray, omega: np.ndarray):
        """Estimates the GLS coefficients given a vector y and matrix X"""
        try:
            P = np.linalg.cholesky(omega)
            PX = P.dot(X)
            Py = P.dot(y)
            coefs, XTOX_inv = self._estimate_ols_coefs(Py,PX)
        except:
            omega_inv = np.linalg.inv(omega)
            XTO = X.T.dot(omega_inv)
            XTOX = XTO.dot(X)
            XTOX_inv = self._quick_matrix_invert(XTOX)
            XTOY = XTO.dot(y)
            coefs = XTOX_inv.dot(XTOY)
        return coefs, XTOX_inv
        
    def fit(
        self,
        y: Optional[np.ndarray] = None,
        X: Optional[np.ndarray] = None,
        omega: Optional[np.ndarray] = None,
        fgls = None,
    ):
        y = self._get_y(y)
        X, exog_names = self._get_X(X)
        if y is None or X is None:
            raise ValueError('X and y is required for fitting')
        if len(y) != X.shape[0]:
            raise ValueError("y and X must be the same size.")
        self.y, self.X, self.exog_names = y, X, exog_names
        self.n, self.k = X.shape
        self.DoF = self.n - self.k
        if omega is not None:
            self.omega = omega
        if self.omega is None:
            self.beta, self.var_X_inv = self._estimate_ols_coefs(y,X)
        if self.omega is not None or fgls is not None:
            if self.omega is not None and fgls is not None:
                raise ValueError('Cannot specify both omega and fgls')
            elif fgls is not None:
                self._assess_fit()
                if type(fgls) == str:
                      if fgls == "homoskedastic":
                          self.omega = self._homoskedastic()
                      elif fgls == "heteroskedastic":
                          self.omega = self._heteroskedastic()
                elif type(fgls) == np.ndarray:
                    self.omega = self._clustered(fgls)
            self.beta, self.var_X_inv = self._estimate_gls_coefs(y,X,self.omega)