import numpy as np, pandas as pd
from typing import Optional, Tuple
from scipy.linalg import qr, solve_triangular
from scipy.stats import norm, t

class OLS:

    def _convert_types(self, z) -> np.ndarray:
        """ Re-shape and convert y to numpy array to work nicely with rest of functions """
        if type(z) in [pd.DataFrame, pd.Series]:
            z2 = z.to_numpy()
        if type(z) == list:
            z2 = np.array(z)
        if type(z) == np.ndarray:
            z2 = z
        else:
            raise TypeError('Array must be a pandas series/dataframe, numpy array or list')
        return z2
             
    def _get_y(self, y: Optional = None) -> np.ndarray:
        """Re-shape and convert y to numpy array to work nicely with rest of functions"""
        if y is None:
            y = self.y
        return self._convert_types(y).reshape(-1)

    def _get_X(self, X: Optional = None) -> Tuple[np.ndarray, np.ndarray]:
        """Re-shape and convert X to numpy array to work nicely with rest of functions
        Also return names for summarising in coefficient table"""
        if X is None:
            X = self.X
        X2 = self._convert_types(X)
        if type(X) == pd.DataFrame:
            exog_names = np.array(X.columns)
        elif type(X) == pd.Series:
            exog_names = np.array(['Unamed Exog Feature'])
        else:
            exog_names = np.arange(X2.shape[1])
        X2 = X2.reshape(-1,len(exog_names))
        return (X2, exog_names)

    def _clear_fitted_attributes(self):
        """Clears fitted attributes of the class"""
        self.beta, self.RSS, self.beta_se, self.conf_int, self.test_stat, self.p_val = None, None, None, None, None, None
        
    def __init__(
        self, 
        y: Optional[np.ndarray] = None, 
        X: Optional[np.ndarray] = None
        ) -> None:
        """Initializes the OLS class to run an least-squares regression"""
        if y is not None:
            self.y = self._get_y(y)
            self.n = len(self.y)
        if X is not None:
            self.X, self.exog_names = self._get_X(X)
            self.k = self.X.shape[1]
        if y is not None and X is not None and len(self.y) != self.X.shape[0]:
            raise ValueError("y and X must be the same size.")
        self._clear_fitted_attributes()

    def _quick_matrix_invert(self, X: np.ndarray) -> np.ndarray:
        """ Find the inverse of a matrix, using QR factorization """
        Q, R = qr(X)
        X_inv = solve_triangular(R, np.identity(X.shape[1])).dot(Q.transpose())
        return X_inv

    def _check_if_fitted(self):
        """Quick helper function that raises an error if the model has not been fitted already"""
        if self.beta is None:
            raise ValueError('Need to fit the model first - run fit()')
        else:
            return True

    def _estimate_ols_coefs(
        self,
        y: Optional[np.ndarray] = None,
        X: Optional[np.ndarray] = None
    ):
        """Estimates the OLS coefficients given a vector y and matrix X"""
        XTX = X.T.dot(X)
        XTY = X.T.dot(y)
        XTX_inv = self._quick_matrix_invert(XTX)
        coefs = XTX_inv.dot(XTY)
        return coefs, XTX_inv

    def fit(
        self,
        y: Optional[np.ndarray] = None,
        X: Optional[np.ndarray] = None,
    ):
        """Estimates the OLS coefficients given a vector y and matrix X"""
        self._clear_fitted_attributes()
        y = self._get_y(y)
        X, exog_names = self._get_X(X)
        if y is None or X is None:
            raise ValueError('X and y is required for fitting')
        if len(y) != X.shape[0]:
            raise ValueError("y and X must be the same size.")
        self.y, self.X, self.exog_names = y, X, exog_names
        self.n, self.k = X.shape
        self.DoF = self.n - self.k
        self.beta, self.var_X_inv = self._estimate_ols_coefs(y,X)

    def _assess_fit(
        self,
        y: Optional[np.ndarray] = None,
        X: Optional[np.ndarray] = None,
    ) -> float:
        """Returns the unadjusted R^2"""
        self._check_if_fitted()
        if (y is None and X is not None) or (y is not None and X is None):
            raise ValueError('Need to either provide both X and y, (or provide neither and R^2 is based on the X and y used for fitting)')
        else:
            y, (X, exog_names) = self._get_y(y), self._get_X(X)
        y_hat = self.predict(self.X)
        residuals = (y - y_hat).reshape(-1, 1)
        RSS = residuals.T.dot(residuals)
        TSS = (y - y.mean()).T.dot(y - y.mean())
        unadj_r_squared = 1 - RSS/TSS
        
        if (y == self.y).all() and (X == self.X).all():
            self.residuals = residuals
            self.RSS = RSS
            self.TSS = TSS
            self.unadj_r_squared = unadj_r_squared
        return unadj_r_squared

    def _standard_error(self,) -> np.ndarray:
        """Returns the standard errors for the coefficients from the fitted model"""
        if self.RSS is None:
            self._check_if_fitted()
            self._assess_fit()
        sigma_sq = self.RSS / float(self.DoF) * np.identity(len(self.beta))
        var_b = sigma_sq.dot(self.var_X_inv)
        self.beta_se = np.sqrt(np.diag(var_b))
        return self.beta_se

    def _confidence_intervals(self, size = 0.95):
        """Returns the confidence intervals for the coefficients from the fitted model"""
        if self.beta_se is None:
            self._check_if_fitted()
            self._standard_error()
        alpha = 1-(1-size)/2
        self.conf_int = np.array([
            self.beta - t.ppf(alpha, self.DoF) * self.beta_se,
            self.beta + t.ppf(alpha, self.DoF) * self.beta_se
        ])
        return self.conf_int

    def _test_statistic(self) -> np.ndarray:
        """Returns the test statistics for the coefficients from the fitted model"""
        if self.conf_int is None:
            self._check_if_fitted()
            self.conf_int = self._confidence_intervals()
        self.test_stat = self.beta.flatten() / self.beta_se
        return self.test_stat

    def _p_value(self, z_dist: bool = False) -> np.ndarray:
        """Returns the p-values for the coefficients from the fitted model."""
        if self.test_stat is None:
            self._check_if_fitted()
            self.test_stat = self._test_statistic()
        if z_dist:
            self.p_val = [norm.cdf(-abs(z)) + 1 - norm.cdf(abs(z)) for z in self.test_stat]
        else:
            self.p_val = [2 * t.sf(abs(x), self.DoF) for x in self.test_stat]
        return self.p_val    

    def predict(
        self,
        X: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict values for y. Returns fitted values if X not provided."""
        self._check_if_fitted()
        X2, exog_names = self._get_X(X)
        y_hat = X2.dot(self.beta)
        if X is None:
            self.y_hat = y_hat
        return y_hat

    def summary(self, z_dist: bool = False) -> pd.DataFrame:
        """Returns the coefficients, standard errors, test statistics and p-values in a Pandas DataFrame."""
        if self.p_val is None:
            self._check_if_fitted()
            self._p_value(z_dist)
        summary = pd.DataFrame(
            data={
                'Coefficient': self.beta.flatten(),
                'Standard Error': self.beta_se,
                'Lower bound': self.conf_int[0],
                'Upper bound': self.conf_int[1],
                'test-statistic': self.test_stat,
                'p-value': self.p_val,
            },
            index=self.exog_names,
        )
        return summary