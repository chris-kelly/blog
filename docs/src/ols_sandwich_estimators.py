from ols_blue import OLS
from scipy.sparse import bsr_array
import pandas as pd, numpy as np
from typing import Optional, Tuple

class error_covariance():

    def _homoskedastic(self) -> np.ndarray:
        """Return an error covariance matrix assuming homoskedasticity"""
        e = self.residuals
        ee = np.identity(self.n) * e.T.dot(e)
        ee /= float(self.DoF)
        return ee

    def _heteroskedastic(self, type: str='HC0') -> np.ndarray:
        """Return an error covariance matrix assuming heteroskedasticity (HC0/1)"""
        ee = np.diagflat(self.residuals**2)
        if type == 'HC1':
            ee *= self.n/self.DoF
        return ee

    def _clustered(self, clusters: np.ndarray, finite_sample_correction=True) -> np.ndarray:
        """Return an error covariance matrix assuming clustered errors"""
        ee = bsr_array((self.n, self.n)).toarray()
        for i in range(self.n):
            for j in range(self.n):
                if clusters[i] == clusters[j]:
                    ee[i, j] = self.residuals[i] * self.residuals[j]
        n_cl = len(np.unique(clusters))
        n, k = self.n, self.k
        if finite_sample_correction:
            ee *= ((n - 1) / (n - k)) * (n_cl / (n_cl - 1))
        return ee

class ols_sandwich(OLS, error_covariance):

    def __init__(self, y=None,X=None,residuals=None) -> None:
        """The sandwich class adapts the standard errors of existing OLS regressions"""
        super().__init__(y=y,X=X)

    def _XeeX(self, ee: np.ndarray):
        """Return the meat of the sandwich estimator"""
        return self.X.T.dot(ee).dot(self.X)

    def _sandwich(self, meat: np.ndarray, bread: Optional[np.ndarray] = None) -> None:
        """Helper function to return a 'sandwich' from bread and meat"""
        if bread is None:
            bread = self.var_X_inv
        sandwich = bread.dot(meat).dot(bread)
        beta_se = np.sqrt(np.diag(sandwich))
        return beta_se
    
    def _cluster_robust_fast(self, clusters: np.ndarray):
        """Rather than specifiying the individual errors before calculating the meat, we can instead calculate "mini" meats and sum them up at the end"""
        def cluster_XeeX(cluster_index):
            j = clusters == cluster_index
            _X, _e = self.X[j, :], self.residuals[j]
            _eX = _e.T.dot(_X)
            _XeeX = _eX.T.dot(_eX)
            return _XeeX
        clusters = clusters.flatten()
        cluster_XeeX = [cluster_XeeX(i) for i in np.unique(clusters)]
        n_cl = len(np.unique(clusters))
        n, k = self.n, self.k
        # finite-sample correction factor.    # sum XeeX across all clusters
        XeeX = ((n - 1) / (n - k)) * (n_cl / (n_cl - 1)) * np.sum(cluster_XeeX, axis=0)
        # summed - i.e. requires averaging, so needs many clusters to ensure V is consistent
        # https://cameron.econ.ucdavis.edu/research/Cameron_Miller_JHR_2015_February.pdf (p.7-9)
        return XeeX
    
    def _beta_se(self, se_correction: Optional[str | np.ndarray] = None) -> None:
        """Return the standard errors of the coefficients"""
        self.beta_se, self.conf_int, self.test_stat, self.p_val= None, None, None, None
        if type(se_correction) == np.ndarray:
            ee = self._clustered(se_correction)
        elif type(se_correction) == str:
            if se_correction == "homoskedastic":
                ee = self._homoskedastic()
            elif se_correction == "heteroskedastic":
                ee = self._heteroskedastic(type='HC1')
        else:
            raise ValueError(f"""Correction type {se_correction} not supported. Please choose from 'homoskedastic' or 'heteroskedastic', or supply an array for cluster groups""")        
        XeeX = self._XeeX(ee)
        beta_se = self._sandwich(XeeX)
        return beta_se
    
    def summary(self, z_dist: bool = False, se_correction: Optional[str] = None) -> pd.DataFrame:
        """Returns the coefficients, standard errors, test statistics and p-values in a Pandas DataFrame."""
        self._check_if_fitted()
        self._assess_fit()
        if se_correction is not None:
            self.beta_se = self._beta_se(se_correction)
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