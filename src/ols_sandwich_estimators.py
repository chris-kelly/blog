from ols_blue import OLS
from scipy.sparse import bsr_array
import pandas as pd, numpy as np
from typing import Optional, Tuple

class sandwich(OLS):

    def __init__(self, y=None,X=None) -> None:
        """The sandwich class adapts the standard errors of existing OLS regressions"""
        super().__init__(y=y,X=X)

    def _homoskedastic(self) -> pd.DataFrame:
        """Return an error covariance matrix assuming homoskedasticity"""
        e = self.residuals
        ee = np.identity(self.n) *  e.T.dot(e) / float(self.DoF)
        return ee

    def _heteroskedastic(self) -> np.ndarray:
        """Return an error covariance matrix assuming heteroskedasticity (HC0)"""
        ee = np.diagflat(self.residuals**2)
        return ee

    def _clustered(self, clusters: np.ndarray) -> np.ndarray:
        ee = bsr_array((self.n, self.n)).toarray()
        for i in range(self.n):
            for j in range(self.n):
                if clusters[i] == clusters[j]:
                    ee[i, j] = self.residuals[i] * self.residuals[j]
        return ee

    def _XeeX(self, ee: np.ndarray):
        """Return the meat of the sandwich estimator"""
        return self.X.T.dot(ee).dot(self.X)

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

    def _sandwich(self, meat: np.ndarray, bread: Optional[np.ndarray] = None) -> None:
        """Helper function to return a 'sandwich' from bread and meat"""
        if bread is None:
            bread = self.var_X_inv
        sandwich = bread.dot(meat).dot(bread)
        self.beta_se = np.sqrt(np.diag(sandwich))

    def summary(self, z_dist: bool = False, correction: Optional[str] = None) -> pd.DataFrame:
        """Returns the coefficients, standard errors, test statistics and p-values in a Pandas DataFrame."""
        self._check_if_fitted()
        self._assess_fit()
        if correction is not None:
            self.beta_se, conf_int, test_stat, p_val= None, None, None, None
            err_message = f"""Correction type {correction} not supported. Please choose from 'homoskedastic' or 'heteroskedastic', or supply an array for cluster groups"""
            if type(correction) == str:
                if correction == "homoskedastic":
                    ee = self._homoskedastic()
                elif correction == "heteroskedastic":
                    ee = self._heteroskedastic()
                else:
                    raise ValueError(err_message)
                XeeX = self._XeeX(ee)
            elif type(correction) == np.ndarray:
                XeeX = self._cluster_robust_fast(correction)
            else:
                raise ValueError(err_message)
            self._sandwich(XeeX)
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