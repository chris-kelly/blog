from typing import TypeAlias, Optional, Tuple, List, Literal
import numpy as np, pandas as pd

pd_np_ls: TypeAlias = pd.DataFrame | pd.Series | np.ndarray | List
panel: TypeAlias = Literal['pooled', 'within', 'fixed', 'fe', 'random', 're', 'between', 'be', 'first_difference', 'fd']

class data_transformation:
    
    def _convert_types(self, z: pd_np_ls) -> np.ndarray:
        """ Convert z to numpy array and reshape to work nicely with rest of functions """
        if type(z) in [pd.DataFrame, pd.Series]:
            z2 = z.to_numpy()
        if type(z) == list:
            z2 = np.array(z)
        if type(z) == np.ndarray:
            z2 = z
        else:
            raise TypeError('Array must be a pandas series/dataframe, numpy array or list')
        return z2
    
    def _get_y(self, y: Optional[pd_np_ls] = None) -> np.ndarray:
        """Reshape and convert y to numpy array to work nicely with rest of functions"""
        if y is None:
            y = self.y
        else:
            y = self._convert_types(y).reshape(-1)
        return y

    def _get_X(self, X: Optional[pd_np_ls] = None, intercept: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Reshape and convert X to numpy array. Also return names for summarising in coefficient table"""
        if X is None:
            X = self.X
        else:
            if type(X) == pd.DataFrame:
                exog_names = np.array(X.columns)
            elif type(X) == pd.Series | type(X) == list:
                exog_names = np.array(['Unnamed Exog Feature'])
            else:
                exog_names = np.arange(X.shape[1])
            X = self._convert_types(X).reshape(-1,len(exog_names))
            if intercept:
                X = np.c_[np.ones(X.shape[0]), X]
                exog_names = np.insert(exog_names, 0, 'Intercept')
            if np.linalg.matrix_rank(X) < X.shape[1]:
                raise ValueError("X is rank deficient.")
        return (X, exog_names)

    def _weight_transform(self, z: pd_np_ls, rho: Optional[pd_np_ls]=None) -> Optional[np.ndarray]:
        """Transforms the weights to a numpy array"""
        z2 = z.copy()
        if rho is not None:
            self.rho = self._convert_types(rho)
        if self.rho is not None:
            z2 = self.rho.dot(z2)
        return z2

    def _clear_fitted_attributes(self):
        """Clears fitted attributes of the class"""
        self.beta, self.RSS, self.beta_se, self.conf_int, self.test_stat, self.p_val = None, None, None, None, None, None
    
    def __init__(
        self, 
        y: Optional[pd_np_ls], 
        X: Optional[pd_np_ls],
        intercept: Optional[bool] = True,
        rho: Optional[pd_np_ls] = None,
        # panel: Optional[panel] = None,
        ) -> None:
        """Initializes the OLS class to run an least-squares regression"""
        if y is not None:
            self.y = self._get_y(y)
            self.n = len(self.y)
            self._y = self._weight_transform(self.y, rho)
        if X is not None:
            self.X, self.exog_names = self._get_X(X=X,intercept=intercept)
            self.k = self.X.shape[1]
            self._X = self._weight_transform(self.X, rho)
        if y is not None and X is not None and len(self.y) != self.X.shape[0]:
            raise ValueError("y and X must be the same size.")    
        self._clear_fitted_attributes()

    

class ls_fit():
        
    def __init__(
        self, 
        y: Optional[pd_np_ls], 
        X: Optional[pd_np_ls],
        intercept: Optional[bool] = True,
        rho: Optional[pd_np_ls] = None,
        # panel: Optional[panel] = None,
        ) -> None:
        super().__init__(y, X, intercept, rho)

    def _quick_matrix_invert(self, X: np.ndarray) -> np.ndarray:
        """ Find the inverse of a matrix, using QR factorization """
        Q, R = qr(X)
        X_inv = solve_triangular(R, np.identity(X.shape[1])).dot(Q.transpose())
        return X_inv

    def _estimate_ols_coefs(
        self,
        y: Optional[np.ndarray] = None,
        X: Optional[np.ndarray] = None
    ):
        """Estimates the LS coefficients given a vector y and matrix X"""
        XTX = X.T.dot(X)
        XTY = X.T.dot(y)
        XTX_inv = self._quick_matrix_invert(XTX)
        coefs = XTX_inv.dot(XTY)
        return coefs, XTX_inv

    def fit(
        self,
        y: Optional[np.ndarray] = None,
        X: Optional[np.ndarray] = None,
        intercept: Optional[bool] = True,
        rho: Optional[pd_np_ls] = None,
    ):
        """Estimates the OLS coefficients given a vector y and matrix X"""
        super().__init__(y, X, intercept, rho)
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