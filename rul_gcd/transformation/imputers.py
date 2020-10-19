import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator

class NaNRemovalImputer(BaseEstimator, TransformerMixin):
    def fit(self, X):
        return self

    def transform(self, X):
        mask = np.all(~np.isfinite(X), axis=1)
        return X[~mask]

class RollingImputer(BaseEstimator, TransformerMixin):
    def __init__(self, window_size, func):
        self.window_size = window_size
        self.function = func
    
    def fit(self, X):
        self.default_value = np.mean(X,  axis=0)
        self.default_value[~np.isfinite(self.default_value)] = 0
        return self
    
    def transform(self, X):
        X = X.copy()
        row, features = np.where(~np.isfinite(X))
        min_limit = np.maximum(row - self.window_size, 0)
        max_limit = np.minimum(row + self.window_size, X.shape[0])
        for r, min_r, max_r, f in zip(row, min_limit, max_limit, features):            
            X[r, f] = self.function(X[min_r:max_r, f])
            if ~np.isfinite(X[r, f]):
                X[r, f] = self.default_value[f]
        return X


class RollingMedianImputer(RollingImputer):
    def __init__(self, window_size):
        super().__init__(window_size, np.median)


class RollingMeanImputer(RollingImputer):
    def __init__(self, window_size):
        super().__init__(window_size, np.mean)


class ForwardFillImputer(BaseEstimator, TransformerMixin):
    def fit(self, X):
        return self

    def transform(self, X):
        mask = np.isnan(X)
        idx = np.where(~mask,np.arange(mask.shape[1]),0)
        np.maximum.accumulate(idx,axis=1, out=idx)
        X[mask] = X[np.nonzero(mask)[0], idx[mask]]
        return X
