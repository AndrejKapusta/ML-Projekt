import numpy as np

# Doplnac numerickych hodnot na zaklade medianu
class imputer():
    def __init__(self, missing_value=np.nan):
        self.missing_value = missing_value
        self.median = 0
    
    
    def _get_mask(self, X, value_to_mask):
        if np.isnan(value_to_mask):
            return np.isnan(X)
        else:
            return np.equal(X, value_to_mask)
    
    
    def fit(self, X, y=None):
        mask = self._get_mask(X, self.missing_value)
        self.median = np.median(X[~mask])
        return self
        
    
    def transform(self, X):
        mask = self._get_mask(X, self.missing_value)
        X[mask] = self.median
        
        return X