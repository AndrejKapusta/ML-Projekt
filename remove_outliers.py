import numpy as np


# Trieda pre normalizaciu atributu std_glucose a kurtosis_oxygen
class remove:

    def __init__(self, missing_value=np.nan):
        self.missing_value = missing_value
        self.percentile_r = 0
        self.percentile_l = 0
        self.whisker_r = 0
        self.whisker_l = 0

    def _get_mask(self, X, value_to_mask):
        if np.isnan(value_to_mask):
            return np.isnan(X)
        else:
            return np.equal(X, value_to_mask)

    def fit(self, X, y=None):
        mask = self._get_mask(X, self.missing_value)
        self.percentile_r = np.percentile(X[~mask], 95)
        self.percentile_l = np.percentile(X[~mask], 5)
        descr = X.describe()
        self.whisker_r = np.min([descr['max'], descr['75%'] + (1.5 * (descr['75%'] - descr['25%']))])
        self.whisker_l = np.max([descr['min'], descr['25%'] - (1.5 * (descr['75%'] - descr['25%']))])
        return self

    def transform(self, X):
        mask = (X > self.whisker_r)
        X[mask] = self.percentile_r
        mask = (X < self.whisker_l)
        X[mask] = self.percentile_l
        return X
