import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import linalg
from sklearn.base import BaseEstimator, TransformerMixin


class CspTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_components) -> None:
        self.n_components = n_components
        self.X_projected = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes, self.class_weight = np.unique(y, return_counts=True)

        classes = np.unique(y)
        if len(classes) > 2:
            raise ValueError(f'CspTransformer suppports only 2 classes, but got ({len(classes)})')
        class_1, class_2 = classes

        X_1 = np.concatenate(X[y == class_1], axis=1)
        X_2 = np.concatenate(X[y == class_2], axis=1)

        X_1 -= X_1.mean(1, keepdims=True)
        X_2 -= X_2.mean(1, keepdims=True)

        r1, r2 = np.dot(X_1, X_1.T) / X_1.shape[1], np.dot(X_2, X_2.T) / X_2.shape[1]

        eigenvals, eigenvecs = linalg.eigh(r1, r1 + r2)
        ix = np.argsort(np.abs(eigenvals - 0.5))[::-1]

        self._ix = ix
        
        self._eigenvals = eigenvals[ix]
        
        self._filters = eigenvecs[:, ix].T

        self.applied_filters = self._filters[:self.n_components]

        return self

    def transform(self, X: np.ndarray, y=None):
        X_transformed = np.stack([np.dot(self.applied_filters, epoch) for epoch in X])
        X_transformed = (X_transformed ** 2).mean(2)
        X_transformed = np.log(X_transformed)
        
        self.X_projected = X_transformed

        return X_transformed
