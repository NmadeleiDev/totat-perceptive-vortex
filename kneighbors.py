from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.stats import mode
import numpy as np

class KNeighborsClassifier(BaseEstimator, ClassifierMixin):
    weights_enum = ['uniform', 'distance']
    distance_enum = ['manhattan', 'euclidian']
    def __init__(self, k: int = 5, weights='uniform', distance='euclidian') -> None:
        self.k = k
        self.weights = weights
        self.distance = distance

        self._process_args()

    def _process_args(self):
        if self.weights not in self.weights_enum:
            raise ValueError(f'weights must be one of {self.weights_enum}')

        if self.distance not in self.distance_enum:
            raise ValueError(f'distance must be one of {self.distance_enum}')

    def _calc_distance(self, X: np.ndarray, sample: np.ndarray) -> np.ndarray:
        if self.distance == 'euclidian':
            return np.sqrt(np.sum((X - sample) ** 2, axis=1))
        elif self.distance == 'manhattan':
            return np.sum(np.abs(X - sample), axis=1)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        result = np.array([mode(self.y[np.argsort(self._calc_distance(self.X, sample))[:self.k]], keepdims=False)[0] for sample in X])

        return result

    def unique_counts_to_probs(self, uniq:np.ndarray, counts:np.ndarray):
        if len(uniq) == 1:
            if uniq[0] == 0:
                return [1, 0]
            else:
                return [0, 1]
        else:
            return counts / self.k

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        result = np.array([self.unique_counts_to_probs(*np.unique(self.y[np.argsort(self._calc_distance(self.X, sample))[:self.k]], return_counts=True)) for sample in X])
        return result