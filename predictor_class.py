import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline

class BaseEegPredictor(BaseEstimator, ClassifierMixin):
    def __init__(self, classifier_pipeline: Pipeline):
        self.clf = classifier_pipeline

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.clf.fit(X, y)

        return self

    def predict(self, X: np.ndarray):
        return self.clf.predict(X)

class EpochedDataPredictor(BaseEegPredictor):
    def get_right_shape_input(self, X: np.ndarray):
        if len(X.shape) == 2:
            inp = np.expand_dims(X, 0)

        elif len(X.shape) == 3:
            inp = X
        else:
            raise ValueError(f'Unexpected X shape: {X.shape}')

        return inp

    def predict(self, X: np.ndarray):
        return self.clf.predict(self.get_right_shape_input(X)).squeeze()

    def predict_proba(self, X: np.ndarray):
        return self.clf.predict_proba(self.get_right_shape_input(X)).squeeze()

class RawFramesDataPredictor(EpochedDataPredictor):
    def __init__(self, classifier_pipeline: Pipeline, frames_per_epoch: int):
        super().__init__(classifier_pipeline)
        self.current_frame = None
        self.frames_per_epoch = frames_per_epoch

    def update_current_frame(self, X: np.ndarray):
        if len(X.shape) != 1:
            raise ValueError(f'Unexpected X shape: {X.shape}')

        X_frame = np.expand_dims(X, 1)

        if self.current_frame is None:
            self.current_frame = X_frame
        elif self.current_frame.shape[1] < self.frames_per_epoch:
            self.current_frame = np.concatenate([self.current_frame, X_frame], axis=1)
        else:
            self.current_frame = np.concatenate([self.current_frame[:, 1:], X_frame], axis=1)

    def predict(self, X: np.ndarray):
        self.update_current_frame(X)
        return super().predict(self.current_frame)

    def predict_proba(self, X: np.ndarray):
        self.update_current_frame(X)
        return super().predict_proba(self.current_frame)

