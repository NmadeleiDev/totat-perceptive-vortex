from typing import Union
import numpy as np
import mne
from bokeh.models import ColumnDataSource
from predictor_class import BaseEegPredictor, RawFramesDataPredictor, EpochedDataPredictor
from queue import Queue

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


class BasePredition:
    def read_queue_and_do_predition(self):
        pass

class CliPredition(BasePredition):
    def read_queue_and_do_predition(self, predictor: Union[RawFramesDataPredictor, EpochedDataPredictor], queue: Queue):
        self.y_pred = []
        self.y_true = []
        while True:
            data, y_true = queue.get()
            # print(len(predictions), len(data))

            if data is None:
                break

            prediction_proba = predictor.predict_proba(data)
            prediction = int(prediction_proba[1] > 0.5)

            self.y_pred.append(prediction)
            self.y_true.append(y_true)

            queue.task_done()

    def print_summary(self):
        self.y_pred = np.array(self.y_pred)
        self.y_true = np.array(self.y_true)

        y_true = self.y_true[self.y_true != 2]
        y_pred = self.y_pred[self.y_true != 2]

        print('\n--- Final prediction metrics ---\n')
        for metr in [accuracy_score, f1_score, roc_auc_score]:
            print(f'{metr.__name__}: {metr(y_true, y_pred)}')

class VisualizedPrediction(BasePredition):
    def __init__(self, source_signal: ColumnDataSource, source_projected: ColumnDataSource, source_text: ColumnDataSource, raw: mne.io.Raw) -> None:
        self.full_data = None
        self.source_signal = source_signal
        self.source_projected = source_projected
        self.source_text = source_text

        self.raw = raw
        self.raw_data = raw.get_data()

        self.y_spread = self.raw_data.max() - self.raw_data.min()

        projected_std = 2

        self.signal_shift = np.expand_dims(np.arange(len(source_signal.data['ys'])) * self.y_spread, 1)
        self.proj_shift = np.expand_dims(np.arange(len(source_projected.data['ys'])) * projected_std, 1)
        # self.proj_shift = np.expand_dims(np.arange(len(source_projected.data['ys'])) * self.y_spread, 1)

    def read_queue_and_do_predition(self, predictor: Union[RawFramesDataPredictor, EpochedDataPredictor], queue: Queue):
        data, y_true = queue.get()

        if data is None:
            return

        prediction_proba = predictor.predict_proba(data)
        prediction = int(prediction_proba[1] > 0.5)
        choise_proba = prediction_proba.max()

        if len(data.shape) == 1:
            data_shaped = np.expand_dims(data, 0)
        elif len(data.shape) == 2:
            data_shaped = data
        else:
            raise ValueError(f'data is of unexpected shape: {data.shape}')
        
        raw_source = np.array(self.source_signal.data['ys'])
        data_scaled = data_shaped.T + self.signal_shift
        ys_signal = np.concatenate([raw_source, data_scaled], axis=1)[:, data_shaped.shape[0]:]
        self.source_signal.data['ys'] = ys_signal.tolist()

        dim_reducer = predictor.clf.named_steps['dim_reducer']

        X_projected = dim_reducer.transform(np.expand_dims(predictor.current_frame, 0))
        raw_proj = np.array(self.source_projected.data['ys'])
        proj_scaled = X_projected.T + self.proj_shift
        ys_proj = np.concatenate([raw_proj, proj_scaled], axis=1)[:, X_projected.shape[0]:]
        self.source_projected.data['ys'] = ys_proj.tolist()

        is_rest_text = '(probably resting)' if choise_proba < 0.9 else ''
        self.source_text.data['text'] = [
                f'probability:\t{np.round(choise_proba, 3)}',
                f'y_pred:\t{prediction}\t{is_rest_text}',
                f'y_true:\t{y_true}', 
            ]
        queue.task_done()
        