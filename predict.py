import argparse
from os import path
from typing import Tuple, Union
from threading import Thread
from queue import Queue

from time import sleep
from utils import add_generic_arguments_to_parser, fitler_raw, get_runs_for_task, load_data
import mne
from mne import Epochs, pick_types, events_from_annotations
import numpy as np

from predictor_class import BaseEegPredictor, RawFramesDataPredictor, EpochedDataPredictor

import joblib

from bokeh.plotting import figure, curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, MultiLine, Text

from prediction_writer import VisualizedPrediction, CliPredition


rand_state = 0

def write_data_to_stream(queue: Queue, data: np.ndarray, labels: Union[np.ndarray, None] = None, sleep_secs=1):
    if len(data.shape) == 2:
        data = data.T
    elif len(data.shape) == 3:
        pass
    else:
        raise ValueError(f'Unexpected data shape: {data.shape}')

    for idx in range(len(data)):
        if labels is not None:
            if len(data.shape) == 2:
                if len(labels) > 1 and idx >= labels[1, 0]:
                    labels = labels[1:, :]
                label = labels[0, 2]
            elif len(data.shape) == 3:
                label = labels[idx, 2]
            else:
                raise ValueError(f'Unexpected data shape: {data.shape}')
        else:
            label = 0

        queue.put((data[idx], label))
        if sleep_secs > 0:
            sleep(sleep_secs)

def create_data_stream_framed(raw: mne.io.Raw, add_sleep:bool) -> Queue:
    queue = Queue(256)
    events, _ = events_from_annotations(raw, event_id=dict(T0=2, T1=0, T2=1))

    thr = Thread(
        target=write_data_to_stream, 
        args=(queue, raw.get_data(), events), 
        kwargs={'sleep_secs': 1/raw.info['sfreq'] if add_sleep else 0}, 
        daemon=True)
    thr.start()

    return queue

def create_data_stream_epoched(raw: mne.io.Raw, tmin: float, tmax: float, add_sleep:bool) -> Queue:
    event_id = dict(hands=2, feet=3)

    events, _ = mne.events_from_annotations(raw, event_id=dict(T1=2, T2=3))

    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                    exclude='bads')

    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                    baseline=None, preload=True)
    epochs_train = epochs.copy().crop(tmin=1., tmax=2.)
    labels = epochs.events[:, -1] - 2

    epochs_data_train = epochs_train.get_data()
    
    queue = Queue(256)

    thr = Thread(
        target=write_data_to_stream, 
        args=(queue, epochs_data_train, labels), 
        kwargs={'sleep_secs': 1 if add_sleep else 0},
        daemon=True
        )
    thr.start()

    return queue

def create_chart_and_return_source(raw: mne.io.Raw, n_projected_channels=4) -> Tuple[ColumnDataSource, ColumnDataSource, ColumnDataSource]:
    raw_data = raw.get_data()
    n_channels = raw.info['nchan']

    data_min, data_max = raw_data.min(), raw_data.max()
    y_spread = data_max - data_min

    window_len = int(raw.info['sfreq'] * 5)
    projection_len = int(raw.info['sfreq'])

    projected_min = -2.194366353088722
    projected_max =  2.3118573501771285
    # projected_std = projected_max - projected_min
    projected_std = 2

    source_signal = ColumnDataSource(
        data=dict(
            xs=[np.arange(window_len)] * n_channels, 
            ys=[np.zeros((window_len, )) + data_min for i in range(n_channels)] ,
            ))

    source_projected = ColumnDataSource(
        data=dict(
            xs=[np.arange(projection_len)] * n_projected_channels, 
            ys=[np.zeros((projection_len, )) + projected_min for i in range(n_projected_channels)],
            ))

    text_lables = ['y_true', 'y_pred', 'predict_proba']
    source_text = ColumnDataSource(
        data=dict(
            x=[5] * len(text_lables),
            y=np.linspace(projected_std * n_projected_channels - len(text_lables) / 2, projected_std * n_projected_channels, len(text_lables)),
            text=text_lables
        )
    )

    plot_signal = figure(height=1400, width=700, title="EEG Signal",
                tools="crosshair,pan,reset,save,wheel_zoom",
                x_range=[0, window_len], y_range=[data_min, data_min + y_spread * n_channels])

    # plot_signal.yaxis.visible = False
    plot_signal.add_glyph(source_signal, MultiLine(xs="xs", ys="ys", line_width=1, line_alpha=0.6))


    plot_projection = figure(height=400, width=600, title="Projected features",
                tools="crosshair,pan,reset,save,wheel_zoom",
                x_range=[0, projection_len], y_range=[projected_min, projected_min + len(text_lables) + projected_std * n_projected_channels])
                # x_range=[0, projection_len], y_range=[data_min, data_min + y_spread * n_projected_channels])

    # plot_projection.yaxis.visible = False
    plot_projection.add_glyph(source_projected, MultiLine(xs="xs", ys="ys", line_width=1, line_alpha=0.6))

    plot_projection.add_glyph(source_text, Text(x="x", y="y", text="text"))

    curdoc().add_root(row(plot_signal, plot_projection, width=1300))
    curdoc().title = "EEG"

    return source_signal, source_projected, source_text

def main(intervactive:bool):
    parser = argparse.ArgumentParser(description='Arguments description')
    parser.add_argument("-p", "--save-path", required=False, default='./', type=str)

    add_generic_arguments_to_parser(parser)

    args = parser.parse_args()

    task, subjects = args.task, args.subjects
    low_pass, high_pass = args.low_pass, args.high_pass
    save_path = args.save_path

    runs = get_runs_for_task(task)

    print(f'Subjects used: {", ".join(map(str, subjects))}')
    print(f'Runs used: {", ".join(map(str, runs))}')
    print(f'Interactive: {intervactive}')

    raw = load_data(subjects, runs=runs, files_base_dir=args.data_files)

    fitler_raw(raw, low_pass, high_pass)

    trained_clf = joblib.load(path.join(save_path, 'eeg_classifier.joblib'))

    queue = create_data_stream_framed(raw, intervactive)
    predictor = RawFramesDataPredictor(trained_clf, int(raw.info['sfreq']))

    if intervactive:
        source_signal, source_projected, source_text = create_chart_and_return_source(raw)
        predict_out = VisualizedPrediction(source_signal, source_projected, source_text, raw=raw)

        curdoc().add_periodic_callback(lambda : predict_out.read_queue_and_do_predition(predictor, queue), int(500/raw.info['sfreq']))
    else:

        predict_out = CliPredition()
        Thread(target=predict_out.read_queue_and_do_predition, args=(predictor, queue), daemon=True).start()
        
        queue.join()

        predict_out.print_summary()
    
if __name__ == '__main__':
    main(False)
else:
    main(True)
