import argparse
from os import path
import matplotlib 
from matplotlib import pyplot as plt
from utils import add_generic_arguments_to_parser, fitler_raw, get_runs_for_task, load_data
import mne
import numpy as np

from csp import CspTransformer
from kneighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.pipeline import Pipeline
from sklearn.model_selection import ShuffleSplit, cross_validate

import joblib

rand_state = 0

def train_pipeline(raw: mne.io.Raw, num_components: int, tmin:float, tmax:float, model_type='lda', use_custom_csp=True):
    event_id = dict(hands=2, feet=3)

    events, _ = mne.events_from_annotations(raw, event_id=dict(T1=2, T2=3))

    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                    exclude='bads')

    epochs_train = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                    baseline=None, preload=True)
    labels = epochs_train.events[:, -1] - 2

    scores = []
    epochs_data_train = epochs_train.get_data()
    cv = ShuffleSplit(10, test_size=0.2, random_state=rand_state)

    if model_type == 'lda':
        predictor = LinearDiscriminantAnalysis()
    else:
        predictor = KNeighborsClassifier()

    if use_custom_csp:
        dim_reducer = CspTransformer(num_components)
    else:
        dim_reducer = mne.decoding.CSP(num_components)

    clf = Pipeline([('dim_reducer', dim_reducer), ('predictor', predictor)])

    scorers = ['f1', 'accuracy']
    scores = cross_validate(clf, epochs_data_train, labels, scoring=scorers, n_jobs=1, cv=cv)

    class_balance = np.mean(labels == labels[0])
    class_balance = max(class_balance, 1. - class_balance)

    print(f'\n--- Training results --- / class_balance={class_balance}\n')
    for scorer in scorers:
        scorer_key = f'test_{scorer}'
        print(f'{scorer}:\t{np.mean(scores[scorer_key])}')

    return clf.fit(epochs_data_train, labels)

def main():
    parser = argparse.ArgumentParser(description='Arguments description')

    parser.add_argument("--t-min", required=False, default=1., type=float)
    parser.add_argument("--t-max", required=False, default=2., type=float)
    parser.add_argument("-c", "--num-components", required=False, default=4, type=int)
    parser.add_argument("-m", "--model", required=False, default='lda', type=str, choices=['lda', 'kneighbors'])
    parser.add_argument("-r", "--reducer", required=False, default='my', type=str, choices=['my', 'mne'])
    
    parser.add_argument("-p", "--save-path", required=False, default='./', type=str)

    add_generic_arguments_to_parser(parser)

    args = parser.parse_args()

    task, subjects = args.task, args.subjects
    low_pass, high_pass = args.low_pass, args.high_pass
    t_min, t_max = args.t_min, args.t_max
    num_components = args.num_components
    model_type = args.model
    save_path = args.save_path

    use_my_csp = args.reducer == 'my'

    runs = get_runs_for_task(task)
    print(f'Subjects used: {", ".join(map(str, subjects))}')
    print(f'Runs used: {", ".join(map(str, runs))}')

    raw = load_data(subjects, runs=runs, files_base_dir=args.data_files)

    fitler_raw(raw, low_pass, high_pass)

    trained_clf = train_pipeline(raw, num_components, t_min, t_max, model_type, use_custom_csp=use_my_csp)

    joblib.dump(trained_clf, path.join(save_path, 'eeg_classifier.joblib'))
    
if __name__ == '__main__':
    main()