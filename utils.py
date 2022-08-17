import argparse
from typing import List
import mne
from mne import concatenate_raws
from mne.channels import make_standard_montage
from mne.datasets import eegbci

from os import path

def load_data(subjects: list[int], runs: list[int], files_base_dir='./files') -> mne.io.Raw:
    raws = []
    for subject in subjects:
        s_str = str(subject).zfill(3)
        for run in runs:
            r_str = str(run).zfill(2)
            raws.append(mne.io.read_raw_edf(path.join(files_base_dir, f'S{s_str}', f'S{s_str}R{r_str}.edf'), preload=True))

    result = concatenate_raws(raws)
    eegbci.standardize(result)
    montage = make_standard_montage('standard_1005')
    result.set_montage(montage)
    return result

def fitler_raw(raw, low_pass: float, high_pass: float):
    raw.filter(low_pass, high_pass, fir_design='firwin', skip_by_annotation='edge')

def add_generic_arguments_to_parser(parser: argparse.ArgumentParser):
    parser.add_argument("-t", "--task", required=False, default=4, type=int, choices=[1, 2, 3, 4])
    parser.add_argument("-s", "--subjects", required=False, nargs='+', default=[1])

    parser.add_argument("--low-pass", required=False, default=7., type=float)
    parser.add_argument("--high-pass", required=False, default=30., type=float)
    
    parser.add_argument("-d", "--data-files", required=False, default='./files', type=str)

def get_runs_for_task(task_id: int) -> List[int]:
    step = 4
    n_runs = 3
    start_idx = 2
    return list(range(task_id + start_idx, step * n_runs + 1 + start_idx, step))