import argparse
import pandas as pd
from train import train_pipeline
from utils import add_generic_arguments_to_parser, fitler_raw, get_runs_for_task, load_data


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

    results = pd.DataFrame()

    for task_id in range(1, 5):
        for subject_id in range(1, 110):
            runs = get_runs_for_task(task_id)
            print(f'Subject: {subject_id}, task: {task_id}')

            raw = load_data([subject_id], runs=runs, files_base_dir=args.data_files)

            fitler_raw(raw, low_pass, high_pass)

            _, scores = train_pipeline(raw, num_components, t_min, t_max, model_type, use_custom_csp=use_my_csp, print_metrics=False)
            results = pd.concat([results, pd.Series(scores).to_frame().T], axis=0, ignore_index=True)

    print('--- Total mean scores ---')
    print(results.mean(axis=0).to_dict())

if __name__ == '__main__':
    main()