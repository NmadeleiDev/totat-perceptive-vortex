import argparse
import matplotlib 
from matplotlib import pyplot as plt
from utils import add_generic_arguments_to_parser, fitler_raw, get_runs_for_task, load_data


def main():
    parser = argparse.ArgumentParser(description='Arguments description')
    add_generic_arguments_to_parser(parser)

    args = parser.parse_args()

    task, subjects = args.task, args.subjects
    low_pass, high_pass = args.low_pass, args.high_pass

    raw = load_data(subjects, runs=get_runs_for_task(task), files_base_dir=args.data_files)
    
    raw.plot_psd()

    fitler_raw(raw, low_pass, high_pass)

    raw.plot_psd()


if __name__ == '__main__':
    main()