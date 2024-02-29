import argparse
import os
from pathlib import Path
from arrow.common.wb_logging import log_local_runs


def main() -> None:
    # Take a single argument, the path to a directory
    parser = argparse.ArgumentParser(description='Log local runs to W&B')
    parser.add_argument('-f',
                        '--path',
                        type=str,
                        default=None,
                        help='The path to the directory containing the data to log.')
    args = parser.parse_args()
    # Print the arguments
    print(args)
    # Make sure the path exists
    if args.path is not None:
        assert os.path.isdir(args.path), "The path provided is not a directory."

    # If the path is not provided, use the current directory
    if args.path is None:
        args.path = os.getcwd()

    # Log the local runs
    log_local_runs(Path(args.path))


# If this is the main file, run the main function
if __name__ == '__main__':
    main()
