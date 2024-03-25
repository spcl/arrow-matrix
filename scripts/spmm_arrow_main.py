import argparse
import os
from mpi4py import MPI
from arrow import arrow_bench
from arrow.common import utils


def main() -> None:

    parser = argparse.ArgumentParser(description='Benchmark the SpMM')
    parser.add_argument('-f', '--path', type=str, default=None,
                        help='The filename prefix of the decomposed graph. If none, synthetic data is generated.')
    parser.add_argument('-w', '--width', type=int, default=0, help='Width of the decomposition / Height of the blocks.')
    parser.add_argument('-c', '--features', type=int, default=16,
                        help='Width of the decomposition / Height of the blocks.')
    parser.add_argument('-b', '--blocked', type=utils.str2bool, nargs="?", default=True,
                        help='If true, the matrix has only one block diagonal,')
    parser.add_argument('-i', '--device', type=str, default='gpu', help='Device to use for the MM. Either cpu or gpu.')
    parser.add_argument('-z', '--iterations', type=int, default=1, help='Number of SpMM iteration to run.')
    parser.add_argument('-r', '--ranksperside', type=int, default=3,
                        help='Number of Ranks per Side (For synthetic data only)')
    parser.add_argument('-m', '--ba_neighbors', type=int, default=3,
                        help='Number of neighbors per bertex (For synthetic data only)')
    # Add slim argument
    parser.add_argument('-s', '--slim', type=utils.str2bool, nargs="?", default=True,
                        help='If true, the decomposition onto ranks is "slim" assigning one rank per row-block.')
    # Add npy argument
    parser.add_argument('-n', '--npy', type=utils.str2bool, nargs="?", default=True,
                        help='If true, the decomposition is loaded from the indices / indptr files.')

    args = vars(parser.parse_args())
    utils.mpi_print(MPI.COMM_WORLD.Get_rank(), str(args))

    try:
        args['wandb_key'] = os.environ['WANDB_API_KEY']
    except KeyError:
        utils.mpi_print(MPI.COMM_WORLD.Get_rank(),
                        "Set the WANDB_API_KEY environment variable to start logging results to Weights & Biases.")
        args['wandb_key'] = None

    arrow_bench.bench_spmm(args['path'],
                           args['width'],
                           args['features'],
                           args['iterations'],
                           args['blocked'],
                           args['device'],
                           args['ranksperside'],
                           args['ba_neighbors'],
                           args['wandb_key'],
                           slim=args['slim'],
                           npy_format=args['npy'])


if __name__ == '__main__':
    main()
