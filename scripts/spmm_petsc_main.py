import argparse
import os
import numpy as np
from arrow.common import utils
from arrow.baseline.spmm_petsc import benchmark_spmm


def main() -> None:

    parser = argparse.ArgumentParser(description='SpMM PETSc benchmark.')

    parser.add_argument('-s',
                        '--seed',
                        type=int,
                        nargs="?",
                        default=42,
                        help='The seed for the random number generator.')
    parser.add_argument('-t',
                        '--type',
                        nargs="?",
                        choices=['float32', 'float64'],
                        default='float32',
                        help='The type of the data.')
    parser.add_argument('-f',
                        '--file',
                        type=str,
                        nargs="?",
                        default=None,
                        help='The file containing a slice of the sparse matrix. Of the form {name}.part.{x}.slice.{y}.npz for a partition into x parts (any y)')
    parser.add_argument('-c',
                        '--columns',
                        type=int,
                        nargs="?",
                        default=32,
                        help='The number of columns in the matrix X.')
    parser.add_argument('-i', '--device', type=str, default='gpu', help="Either gpu or cpu.")
    parser.add_argument('-z', '--iterations', type=int, default=3, help="Number of iterations to benchmark.")
    parser.add_argument('--gpu-tiling', type=utils.str2bool, nargs="?", default=False, help='Use GPU tiling.')
    # Add a dryrun argument to test the script without running the benchmark
    parser.add_argument('--dryrun', type=utils.str2bool, nargs="?", default=False, help='Run a dryrun (no benchmark).')
    parser.add_argument('-m', '--memory', type=float, default=0.9, help='The fraction of GPU memory to use.')

    args = vars(parser.parse_args())

    # Get the wandb_key from the environment
    try:
        args['wandb_key'] = os.environ['WANDB_API_KEY']
    except KeyError:
        print("Set the WANDB_API_KEY environment variable to start logging results to Weights & Biases.")
        args['wandb_key'] = None

    if args['file'] == "None":
        args['file'] = None

    rng = np.random.default_rng(args['seed'])
    dtype = np.dtype(args['type'])
    k = args['columns']
    iterations = args['iterations']
    device = args['device']
    wandb_api_key = args['wandb_key']
    file = args['file']
    mem_fraction = args['memory']

    benchmark_spmm(file,
                   k,
                   iterations,
                   device,
                   wandb_api_key,
                   dtype,
                   rng,
                   args['gpu_tiling'],
                   args['dryrun'],
                   mem_fraction)


if __name__ == "__main__":
    main()
