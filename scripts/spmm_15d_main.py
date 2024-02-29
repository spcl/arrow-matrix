import argparse
import math
try:
    import cupy as cp
except ImportError:
    cp = None

import numpy as np
import os

from mpi4py import MPI
from scipy import sparse
from timeit import repeat

from arrow.common import wb_logging, utils
from arrow.baseline.spmm_15d import spmm_15d_cpu, spmm_15d_gpu
from arrow.baseline.spmm_15d import generate_15d_decomposition, generate_15d_decomposition_new


def main():
    parser = argparse.ArgumentParser(description='SpMM15D benchmark.')
    parser.add_argument('-d',
                        '--dataset',
                        nargs="?",
                        choices=['random', 'file'],
                        default='random',
                        help='The source of the sparse matrix.')
    parser.add_argument('-s',
                        '--seed',
                        type=int,
                        nargs="?",
                        default=42,
                        help='The seed for the random number generator.')
    parser.add_argument('-v',
                        '--vertices',
                        type=int,
                        nargs="?",
                        default=100000,
                        help='The number of vertices in the graph.')
    parser.add_argument('-e', '--edges', type=int, nargs="?", default=1000000, help='The number of edges in the graph.')
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
                        help='The file containing the sparse matrix.')
    parser.add_argument('-c',
                        '--columns',
                        type=int,
                        nargs="?",
                        default=128,
                        help='The number of columns in the matrix X.')
    parser.add_argument('-r',
                        '--replication',
                        type=int,
                        nargs="?",
                        default=0,
                        help='The replication factor for the A-stationary 1.5D decomposition. If 0, the largest possible value is used.')
    parser.add_argument('--validate', type=utils.str2bool, nargs="?", default=True, help='Validate the result.')
    parser.add_argument('-i', '--device', type=str, default='gpu', help="Either gpu or cpu.")
    parser.add_argument('-z', '--iterations', type=int, default=10, help="Number of iterations to benchmark.")
    parser.add_argument('--gpu-tiling', type=utils.str2bool, nargs="?", default=False, help='Use GPU tiling.')
    parser.add_argument('-y', '--decomposition', type=str, default='old', help='old or new decomposition.')

    args = vars(parser.parse_args())

    try:
        args['wandb_key'] = os.environ['WANDB_API_KEY']
    except KeyError:
        utils.mpi_print(MPI.COMM_WORLD.Get_rank(),
                        "Set the WANDB_API_KEY environment variable to start logging results to Weights & Biases.")
        args['wandb_key'] = None

    rng = np.random.default_rng(args['seed'])
    dtype = np.dtype(args['type'])

    world_comm = MPI.COMM_WORLD
    world_size = world_comm.Get_size()
    world_rank = world_comm.Get_rank()

    if args['replication'] == 0:
        def largest_power_of_two_square_constant_time(x):
            """Compute the largest power of two whose square is at most x using a constant time algorithm."""
            n = math.floor(math.log2(x) / 2)
            return 2 ** n


        # Set to the next lowest power of two whose square is less than or equal to the number of processes.
        args['replication'] = largest_power_of_two_square_constant_time(world_size)
        utils.mpi_print(world_rank, f"Using replication factor {args['replication']}")

    # A matrix
    A = None
    if args['dataset'] == 'file':
        if args['file'] is None:
            utils.mpi_print(world_rank, "Please specify the file contaning the adjacency matrix.")
            exit(1)
        absolute_path = os.path.abspath(args['file'])
        if args['decomposition'] == 'old' and not os.path.exists(absolute_path):
            utils.mpi_print(world_rank, f"The file {args['file']} does not exist.")
            exit(1)
        folder, filename = os.path.split(absolute_path)
        if args['decomposition'] == 'old' and not filename.endswith('.npz'):
            utils.mpi_print(world_rank, f"The file {args['file']} is not a .npz file.")
            exit(1)
        utils.mpi_print(world_rank, f"Loading adjacency matrix from {args['file']}...")
        if world_rank == 0:
            if args['decomposition'] != 'old':
                indptr = np.lib.format.open_memmap(f'{absolute_path}_indptr.npy', mode='r')
                indices = np.lib.format.open_memmap(f'{absolute_path}_indices.npy', mode='r')
                data = np.lib.format.open_memmap(f'{absolute_path}_data.npy', mode='r')
                A = (data, indices, indptr)
            else:
                A = sparse.load_npz(absolute_path)
                if A.dtype != dtype:
                    utils.mpi_print(world_rank, f"Converting matrix from {A.dtype} to {dtype}...")
                    A = A.astype(dtype)
    elif args['dataset'] == 'random':
        utils.mpi_print(
            world_rank,
            f"Generating random adjacency matrix with {args['vertices']} vertices and {args['edges']} edges...")
        if world_rank == 0:
            A = utils.generate_sparse_matrix(args['vertices'], args['vertices'], args['edges'], dtype, rng)
    else:
        raise NotImplementedError

    rfactor = args['replication']

    decomp_func = generate_15d_decomposition
    if args['decomposition'] != 'old':
        decomp_func = generate_15d_decomposition_new
    lA, lX, lY, cart_comm, bcast_comm, reduce_comm, max_lNKb = decomp_func(
        A, args['columns'], dtype, rfactor, rng)

    gpu_block_size = None
    if args['device'] == 'gpu' and args['gpu_tiling']:
        available_memory = 0.47 * cp.cuda.Device(0).mem_info[0]
        lA_memory = np.amax([block.data.nbytes + block.indptr.nbytes + block.indices.nbytes for block in lA])
        if lA_memory > available_memory:
            raise ValueError(f"At least one A-block does not fit in GPU memory. Available: {available_memory}, "
                             f"required: {lA_memory}")
        remaining_memory = available_memory - lA_memory
        max_cols = int(remaining_memory // (4 * max_lNKb * dtype.itemsize))
        if max_cols < lX.shape[1]:
            gpu_block_size = max_cols
        utils.mpi_print(world_rank, f"Available memory: {available_memory}, A-block memory: {lA_memory}, "
                                    f"GPU block size: {gpu_block_size}")

    # Validation
    if args['validate']:
        utils.mpi_print(world_rank, "Validating the result...")
        Px, Py = world_size // rfactor, rfactor
        bcast_rank = bcast_comm.Get_rank()
        reduce_rank = reduce_comm.Get_rank()

        # Gather X
        if world_rank == 0:
            if args['decomposition'] != 'old':
                A = sparse.csr_matrix(A, dtype=dtype)
            X = np.empty((A.shape[1], args['columns']), dtype)
            X[:lX.shape[0]] = lX
            idx = lX.shape[0]
            for r in range(Py, world_size, Py):
                start = idx
                end = min(X.shape[0], idx + lX.shape[0])
                world_comm.Recv(X[start:end], source=r)
                idx = end
        elif world_rank % Py == 0:
            assert reduce_rank == 0
            world_comm.Send(lX, dest=0)

        world_comm.Barrier()

        if world_rank == 0:
            ref = A @ X

        spmm_15d_cpu(lA, lX, lY, cart_comm, bcast_comm, reduce_comm)

        # Gather Y
        if world_rank == 0:
            Y = np.zeros((A.shape[0], args['columns']), dtype)
            Y[:lY.shape[0]] = lY
            idx = lY.shape[0]
            for r in range(Py, world_size, Py):
                start = idx
                end = min(Y.shape[0], idx + lY.shape[0])
                world_comm.Recv(Y[start:end], source=r)
                idx = end
            utils.mpi_print(
                world_rank, f"CPU validation: {np.allclose(Y, ref)} "
                            f"({np.linalg.norm(Y - ref) / np.linalg.norm(ref)})")
        elif world_rank % Py == 0:
            assert reduce_rank == 0
            world_comm.Send(lY, dest=0)

        world_comm.Barrier()

        if args['device'] == 'gpu':

            spmm_15d_gpu(lA, lX, lY, cart_comm, bcast_comm, reduce_comm, max_lNKb, gpu_block_size)

            # Gather Y
            if world_rank == 0:
                Y = np.zeros((A.shape[0], args['columns']), dtype)
                Y[:lY.shape[0]] = lY
                idx = lY.shape[0]
                for r in range(Py, world_size, Py):
                    start = idx
                    end = min(Y.shape[0], idx + lY.shape[0])
                    world_comm.Recv(Y[start:end], source=r)
                    idx = end
                utils.mpi_print(
                    world_rank, f"GPU validation: {np.allclose(Y, ref)} "
                                f"({np.linalg.norm(Y - ref) / np.linalg.norm(ref)})")
            elif world_rank % Py == 0:
                assert reduce_rank == 0
                world_comm.Send(lY, dest=0)

    n_iterations = args['iterations']

    dataset_name = 'random'
    if args['dataset'] == 'file':
        dataset_name = args['file']

    if args['device'] == 'cpu':
        wb_logging.wandb_init(world_comm, dataset_name, args['columns'], n_iterations, 'cpu',
                              f'15D_Alex_c_{rfactor}_v0.3', lX.shape[0], args['wandb_key'])

        # Benchmark
        utils.mpi_print(world_rank, "Benchmarking on CPU...")
        cpu_runtimes = repeat("SpMM15D_cpu(lA, lX, lY, cart_comm, bcast_comm, reduce_comm)",
                              setup="cart_comm.Barrier()",
                              repeat=n_iterations,
                              number=1,
                              globals={
                                  **locals(),
                                  **globals()
                              })

        for i, sample in enumerate(cpu_runtimes):
            wb_logging.log({'spmm_time': sample, 'iteration': i})

        utils.mpi_print(
            world_rank, f"CPU: {utils.time_to_ms(np.median(cpu_runtimes))} +- {utils.time_to_ms(np.std(cpu_runtimes))}")
        wb_logging.finish()

    else:

        wb_logging.wandb_init(world_comm, dataset_name, args['columns'], n_iterations, 'gpu',
                              f'15D_Alex_c_{rfactor}_v0.3', lX.shape[0], args['wandb_key'])
        wb_logging.set_iteration_data({'gpu_tiling': True if gpu_block_size else False})

        utils.mpi_print(world_rank, "Benchmarking on GPU...")
        gpu_stmt = "SpMM15D_gpu(lA, lX, lY, cart_comm, bcast_comm, reduce_comm, max_lNKb, gpu_block_size)"
        gpu_setup = "cp.cuda.get_current_stream().synchronize(); cart_comm.Barrier()"
        gpu_runtimes = repeat(gpu_stmt,
                              setup=gpu_setup,
                              repeat=n_iterations,
                              number=1,
                              globals={
                                  **locals(),
                                  **globals()
                              })
        utils.mpi_print(
            world_rank, f"GPU: {utils.time_to_ms(np.median(gpu_runtimes))} +- {utils.time_to_ms(np.std(gpu_runtimes))}")

        for i, sample in enumerate(gpu_runtimes):
            wb_logging.log({'spmm_time': sample, 'iteration': i})

        wb_logging.finish()


if __name__ == "__main__":
    main()