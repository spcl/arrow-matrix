import os
import sys
import time
from typing import Union
import igraph
import numpy as np
from mpi4py import MPI
from arrow import decomposition, arrow_dec_mpi
from arrow.common import wb_logging, graphio


def bench_spmm(path: Union[str, None],
               width: int,
               n_features: int,
               iterations: int,
               blocked: bool,
               device: str,
               p_per_side=3,
               ba_neighbors: int = 5,
               wandb_api_key: str = None,
               datatype=np.float32,
               slim=False,
               npy_format=False):
    assert width > 0

    comm = MPI.COMM_WORLD

    if path is None:
        path = 'tmp/test_ba' + "_" + str(p_per_side) + "_" + str(ba_neighbors)
        if comm.Get_rank() == 0:
            os.makedirs("tmp", exist_ok=True)
            # If no path given, generate random graph
            g = igraph.Graph.Barabasi(p_per_side * width, ba_neighbors, 503, directed=False)
            d = decomposition.arrow_decomposition(g, width, 3, block_diagonal=blocked)
            if npy_format:
                graphio.save_decomposition_new(g, d, path, block_diagonal=blocked)
            else:
                graphio.save_decomposition(g, d, path, block_diagonal=blocked)
            print("DATASET GENERATED -- ", g.vcount(), " vertices")

        comm.Barrier()

    name = "Arrow_v0.45"
    if blocked:
        name += "_BlockDiagonal"
    if slim:
        name += "_Slim"

    wb_logging.wandb_init(comm, path, n_features, iterations, device, name, width, wandb_api_key)

    assert len(path)

    blocks, n_blocks, to_prev, to_next = arrow_dec_mpi.ArrowDecompositionMPI.load_decomposition_new(comm, path, width,
                                                                                                    blocked, datatype,
                                                                                                    slim=slim,
                                                                                                    use_npy=npy_format,
                                                                                                    use_mmap=False)

    if blocks is not None:
        print("RANK loaded decomposition", comm.Get_rank(), n_blocks, "(slim): ", slim, flush=True)

    comm.Barrier()

    if np.sum(n_blocks) == 0:
        # Abort early if no matrix found.
        print("ERROR: Empty Matrix. Check that the file exists and all parameters match (width, block diagonal).",
              file=sys.stderr)
        return

    if not slim and np.sum(2 * n_blocks - 1) > comm.Get_size():
        print("ERROR: Not enough ranks available. Minimum number of ranks to process the matrix:",
              np.sum(2 * n_blocks - 1), file=sys.stderr)
        return

    if slim and np.sum(n_blocks) > comm.Get_size():
        print("ERROR: Not enough ranks available. Minimum number of ranks to process the matrix (slim mode):",
              np.sum(n_blocks), file=sys.stderr)
        return

    # Create comms & Allocate processors to block matrices
    arrow = arrow_dec_mpi.ArrowDecompositionMPI.initialize(comm, n_blocks, to_prev, to_next, width, n_features, device,
                                                           blocked, slim)

    print("RANK ", comm.Get_rank(), "ARROW initialized")

    rank = comm.Get_rank()
    rng = np.random.default_rng(42 + rank)

    comm.Barrier()
    if arrow is not None:

        wb_logging.log({"actual_ranks": arrow.comm.Get_size()})
        if arrow.comm.Get_size() == 0:
            print("Actual size", arrow.comm.Get_size(), flush=True)

        tic = time.perf_counter()

        # Set the A matrix
        print("RANK", comm.Get_rank(), "load from blocks...", flush=True)
        arrow.B.load_sparse_matrix_from_blocks(blocks)

        # Initialize C and X to 0
        arrow.B.zero_rhs(width, n_features)

        arrow.comm.Barrier()

        toc = time.perf_counter()
        wb_logging.log({"init_time": toc - tic})

        # SPMM
        for i in range(iterations):

            # Initialize X (Features)
            if arrow.matrix_index == 0 and arrow.B.is_column_rank():
                X_p0 = 2 * rng.random((width, n_features), dtype=datatype) - 1
                arrow.B.set_features(X_p0)
            arrow.comm.Barrier()

            # SPMM
            fail = False
            try:
                wb_logging.set_iteration_data({"iteration": i})
                tic = time.perf_counter()
                arrow.step()
                toc = time.perf_counter()
                wb_logging.log({"spmm_time": toc - tic})
                print("RANK", comm.Get_rank(), "Iteration", i, " -- ", toc - tic, "s", flush=True)
            except Exception as e:
                print("RANK", comm.Get_rank(), "EXCEPTION", e, flush=True)
                fail = True
            fail = arrow.comm.allreduce(fail, op=MPI.LOR)
            if fail:
                print("RANK", comm.Get_rank(), "FAILED")
                break

    wb_logging.finish()
    comm.Barrier()
