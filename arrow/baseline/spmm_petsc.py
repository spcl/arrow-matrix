import ctypes
import time

import numpy as np
import scipy.sparse
from arrow.matrix_slice import MatrixSlice
from mpi4py import MPI
from arrow.common import wb_logging, utils

"""
Implements a generalized version of the PETSc variant of the hypergraph-partioned SpMM algorithm.

Overview of algorithm:

* Each rank loads its slice of the matrix.
Each slice contains a disjoint set of rows. In the following, we will use n_i to denote the number of rows in the slice of rank i.

* Assume each rank has its own local matrix X_i of size n_i x k already provided.
Conceptually X is given by stacking the X_i to yield an n x k matrix.

* Each rank owns a "local" slice of columns, which we would like to identify. For this purpose, we will all-gather the n_i.
Now, the local slice of rank i start at row \\sum_{j<i} n_j and ends at \\sum_{j<=i} n_j.
To compute this range, compute partial sums over the n_i's. Extract the local slice of A_i as a matrix A_i_local.

construct_receive_tables:
* Each rank now constructs two arrays of equal size, x_index_in and rank_in, that are needed for the communication.
Specifically, they will identify which rows of the X matrix we will need at rank i and from which rank those rows will be sent.
You can do this by finding the nonzero columns of A_i that are NOT in the local slice (with the range given from the previous step).
A column is "nonzero" if it contains at least one nonzero).
These are the indeces x_index_in. To get rank_in, use the partial sum of n_i's, 
specifically for index "x_index", do a search for x_index withing the partial sums to find the first index j where \\sum_{j<=i} n_j > x_index.
 Store j at the same position in rank_in as x_index is within x_index_in.

construct_send_tables:
* Now, we would like to learn where each rank needs to send its rows of X_i. To do this, first each rank computes for each rank how many times that rank appears in rank_in. Do an all-to-all on the result. Now, each rank knows how many rows it must send to each other rank. Then, do an all-to-all-v using the sorted x_index_in accordingly. 

encapsulated in MatrixSlice:
* Now, each rank has the following information:
    * A_i_local: the local slice of A_i
    * A_i_nonlocal: the non-local slice of A_i
    * x_index_in: the indices of the rows of X_i that need to be received
    * rank_in: the ranks from which the rows of X_i need to be received
    * x_index_out: the indices of the rows of X_i that need to be sent
    * rank_out: the ranks to which the rows of X_i need to be sent
    * all_n_i: the number of rows in each rank's slice
    * start_col: the start column of the local slice of A_i
    * end_col: the end column of the local slice of A_i
    
spmm:
* Now, we can do the following:
    * Multiply A_i_local with X_i_local
    * Communicate X_i_nonlocal (non-blocking), using spmm_exchange_X
    * Multiply A_i_nonlocal with X_i_nonlocal
    * Add the results to Y_i_local


"""

try:
    import cupy as cp

    def _sp2cp(matrix: scipy.sparse.csr_matrix) -> cp.sparse.csr_matrix:
        """ Converts a SciPy CSR matrix to a CuPy CSR matrix.

        :param matrix: The SciPy CSR matrix.
        :return: The CuPy CSR matrix.
        """
        if isinstance(matrix, scipy.sparse.csr_matrix):
            tmp = cp.sparse.csr_matrix(
                (cp.asarray(matrix.data), cp.asarray(matrix.indices), cp.asarray(matrix.indptr)),
                shape=matrix.shape, dtype=matrix.dtype)
        else:
            tmp = cp.sparse.csc_matrix(
                (cp.asarray(matrix.data), cp.asarray(matrix.indices), cp.asarray(matrix.indptr)),
                shape=matrix.shape, dtype=matrix.dtype)
        tmp._has_canonical_format = True
        return tmp
except ImportError:
    cp = None


def load_matrix_slice(some_slice: str, rank: int) -> scipy.sparse.csr_matrix:
    """
    Load the matrix slice for the given rank.

    The slice file some_slice has for example the form:
    webbase-2001.part.16.slice.15.npz
    or
    webbase-2001.part.16.slice.11.npz
    etc...
    if the rank is i, then we load slice
    webbase-2001.part.16.slice.{i}.npz

    @param some_slice: The file name of the matrix slice
    @param rank: The rank of the matrix slice to be loaded
    """
    # replace the slice number with i by splitting it by . and replacing the second-to-last element
    matrix_file = some_slice.split('.')
    matrix_file[-2] = str(rank)
    matrix_file = '.'.join(matrix_file)
    utils.mpi_print(rank, f"Rank {rank} loading {matrix_file}")
    return scipy.sparse.load_npz(matrix_file)


def spmm_exchange_x_bulk(comm, matrix_slice: MatrixSlice, X_i_local, X_i_nonlocal):
    """
    Exchange the non-local slice of X_i with the other ranks.
    Non-blocking operation that returns the send and receive lists.
    Sends one message per rank
    @param comm: The MPI communicator
    @param matrix_slice: The MatrixSlice object encapsulating the communication tables
    @param X_i_local: The local slice of X_i
    @param X_i_nonlocal: The non-local slice of X_i
    @return: send_list, recv_list where
    send_list is a list of MPI requests for the send operations and
    recv_list is a list of MPI requests for the receive operations
    """
    # Send one buffer per rank (containing all rows that need to be sent to that rank)
    send_list = []
    assert MatrixSlice._is_sorted(matrix_slice.rank_out)
    for p in range(comm.Get_size()):
        # If there are no rows to send to rank p, skip it
        if matrix_slice.send_count[p] == 0:
            continue

        # get all indices where rank_out == p
        indices = matrix_slice.x_index_out_localized[matrix_slice.send_sdispl[p]:matrix_slice.send_sdispl[p] + matrix_slice.send_count[p]]
        # Assert that indices are in range
        send_list.append(comm.Isend(X_i_local[indices,:],
                                    dest=p,
                                    tag=0))

    # Receive one buffer per rank (containing all rows that need to be received from that rank)
    recv_list = []
    for p in range(comm.Get_size()):
        # If there are no rows to receive from rank p, skip it
        if matrix_slice.recv_count[p] == 0:
            continue

        recv_list.append(comm.Irecv(X_i_nonlocal[matrix_slice.recv_sdispl[p]:matrix_slice.recv_sdispl[p] + matrix_slice.recv_count[p],:],
                                    source=p,
                                    tag=0))

    return send_list, recv_list


def spmm_exchange_X(comm, matrix_slice: MatrixSlice, X_i_local, X_i_nonlocal):
    """
    Exchange the non-local slice of X_i with the other ranks.
    Non-blocking operation that returns the send and receive lists.
    Sends one message per row exchanged (thus, this is not so scalable)
    Preferrably, use spmm_exchange_X_bulk instead.
    @param comm: The MPI communicator
    @param matrix_slice: The MatrixSlice object encapsulating the communication tables
    @param X_i_local: The local slice of X_i
    @param X_i_nonlocal: The non-local slice of X_i
    """
    # Send
    send_list = []
    for i, p in enumerate(matrix_slice.rank_out):
        assert MPI.TAG_UB == 0 or MPI.TAG_UB > matrix_slice.x_index_out_localized[i], f"Tag {matrix_slice.x_index_out_localized[i]} is too large for MPI (TAG_UB={MPI.TAG_UB})"
        send_list.append(comm.Isend(X_i_local[matrix_slice.x_index_out_localized[i],:],
                                    dest=p,
                                    tag=matrix_slice.x_index_out_localized[i]))

    # Receive
    recv_list = []
    assert MatrixSlice._is_sorted(matrix_slice.x_index_in)
    for i, p in enumerate(matrix_slice.rank_in):
        assert MPI.TAG_UB == 0 or MPI.TAG_UB > matrix_slice.x_index_in_localized[i], f"Tag {matrix_slice.x_index_out_localized[i]} is too large for MPI (TAG_UB={MPI.TAG_UB})"
        # Note that his works because the indices are sorted
        recv_list.append(comm.Irecv(X_i_nonlocal[i,:],
                                    source=p,
                                    tag=matrix_slice.x_index_in_localized[i]))

    return send_list, recv_list


def spmm_cpu(comm, matrix_slice: MatrixSlice, X_i_local, Y_i_local, X_i_nonlocal):
    """
    Multiply the local slice of A_i with the local slice of X_i
    communicate the non-local slice,
    and multiply the non-local slice of A_i with the non-local slice of X_i
    Increment the content of Y_i_local by the result of the multiplication.
    """
    assert matrix_slice.A_i_local.shape[1] == X_i_local.shape[0]
    assert matrix_slice.A_i_nonlocal.shape[1] == X_i_nonlocal.shape[0]
    assert matrix_slice.A_i_local.shape[0] == Y_i_local.shape[0]
    assert matrix_slice.A_i_nonlocal.shape[0] == Y_i_local.shape[0]

    # Communicate the non-local slice
    tic = time.perf_counter()
    send_list, recv_list = spmm_exchange_x_bulk(comm, matrix_slice, X_i_local, X_i_nonlocal)
    toc = time.perf_counter()
    wb_logging.log({"comm_init_time": toc - tic})

    # Multiply the local slice of A_i with the local slice of X_i
    tic = time.perf_counter()
    Y_i_local += matrix_slice.A_i_local @ X_i_local
    toc = time.perf_counter()
    wb_logging.log({"local_spmm_kernel_time": toc - tic})

    # Wait on all receives
    tic = time.perf_counter()
    MPI.Request.Waitall(recv_list)
    toc = time.perf_counter()
    wb_logging.log({"receive_wait_time": toc - tic})

    # Multiply the non-local slice of A_i with the non-local slice of X_i
    tic = time.perf_counter()
    Y_i_local += matrix_slice.A_i_nonlocal @ X_i_nonlocal
    toc = time.perf_counter()
    wb_logging.log({"nonlocal_spmm_kernel_time": toc - tic})

    # Wait on all sends (otherwise we might overwrite the send buffer)
    tic = time.perf_counter()
    MPI.Request.Waitall(send_list)
    toc = time.perf_counter()
    wb_logging.log({"send_wait_time": toc - tic})

    return Y_i_local


def spmm_gpu(comm, matrix_slice: MatrixSlice, X_i_local, Y_i_local, X_i_nonlocal,
             bsize_local: int = None, bsize_nonlocal: int = None):
    """
    GPU version of spmm.
    Multiply the local slice of A_i with the local slice of X_i
    communicate the non-local slice,
    and multiply the non-local slice of A_i with the non-local slice of X_i
    Set the content of Y_i_local to the result of the multiplication.
    """
    assert matrix_slice.A_i_local.shape[1] == X_i_local.shape[0]
    # assert matrix_slice.A_i_nonlocal.shape[1] == X_i_nonlocal.shape[0]
    assert matrix_slice.A_i_local.shape[0] == Y_i_local.shape[0]
    # assert matrix_slice.A_i_nonlocal.shape[0] == Y_i_local.shape[0]

    # Communicate the non-local slice
    tic = time.perf_counter()
    send_list, recv_list = spmm_exchange_x_bulk(comm, matrix_slice, X_i_local, X_i_nonlocal)
    toc = time.perf_counter()
    wb_logging.log({"comm_init_time": toc - tic})

    # Multiply the local slice of A_i with the local slice of X_i]
    tic = time.perf_counter()
    buf = X_i_local
    x_cols = bsize_local or buf.shape[1]
    dev_X = cp.empty((buf.shape[0] * x_cols, ), dtype=buf.dtype)
    dev_A = _sp2cp(matrix_slice.A_i_local)
    if bsize_local:
        for i in range(0, buf.shape[1], bsize_local):
            rows, cols = buf.shape[0], min(buf.shape[1], i + bsize_local) - i
            current_buf = buf[:, i:min(buf.shape[1], i + bsize_local)]
            srcptr = current_buf.ctypes.data_as(ctypes.c_void_p)
            dev_X.data.copy_from_host(srcptr, rows * cols * buf.dtype.itemsize)
            current_X = cp.ndarray(shape=(rows, cols), dtype=buf.dtype, memptr=dev_X.data)
            current_Y = dev_A @ current_X
            Y_i_local[:, i:min(buf.shape[1], i + bsize_local)] += cp.asnumpy(current_Y)
            del current_Y
    else:
        rows, cols = buf.shape[0], buf.shape[1]
        srcptr = buf.ctypes.data_as(ctypes.c_void_p)
        dev_X.data.copy_from_host(srcptr, rows * cols * buf.dtype.itemsize)
        current_X = cp.ndarray(shape=(rows, cols), dtype=buf.dtype, memptr=dev_X.data)
        current_Y = dev_A @ current_X
        Y_i_local[:] += cp.asnumpy(current_Y)
        del current_Y
    del dev_A
    del dev_X
    toc = time.perf_counter()
    wb_logging.log({"local_spmm_kernel_time": toc - tic})
    tic = time.perf_counter()
    # Wait on all receives
    MPI.Request.Waitall(recv_list)
    toc = time.perf_counter()
    wb_logging.log({"receive_wait_time": toc - tic})

    tic = time.perf_counter()
    # Multiply the non-local slice of A_i with the non-local slice of X_i
    if bsize_nonlocal:
        x_rows = max(b.shape[1] for b in matrix_slice.A_i_nonlocal)
        dev_X = cp.empty((x_rows * bsize_nonlocal, ), dtype=buf.dtype)
        buf_idx = 0
        for j, A_i_nonlocal in enumerate(matrix_slice.A_i_nonlocal):
            buf = X_i_nonlocal[buf_idx:buf_idx + A_i_nonlocal.shape[1], :]
            dev_A = _sp2cp(A_i_nonlocal) 
            for i in range(0, buf.shape[1], bsize_nonlocal):
                rows, cols = buf.shape[0], min(buf.shape[1], i + bsize_nonlocal) - i
                current_buf = buf[:, i:min(buf.shape[1], i + bsize_nonlocal)]
                srcptr = current_buf.ctypes.data_as(ctypes.c_void_p)
                # print(f"Rank {comm.Get_rank()}: block {j}, round {i}: {rows} x {cols}")
                dev_X.data.copy_from_host(srcptr, rows * cols * buf.dtype.itemsize)
                current_X = cp.ndarray(shape=(rows, cols), dtype=buf.dtype, memptr=dev_X.data)
                current_Y = dev_A @ current_X
                Y_i_local[:, i:min(buf.shape[1], i + bsize_nonlocal)] += cp.asnumpy(current_Y)
                del current_Y
            buf_idx += A_i_nonlocal.shape[1]
    else:
        buf = X_i_nonlocal
        dev_X = cp.empty((buf.shape[0] * buf.shape[1], ), dtype=buf.dtype)
        dev_A = _sp2cp(matrix_slice.A_i_nonlocal)
        rows, cols = buf.shape[0], buf.shape[1]
        srcptr = buf.ctypes.data_as(ctypes.c_void_p)
        dev_X.data.copy_from_host(srcptr, rows * cols * buf.dtype.itemsize)
        current_X = cp.ndarray(shape=(rows, cols), dtype=buf.dtype, memptr=dev_X.data)
        current_Y = dev_A @ current_X
        Y_i_local[:] += cp.asnumpy(current_Y)
        del current_Y
        del dev_A
        del dev_X
    toc = time.perf_counter()
    wb_logging.log({"nonlocal_spmm_kernel_time": toc - tic})

    # Wait on all sends (otherwise we might overwrite the send buffer)
    tic = time.perf_counter()
    MPI.Request.Waitall(send_list)
    toc = time.perf_counter()
    wb_logging.log({"send_wait_time": toc - tic})

    return Y_i_local


def compute_gpu_tiling_size(comm, mat_slice: MatrixSlice, k: int, Y_i_local: np.ndarray, X_i_nonlocal: np.ndarray, mem_fraction: float):
    """
    Compute the block size for the GPU tiling.
    @param comm: The MPI communicator
    @param mat_slice: The MatrixSlice object encapsulating the communication tables
    @param k: The number of columns of X_i
    @param Y_i_local: The local slice of Y_i
    @param X_i_nonlocal: The non-local slice of X_i
    @param mem_fraction: The fraction of GPU memory to use
    @return: gpu_bsize_local, gpu_bsize_nonlocal
    """
    gpu_bsize_local = None
    gpu_bsize_nonlocal = None
    dtype = X_i_nonlocal.dtype

    rank = comm.Get_rank()
    comm_size = comm.Get_size()
    # NOTE: For local debugging
    # available_memory = 0.1 * cp.cuda.Device(0).mem_info[0] / comm_size
    # comm.Barrier()
    utils.mpi_print(rank, f"Using {mem_fraction} of GPU memory")
    available_memory = mem_fraction * cp.cuda.Device(0).mem_info[0]

    # Compute the maximum block size for the local slice
    block = mat_slice.A_i_local
    lA_memory = block.data.nbytes + block.indptr.nbytes + block.indices.nbytes
    if lA_memory > available_memory:
        raise ValueError(f"The local A-block does not fit in GPU memory. Available: {available_memory}, "
                         f"required: {lA_memory}")
    remaining_memory = available_memory - lA_memory
    max_cols = int(remaining_memory // ((block.shape[0] + block.shape[1]) * dtype.itemsize))
    if max_cols < k:
        gpu_bsize_local = max_cols
    utils.mpi_print(rank, f"Available memory: {available_memory}, A-block memory: {lA_memory}, "
                          f"GPU block size: {gpu_bsize_local}")

    # Compute the maximum block size for the non-local slice
    block = mat_slice.A_i_nonlocal
    required_memory = block.data.nbytes + block.indptr.nbytes + block.indices.nbytes
    required_memory += 2 * X_i_nonlocal.nbytes
    required_memory += 2 * Y_i_local.nbytes

    print(f"Rank {rank}: required memory: {required_memory}; {required_memory/available_memory} of available memory", flush=True)

    if required_memory > available_memory:
        # Break non-local slice into P-1 blocks
        l = block.shape[1]
        n = comm_size - 1
        block_rows = l // n
        A_i_nonlocal_blocks = []
        for i in range(l % n):
            A_i_nonlocal_blocks.append(block[:, i * (block_rows + 1):(i + 1) * (block_rows + 1)])
        sidx = l % n * (block_rows + 1)
        for i in range(n - l % n):
            A_i_nonlocal_blocks.append(block[:, sidx + i * block_rows:sidx + (i + 1) * block_rows])
        for b in A_i_nonlocal_blocks:
            b.sort_indices()
            b.sum_duplicates()
            b.eliminate_zeros()
        mat_slice.A_i_nonlocal = A_i_nonlocal_blocks
        lA_memory = max(
            [block.data.nbytes + block.indptr.nbytes + block.indices.nbytes for block in A_i_nonlocal_blocks])
        if lA_memory > available_memory:
            raise ValueError(f"The non-local A-block does not fit in GPU memory. Available: {available_memory}, "
                             f"required: {lA_memory}")
        remaining_memory = available_memory - lA_memory
        x_rows = max([block.shape[1] for block in A_i_nonlocal_blocks])
        max_cols = int(remaining_memory // ((block.shape[0] + x_rows) * dtype.itemsize))
        gpu_bsize_nonlocal = min(max_cols, k)
        utils.mpi_print(rank, f"Available memory: {available_memory}, A-block memory: {lA_memory}, "
                              f"GPU block size: {gpu_bsize_nonlocal}")

    return gpu_bsize_local, gpu_bsize_nonlocal


def benchmark_spmm(matrix_slice_file: str,
                   k: int,
                   iterations: int,
                   device: str,
                   wandb_api_key: str,
                   dtype: np.dtype,
                   rng: np.random.Generator,
                   gpu_tiling: bool = False,
                   dryrun: bool = False,
                   mem_fraction: float = 0.9):
    """
    Benchmark an SpMM run (with multiple iterations) on the given matrix.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    comm_size = comm.Get_size()
    width = 0
    name = "PETSc_v0.1"

    dataset_name = matrix_slice_file.split('.')[0] if matrix_slice_file is not None else None

    wb_logging.wandb_init(comm, dataset_name, k, iterations, device, name, width, wandb_api_key)

    utils.mpi_print(rank, "Loading matrix slice...")
    # Extract the number of parts from the file name
    if matrix_slice_file is None:
        # NOTE: For testing purposes, we generate a random matrix
        scale = 4*1024
        A_i = utils.generate_sparse_matrix(scale, comm_size * scale, scale * 10, dtype, rng)
    else:
        nr_parts = int(matrix_slice_file.split('.')[-4])
        if nr_parts != comm_size:
            raise ValueError(f"Number of parts in file name ({nr_parts}) does not match number of ranks ({comm_size})")

        # Load the matrix slice
        A_i = load_matrix_slice(matrix_slice_file, rank).astype(dtype)
        A_i.eliminate_zeros()
        A_i.sort_indices()
        A_i.sum_duplicates()

        total_n = comm.reduce(A_i.shape[0], op=MPI.SUM, root=0)
        total_nnz = comm.reduce(A_i.nnz, op=MPI.SUM, root=0)
        utils.mpi_print(rank, f"Total number of rows: {total_n}, total number of nonzeros: {total_nnz}")

    # Initialize the communication arrays
    utils.mpi_print(rank, "Initializing communication arrays...")
    mat_slice = MatrixSlice.initialize(comm, A_i)

    wb_logging.log({"nonlocal_columns": mat_slice.A_i_nonlocal.shape[1], "local_columns": mat_slice.A_i_local.shape[1]})

    if dryrun:
        return

    # Initialize the local result
    Y_i_local = np.zeros((A_i.shape[0], k), dtype=dtype)
    X_i_nonlocal = np.zeros((mat_slice.rank_in.size, k), dtype=dtype)

    gpu_bsize_local = None
    gpu_bsize_nonlocal = None
    if device == 'gpu' and gpu_tiling:
        gpu_bsize_local, gpu_bsize_nonlocal = compute_gpu_tiling_size(comm, mat_slice, k, Y_i_local, X_i_nonlocal, mem_fraction)

    wb_logging.set_iteration_data({'gpu_tiling': True if (gpu_bsize_local or gpu_bsize_nonlocal) else False})

    func = spmm_gpu if device == 'gpu' else spmm_cpu

    if device == 'gpu':
        kwargs = {'bsize_local': gpu_bsize_local, 'bsize_nonlocal': gpu_bsize_nonlocal}
    else:
        kwargs = {}

    # Call SpMM in a loop
    utils.mpi_print(rank, "Starting Spmm iterations...")
    for i in range(iterations):
        wb_logging.set_iteration_data({"iteration": i})
        # For consistency, use a new random matrix each time
        X_i = utils.generate_dense_matrix(A_i.shape[0], k, dtype, rng)
        Y_i_local *= 0

        comm.Barrier()
        tic = time.perf_counter()
        try:
            Y_i_local = func(comm, mat_slice, X_i, Y_i_local, X_i_nonlocal, **kwargs)
            success = True
        except(ValueError, RuntimeError) as e:
            print(f"Rank {rank} encountered an error: {e}", flush=True)
            success = False

        toc = time.perf_counter()
        wb_logging.log({"spmm_time": toc - tic})
        success = comm.allreduce(success, op=MPI.LAND)
        toc = time.perf_counter()
        wb_logging.log({"spmm_time_w_allreduce": toc - tic})
        if not success:
            break

    wb_logging.finish()
    comm.Barrier()
