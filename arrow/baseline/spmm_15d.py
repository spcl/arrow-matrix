import ctypes
import numpy as np
import numpy.typing as npt
import time

try:
    import cupy as cp
except ImportError as e:
    print(e)
    cp = None

from mpi4py import MPI
from scipy import sparse
from typing import List
from arrow.common.sp2cp import _sp2cp
from arrow.common import wb_logging, utils


def generate_15d_decomposition(A: sparse.csr_matrix, X_cols: int, dtype: npt.DTypeLike, c: int,
                               rng: np.random.Generator):
    """ Generates the A-stationary 1.5D decomposition with replication factor c for Y = A @ X.
    
    :param A: The matrix A.
    :param X_cols: The number of columns of X.
    :param dtype: The data type of the matrices.
    :param c: The replication factor.
    :param rng: The random number generator to use.
    """

    world_comm = MPI.COMM_WORLD
    world_size = world_comm.Get_size()
    world_rank = world_comm.Get_rank()

    p_div_c = world_size // c
    if world_size != p_div_c * c:
        raise ValueError("The number of processes must be divisible by the replication factor.")

    rounds = p_div_c // c
    if rounds * c != p_div_c:
        raise ValueError("The number of processes must be divisible by the square of the replication factor.")

    # Cartesian process grid P/c x c.
    cart_comm = world_comm.Create_cart((p_div_c, c))
    cart_rank = cart_comm.Get_rank()
    cart_size = cart_comm.Get_size()
    x, y = cart_comm.Get_coords(cart_rank)

    assert cart_size == world_size
    assert cart_rank == world_rank

    # Subcommunicators for broadcasting X.
    bcast_comm = cart_comm.Sub((True, False))
    bcast_rank = bcast_comm.Get_rank()
    bcast_size = bcast_comm.Get_size()
    bx = bcast_comm.Get_coords(bcast_rank)[0]

    assert bcast_size == p_div_c
    assert bx == x

    # Subcommunicators for reducing Y.
    reduce_comm = cart_comm.Sub((False, True))
    reduce_rank = reduce_comm.Get_rank()
    reduce_size = reduce_comm.Get_size()
    ry = reduce_comm.Get_coords(reduce_rank)[0]

    assert reduce_size == c
    assert ry == y

    # Global sizes
    utils.mpi_print(cart_rank, "Broadcasting global sizes...")
    if cart_rank == 0:
        global_sizes = np.array([A.shape[0], A.shape[1], A.nnz], dtype=np.int64)
    else:
        global_sizes = np.empty(3, dtype=np.int64)
    cart_comm.Bcast(global_sizes, root=0)
    NI, NK, NNZ = global_sizes
    NJ = X_cols

    # Local sizes
    lNI, lNK, lNKb = int(np.ceil(NI / p_div_c)), int(np.ceil(NK / c)), int(np.ceil(NK / p_div_c))
    lNK = lNKb * rounds
    lNJ = NJ
    lNNZ = int(np.ceil(NNZ / world_size))

    # Distribute the adjacency matrix
    utils.mpi_print(cart_rank, "Distributing the adjacency matrix...")
    lA = None
    if cart_rank == 0:
        for i in range(p_div_c):
            for j in range(c):
                block = sparse.csr_matrix(A[i * lNI:min(NI, (i + 1) * lNI), j * lNK:min(NK, (j + 1) * lNK)])
                block.sum_duplicates()
                block.sort_indices()
                block._has_canonical_format = True
                if x == i and y == j:
                    lA = block
                    lNI = block.shape[0]
                    lNK = block.shape[1]
                    lNNZ = block.nnz
                else:
                    dst = cart_comm.Get_cart_rank((i, j))
                    size_buffer = np.array([block.shape[0], block.shape[1], block.nnz], dtype=np.int32)
                    cart_comm.Send(size_buffer, dest=dst, tag=0)
                    cart_comm.Send(block.indptr, dest=dst, tag=1)
                    cart_comm.Send(block.indices, dest=dst, tag=2)
                    cart_comm.Send(block.data, dest=dst, tag=3)
    else:
        size_buffer = np.empty(3, dtype=np.int32)
        cart_comm.Recv(size_buffer, source=0, tag=0)
        lNI, lNK, lNNZ = size_buffer
        indptr = np.empty(lNI + 1, dtype=np.int32)
        indices = np.empty(lNNZ, dtype=np.int32)
        data = np.empty(lNNZ, dtype=dtype)
        cart_comm.Recv(indptr, source=0, tag=1)
        cart_comm.Recv(indices, source=0, tag=2)
        cart_comm.Recv(data, source=0, tag=3)
        lA = sparse.csr_matrix((data, indices, indptr), shape=(lNI, lNK), dtype=dtype)

    cart_comm.Barrier()

    # Split lA into "rounds" blocks.
    lA_blocks = []
    for i in range(p_div_c // c):
        new_block = sparse.csr_matrix(lA[:, i * lNKb:min(lNK, (i + 1) * lNKb)])
        new_block.sum_duplicates()
        new_block.sort_indices()
        new_block._has_canonical_format = True
        lA_blocks.append(new_block)

    assert len(lA_blocks) == p_div_c // c

    cart_comm.Barrier()

    # The X matrix is replicated in the "reduce" communicators.
    # Therefore, we generate a random block in reduce-rank 0 and then bcast.
    # TODO Measure cost of distributing X
    utils.mpi_print(cart_rank, f"Generating matrix X with shape ({NK}, {NJ})...")
    # NOTE: Due to way we round-up the block sizes, it is possible that the last block is empty.
    actual_lNKb = min(NK, (bcast_rank + 1) * lNKb) - bcast_rank * lNKb
    if actual_lNKb <= 0:
        actual_lNKb = 0
    if reduce_rank == 0:
        X = utils.generate_dense_matrix(actual_lNKb, lNJ, dtype, rng)
    else:
        X = np.empty((actual_lNKb, lNJ), dtype=dtype)
    if X.size > 2**30:
        chunk = 2**30 // lNJ
        for i in range(0, actual_lNKb, chunk):
            reduce_comm.Bcast(X[i:min(actual_lNKb, i + chunk)], root=0)
    else:
        reduce_comm.Bcast(X, root=0)

    Y = np.empty((lNI, lNJ), dtype=dtype)

    return lA_blocks, X, Y, cart_comm, bcast_comm, reduce_comm, lNKb


def generate_15d_decomposition_new(A: sparse.csr_matrix, X_cols: int, dtype: npt.DTypeLike, c: int,
                                   rng: np.random.Generator):
    """ Generates the A-stationary 1.5D decomposition with replication factor c for Y = A @ X.
    TODO: Update documentation.
    
    :param A: The matrix A.
    :param X_cols: The number of columns of X.
    :param dtype: The data type of the matrices.
    :param c: The replication factor.
    :param rng: The random number generator to use.
    """

    world_comm = MPI.COMM_WORLD
    world_size = world_comm.Get_size()
    world_rank = world_comm.Get_rank()

    p_div_c = world_size // c
    if world_size != p_div_c * c:
        raise ValueError("The number of processes must be divisible by the replication factor.")

    rounds = p_div_c // c
    if rounds * c != p_div_c:
        raise ValueError("The number of processes must be divisible by the square of the replication factor.")

    # Cartesian process grid P/c x c.
    cart_comm = world_comm.Create_cart((p_div_c, c))
    cart_rank = cart_comm.Get_rank()
    cart_size = cart_comm.Get_size()
    x, y = cart_comm.Get_coords(cart_rank)

    assert cart_size == world_size
    assert cart_rank == world_rank

    # Subcommunicators for broadcasting X.
    bcast_comm = cart_comm.Sub((True, False))
    bcast_rank = bcast_comm.Get_rank()
    bcast_size = bcast_comm.Get_size()
    bx = bcast_comm.Get_coords(bcast_rank)[0]

    assert bcast_size == p_div_c
    assert bx == x

    # Subcommunicators for reducing Y.
    reduce_comm = cart_comm.Sub((False, True))
    reduce_rank = reduce_comm.Get_rank()
    reduce_size = reduce_comm.Get_size()
    ry = reduce_comm.Get_coords(reduce_rank)[0]

    assert reduce_size == c
    assert ry == y

    # Global sizes
    utils.mpi_print(cart_rank, "Broadcasting global sizes...")
    if cart_rank == 0:
        # global_sizes = np.array([A.shape[0], A.shape[1], A.nnz], dtype=np.int64)
        global_sizes = np.array([A[2].size - 1, A[2].size - 1, A[0].size], dtype=np.int64)
    else:
        global_sizes = np.empty(3, dtype=np.int64)
    cart_comm.Bcast(global_sizes, root=0)
    NI, NK, NNZ = global_sizes
    NJ = X_cols

    # Local sizes
    lNI, lNK, lNKb = int(np.ceil(NI / p_div_c)), int(np.ceil(NK / c)), int(np.ceil(NK / p_div_c))
    lNK = lNKb * rounds
    lNJ = NJ
    lNNZ = int(np.ceil(NNZ / world_size))

    # Distribute the adjacency matrix
    utils.mpi_print(cart_rank, "Distributing the adjacency matrix...")
    lA = None
    if cart_rank == 0:
        for i in range(p_div_c):
            num_rows = min(NI, (i + 1) * lNI) - i * lNI
            num_cols = NK
            indptr = np.empty(num_rows + 1, dtype=np.int32)
            indptr[:-1] = A[2][i * lNI:min(NI, (i + 1) * lNI)]
            indptr[-1] = A[2][min(NI, (i + 1) * lNI)]
            indices = A[1][indptr[0]:indptr[-1]]
            data = A[0][indptr[0]:indptr[-1]]
            # Fix indptr
            indptr -= indptr[0]
            indptr[-1] = data.size
            # Load full rows
            row_block = sparse.csr_matrix((data, indices, indptr), shape=(num_rows, num_cols), dtype=dtype)
            for j in range(c):
                # block = sparse.csr_matrix(A[i * lNI:min(NI, (i + 1) * lNI), j * lNK:min(NK, (j + 1) * lNK)])
                block = sparse.csr_matrix(row_block[:, j * lNK:min(NK, (j + 1) * lNK)])
                block.sort_indices()
                block.sum_duplicates()
                block._has_canonical_format = True
                if x == i and y == j:
                    lA = block
                    lNI = block.shape[0]
                    lNK = block.shape[1]
                    lNNZ = block.nnz
                else:
                    dst = cart_comm.Get_cart_rank((i, j))
                    size_buffer = np.array([block.shape[0], block.shape[1], block.nnz], dtype=np.int32)
                    cart_comm.Send(size_buffer, dest=dst, tag=0)
                    cart_comm.Send(block.indptr, dest=dst, tag=1)
                    cart_comm.Send(block.indices, dest=dst, tag=2)
                    cart_comm.Send(block.data, dest=dst, tag=3)
    else:
        size_buffer = np.empty(3, dtype=np.int32)
        cart_comm.Recv(size_buffer, source=0, tag=0)
        lNI, lNK, lNNZ = size_buffer
        indptr = np.empty(lNI + 1, dtype=np.int32)
        indices = np.empty(lNNZ, dtype=np.int32)
        data = np.empty(lNNZ, dtype=dtype)
        cart_comm.Recv(indptr, source=0, tag=1)
        cart_comm.Recv(indices, source=0, tag=2)
        cart_comm.Recv(data, source=0, tag=3)
        lA = sparse.csr_matrix((data, indices, indptr), shape=(lNI, lNK), dtype=dtype)

    cart_comm.Barrier()

    # Split lA into "rounds" blocks.
    lA_blocks = []
    for i in range(p_div_c // c):
        new_block = sparse.csr_matrix(lA[:, i * lNKb:min(lNK, (i + 1) * lNKb)])
        new_block.sum_duplicates()
        new_block.sort_indices()
        new_block._has_canonical_format = True
        lA_blocks.append(new_block)

    assert len(lA_blocks) == p_div_c // c

    cart_comm.Barrier()

    # The X matrix is replicated in the "reduce" communicators.
    # Therefore, we generate a random block in reduce-rank 0 and then bcast.
    # TODO Measure cost of distributing X
    utils.mpi_print(cart_rank, f"Generating matrix X with shape ({NK}, {NJ})...")
    # NOTE: Due to way we round-up the block sizes, it is possible that the last block is empty.
    actual_lNKb = min(NK, (bcast_rank + 1) * lNKb) - bcast_rank * lNKb
    if actual_lNKb <= 0:
        actual_lNKb = 0
    if reduce_rank == 0:
        X = utils.generate_dense_matrix(actual_lNKb, lNJ, dtype, rng)
    else:
        X = np.empty((actual_lNKb, lNJ), dtype=dtype)
    if X.size > 2**30:
        chunk = 2**30 // lNJ
        for i in range(0, actual_lNKb, chunk):
            reduce_comm.Bcast(X[i:min(actual_lNKb, i + chunk)], root=0)
    else:
        reduce_comm.Bcast(X, root=0)

    Y = np.empty((lNI, lNJ), dtype=dtype)

    return lA_blocks, X, Y, cart_comm, bcast_comm, reduce_comm, lNKb


def spmm_15d_cpu(A: List[sparse.csr_matrix], X: np.ndarray, Y: np.ndarray, cart_comm: MPI.Cartcomm,
                 bcast_comm: MPI.Cartcomm, reduce_comm: MPI.Cartcomm):
    """ Performs A-stationary 1.5D-based SpMM. CPU execution. """

    # Get coordinates.
    p_div_c, c = cart_comm.Get_topo()[0]
    cart_rank = cart_comm.Get_rank()
    bcast_rank = bcast_comm.Get_rank()
    reduce_size = reduce_comm.Get_size()
    i, j = cart_comm.Get_coords(cart_rank)

    rounds = p_div_c // c
    assert rounds * c == p_div_c

    Y[:] = 0

    bcast_timer = 0
    kernel_timer = 0
    reduce_timer = 0

    for r in range(rounds):

        # Broadcast X.
        tic = time.perf_counter()
        q = j * rounds + r
        buf = X if bcast_rank == q else np.empty_like(X, shape=(A[r].shape[1], X.shape[1]))
        if buf.size > 2**30:
            chunk = 2**30 // buf.shape[1]
            for i in range(0, buf.shape[0], chunk):
                bcast_comm.Bcast(buf[i:min(buf.shape[0], i + chunk)], root=q)
        else:
            bcast_comm.Bcast(buf, root=q)
        toc = time.perf_counter()
        bcast_timer += toc - tic

        # Compute Y = A @ X.
        tic = time.perf_counter()
        Y[:] += A[r] @ buf
        toc = time.perf_counter()
        kernel_timer += toc - tic

    # Reduce Y.
    if reduce_size > 1:
        tic = time.perf_counter()
        if Y.size > 2**30:
            chunk = 2**30 // Y.shape[1]
            for i in range(0, Y.shape[0], chunk):
                reduce_comm.Allreduce(MPI.IN_PLACE, Y[i:min(Y.shape[0], i + chunk)], op=MPI.SUM)
        else:
            reduce_comm.Allreduce(MPI.IN_PLACE, Y, op=MPI.SUM)
        toc = time.perf_counter()
        reduce_timer += toc - tic

    wb_logging.log({"spmm_bcast_time": bcast_timer})
    wb_logging.log({"spmm_kernel_time": kernel_timer})
    wb_logging.log({"spmm_reduce_time": reduce_timer})
    return Y


def spmm_15d_gpu(A: List[sparse.csr_matrix], X: np.ndarray, Y: np.ndarray, cart_comm: MPI.Cartcomm,
                 bcast_comm: MPI.Cartcomm, reduce_comm: MPI.Cartcomm, max_rows: int, block_size: int = None):
    """ Performs A-stationary 1.5D-based SpMM. GPU execution. """

    # Get coordinates.
    p_div_c, c = cart_comm.Get_topo()[0]
    cart_rank = cart_comm.Get_rank()
    bcast_rank = bcast_comm.Get_rank()
    reduce_size = reduce_comm.Get_size()
    i, j = cart_comm.Get_coords(cart_rank)

    rounds = p_div_c // c
    assert rounds * c == p_div_c

    Y[:] = 0

    bcast_timer = 0
    kernel_timer = 0
    reduce_timer = 0

    x_cols = block_size or X.shape[1]
    dev_X = cp.empty((max_rows * x_cols, ), dtype=X.dtype)

    for r in range(rounds):

        # Broadcast X.
        tic = time.perf_counter()
        q = j * rounds + r
        buf = X if bcast_rank == q else np.empty_like(X, shape=(A[r].shape[1], X.shape[1]))
        if buf.size > 2**30:
            chunk = 2**30 // buf.shape[1]
            for i in range(0, buf.shape[0], chunk):
                bcast_comm.Bcast(buf[i:min(buf.shape[0], i + chunk)], root=q)
        else:
            bcast_comm.Bcast(buf, root=q)
        toc = time.perf_counter()
        bcast_timer += toc - tic

        # Compute Y = A @ X.
        tic = time.perf_counter()
        dev_A = _sp2cp(A[r])
        if block_size:
            for i in range(0, buf.shape[1], block_size):
                rows, cols = buf.shape[0], min(buf.shape[1], i + block_size) - i
                current_buf = buf[:, i:min(buf.shape[1], i + block_size)].copy()
                srcptr = current_buf.ctypes.data_as(ctypes.c_void_p)
                dev_X.data.copy_from_host(srcptr, rows * cols * buf.dtype.itemsize)
                current_X = cp.ndarray(shape=(rows, cols), dtype=buf.dtype, memptr=dev_X.data)
                current_Y = dev_A @ current_X
                Y[:, i:min(buf.shape[1], i + block_size)] += cp.asnumpy(current_Y)
                del current_Y
        else:
            rows, cols = buf.shape[0], buf.shape[1]
            srcptr = buf.ctypes.data_as(ctypes.c_void_p)
            dev_X.data.copy_from_host(srcptr, rows * cols * buf.dtype.itemsize)
            current_X = cp.ndarray(shape=(rows, cols), dtype=buf.dtype, memptr=dev_X.data)
            current_Y = dev_A @ current_X
            Y[:] += cp.asnumpy(current_Y)
            del current_Y
        del dev_A
        toc = time.perf_counter()
        kernel_timer += toc - tic

    # Reduce Y.
    if reduce_size > 1:
        tic = time.perf_counter()
        if Y.size > 2**30:
            chunk = 2**30 // Y.shape[1]
            for i in range(0, Y.shape[0], chunk):
                reduce_comm.Allreduce(MPI.IN_PLACE, Y[i:min(Y.shape[0], i + chunk)], op=MPI.SUM)
        else:
            reduce_comm.Allreduce(MPI.IN_PLACE, Y, op=MPI.SUM)
        toc = time.perf_counter()
        reduce_timer += toc - tic

    wb_logging.log({"spmm_bcast_time": bcast_timer})
    wb_logging.log({"spmm_kernel_time": kernel_timer})
    wb_logging.log({"spmm_reduce_time": reduce_timer})
    return Y
