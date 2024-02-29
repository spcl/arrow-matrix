import time
from typing import List, Union
import numpy
import numpy as np
from mpi4py.MPI import Request

from mpi4py import MPI
from scipy import sparse
from arrow.common import wb_logging
from arrow.arrow_matrix import ArrowMatrix
from arrow.common.sp2cp import _sp2cp

try:
    import cupy as cp
except Exception as e:
    print(e)
    pass

try:
    import wandb
except Exception as e:
    pass


class ArrowSlimMPI(ArrowMatrix):

    # The number of tiles per side
    tiles_per_side: int

    # row block (is none for root)
    A_0i: Union[sparse.csr_matrix, None]
    # column block (is none for root)
    A_i0: Union[sparse.csr_matrix, None]

    # Diagonal block
    A_ii: Union[sparse.csr_matrix, None]

    X_0: Union[numpy.ndarray, None]
    X_i: Union[numpy.ndarray, None]

    C_0: Union[numpy.ndarray, None]
    C_0_buf: Union[numpy.ndarray, None]
    C_i: Union[numpy.ndarray, None]

    comm: MPI.Comm

    def __init__(self, comm: MPI.Comm):
        self.comm = comm
        self.column_comm = comm
        # Initializes the class
        comm_size = comm.Get_size()
        assert comm_size >= 1

        self.tiles_per_side = comm_size

        self.A_i0 = None
        self.A_0i = None
        self.A_ii = None
        self.X_i = None
        self.X_0 = None
        self.C_i = None
        self.C_0 = None
        self.C_0_buf = None
        self.A_0i_nnz_columns = None
        self.A_i0_nnz_columns = None
        self.A_ii_nnz_columns = None


    def spmm(self, device: str = 'cpu'):
        """
         Compute the SpMM
        :param device: 'cpu' for CPU and 'gpu' for GPU
        :return:
        """
        self._arrow_spmm(device=device)


    def _ad_spmm(self, rank, bcast_request: Union[Request, None]) -> None:
        """
        Computation of the i-th row tile (indexed from 0).
        C_0 will be updated through a reduce in row_rank 0, as follows:
        C_0 = C_0 + \sum_i A_0i X_i.

        :return:
        """
        assert self.A_0i is not None
        assert self.X_i is not None
        assert self.C_0 is not None
        assert self.comm is not None
        assert self.X_0 is not None
        assert self.C_i is not None
        assert rank == 0 or self.A_ii is not None
        assert rank == 0 or self.A_i0 is not None
        # Assert dimensions match:
        assert self.X_0.shape[1] == self.X_i.shape[1]
        assert rank == 0 or self.A_ii_nnz_columns is not None or self.X_i.shape[0] == self.A_ii.shape[0]
        assert self.A_0i_nnz_columns is not None or self.X_0.shape[0] == self.A_0i.shape[1]
        assert self.A_i0_nnz_columns is not None or rank == 0 or self.X_i.shape[0] == self.A_i0.shape[0]
        assert rank == 0 or self.X_i.shape[0] == self.C_i.shape[0]
        assert rank == 0 or self.X_i.shape[1] == self.C_i.shape[1]
        assert self.X_0.shape[0] == self.C_0.shape[0]
        assert self.X_0.shape[1] == self.C_0.shape[1]

        # First Matmul: A_0i X_i + initiate reduce
        kernel_time = 0
        tic = time.perf_counter()
        self.C_0[:] = 0
        if self.A_0i_nnz_columns is not None:
            self.C_0 += self.A_0i @ self.X_i[self.A_0i_nnz_columns, :]
        else:
            self.C_0 += self.A_0i @ self.X_i
        toc = time.perf_counter()
        kernel_time += toc-tic

        tic = time.perf_counter()
        self.comm.Reduce(self.C_0, self.C_0_buf, MPI.SUM, root=0)
        reduce_request = None
        toc = time.perf_counter()
        wb_logging.log({"spmm_row_reduce": toc - tic})

        # Second Matmul: A_ii X_i
        tic = time.perf_counter()
        if rank > 0:
            if self.A_ii_nnz_columns is not None:
                self.C_i = self.A_ii @ self.X_i[self.A_ii_nnz_columns, :]
            else:
                self.C_i = self.A_ii @ self.X_i
        toc = time.perf_counter()
        kernel_time += toc-tic

        # Wait for broadcast to complete
        tic = time.perf_counter()
        if bcast_request is not None:
            MPI.Request.Wait(bcast_request)
        toc = time.perf_counter()
        wb_logging.log({"spmm_bcast_wait_time": toc - tic})

        # Third Matmul: A_i0 X_0
        tic = time.perf_counter()
        if rank > 0:
            if self.A_i0_nnz_columns is not None:
                self.C_i += self.A_i0 @ self.X_0[self.A_i0_nnz_columns, :]
            else:
                self.C_i += self.A_i0 @ self.X_0
        toc = time.perf_counter()
        kernel_time += toc-tic
        wb_logging.log({"spmm_kernel_time": kernel_time})

        # Wait for reduce to complete
        if reduce_request is not None:
            MPI.Request.Wait(reduce_request)
        if rank == 0:
            tmp = self.C_i
            self.C_i = self.C_0_buf
            self.C_0_buf = tmp


    def _ad_spmm_gpu(self, rank, bcast_request) -> None:
        """
        Computation of the i-th tile (GPU Version)
        :return:
        """
        assert self.A_0i is not None
        assert self.X_i is not None
        assert self.C_0 is not None
        assert self.comm is not None
        assert self.X_0 is not None
        assert self.C_i is not None
        assert rank == 0 or self.A_ii is not None
        assert rank == 0 or self.A_i0 is not None
        # Assert dimensions match:
        assert self.X_0.shape[1] == self.X_i.shape[1]
        assert rank == 0 or self.A_ii_nnz_columns is not None or self.X_i.shape[0] == self.A_ii.shape[0]
        assert self.A_0i_nnz_columns is not None or self.X_0.shape[0] == self.A_0i.shape[1]
        assert self.A_i0_nnz_columns is not None or rank == 0 or self.X_i.shape[0] == self.A_i0.shape[0]
        assert rank == 0 or self.X_i.shape[0] == self.C_i.shape[0]
        assert rank == 0 or self.X_i.shape[1] == self.C_i.shape[1]
        assert self.X_0.shape[0] == self.C_0.shape[0]
        assert self.X_0.shape[1] == self.C_0.shape[1]

        kernel_time = 0
        tic = time.perf_counter()
        tic_conv = time.perf_counter()
        A = _sp2cp(self.A_0i)
        if self.A_0i_nnz_columns is not None:
            X_i = cp.asarray(self.X_i[self.A_0i_nnz_columns, :])
        else:
            X_i = cp.asarray(self.X_i)
        toc_conv = time.perf_counter()
        C = A @ X_i
        self.C_0 = cp.asnumpy(C)
        toc = time.perf_counter()
        wb_logging.log({"spmm_to_gpu_time": toc_conv - tic_conv})
        del A
        kernel_time += toc - tic

        tic = time.perf_counter()
        self.comm.Reduce(self.C_0, self.C_0_buf, MPI.SUM, root=0)
        reduce_request = None
        toc = time.perf_counter()
        wb_logging.log({"spmm_row_reduce": toc - tic})

        # Second Matmul: A_ii X_i
        tic = time.perf_counter()
        if rank > 0:
            if self.A_ii_nnz_columns is not None:
                X_i = cp.asarray(self.X_i[self.A_ii_nnz_columns, :])
            elif self.A_0i_nnz_columns is not None:
                X_i = cp.asarray(self.X_i)
            A = _sp2cp(self.A_ii)
            C_i = A @ X_i
            del A
        toc = time.perf_counter()
        kernel_time += toc - tic

        # Wait for broadcast to complete
        tic = time.perf_counter()
        if bcast_request is not None:
            MPI.Request.Wait(bcast_request)
        toc = time.perf_counter()
        wb_logging.log({"spmm_bcast_wait_time": toc - tic})

        # Third Matmul: A_i0 X_0
        tic = time.perf_counter()
        if rank > 0:
            A = _sp2cp(self.A_i0)
            if self.A_i0_nnz_columns is not None:
                X = cp.asarray(self.X_0[self.A_i0_nnz_columns, :])
            else:
                X = cp.asarray(self.X_0)
            C_i[:] += A @ X
            self.C_i = cp.asnumpy(C_i)

        toc = time.perf_counter()
        kernel_time += toc - tic
        wb_logging.log({"spmm_kernel_time": kernel_time})

        # Wait for reduce to complete
        if reduce_request is not None:
            MPI.Request.Wait(reduce_request)
        if rank == 0:
            tmp = self.C_i
            self.C_i = self.C_0_buf
            self.C_0_buf = tmp

    def _arrow_spmm(self, device: str = 'cpu'):
        """
        The layout is as follows: Each processor of index i is responsible for A_0i, A_ii, and A_0i.

        Precondition: Rank i contains X_i
        Precondition: All ranks have the 'correct' slices of A, namely A_0i, A_ii, and A_i0
        Precondition: Each rank i has C_i initialized
        Post-condition: Each rank i has C_i computed by incrementing by the result of the MM
        :param device: 'cpu' or 'gpu'
        :return:
        """
        assert self.comm is not None
        assert self.tiles_per_side > 0
        assert self.tiles_per_side == self.comm.Get_size()

        rank = self.comm.Get_rank()

        wb_logging.log({"spmm_x_send_time": 0})

        if rank == 0:
            assert self.X_i is not None
            self.X_0 = self.X_i

        assert self.X_0 is not None
        # Nonblocking broadcast of X_0
        tic = time.perf_counter()
        x_bcast_request = None
        self.comm.Bcast(self.X_0, root=0)
        toc = time.perf_counter()
        wb_logging.log({"spmm_x_bcast_time": toc - tic})

        if device == 'cpu':
            self._ad_spmm(rank, x_bcast_request)
        else:
            self._ad_spmm_gpu(rank, x_bcast_request)

    def result_tile(self):
        return self.C_i

    def set_features(self, X: np.ndarray) -> None:
        """
        Sets a SLICE of features.
        NOTE: Does not copy the data, but instead sets a reference to X.
        :param X:
        :return:
        """
        assert X is not None
        self.X_i = X

    def feature_tile(self):
        return self.X_i

    def load_sparse_matrix_from_blocks(self, blocks: List[List[sparse.csr_matrix]]) -> None:
        assert len(blocks) == self.tiles_per_side

        rank = self.comm.Get_rank()

        # first block of row
        self.A_0i = blocks[0][rank]
        assert self.A_0i.shape[0] == self.A_0i.shape[1]
        assert self.A_0i.has_sorted_indices
        assert self.A_0i.has_canonical_format
        self.A_0i.check_format()

        if rank > 0:
            self.A_ii = blocks[rank][rank]

            assert self.A_ii is not None
            assert self.A_ii.has_sorted_indices
            assert self.A_ii.has_canonical_format
            self.A_ii.check_format()

            self.A_i0 = blocks[rank][0]

            assert self.A_ii.shape == self.A_i0.shape
            assert self.A_i0 is not None
            assert self.A_i0.has_sorted_indices
            assert self.A_i0.has_canonical_format
            self.A_i0.check_format()

        self._optimize_Ai_slices()


    def _optimize_Ai_slices(self, threshold = 0.3):
        def nonzero_column_indices(matrix):
            return np.unique(matrix.nonzero()[1])

        # For each matrix that has fewer that threshold * n columns, we can optimize the slice
        if self.A_i0 is not None:
            nnz_columns = nonzero_column_indices(self.A_i0)
            if len(nnz_columns) < threshold * self.A_i0.shape[1]:
                self.A_i0 = self.A_i0[:, nnz_columns]
                self.A_i0_nnz_columns = nnz_columns

        if self.A_ii is not None:
            nnz_columns = nonzero_column_indices(self.A_ii)
            if len(nnz_columns) < threshold * self.A_ii.shape[1]:
                self.A_ii = self.A_ii[:, nnz_columns]
                self.A_ii_nnz_columns = nnz_columns

        if self.A_0i is not None:
            nnz_columns = nonzero_column_indices(self.A_0i)
            if len(nnz_columns) < threshold * self.A_0i.shape[1]:
                self.A_0i = self.A_0i[:, nnz_columns]
                self.A_0i_nnz_columns = nnz_columns



    def zero_rhs(self, number_of_rows_per_rank: int, number_of_columns: int, dtype=np.float32) -> None:
        """
        Clears the feature matrix X and the result matrix C.
        You must call this before the first SpMM iteration to initialize the right buffers.
        :param number_of_rows_per_rank:
        :param number_of_columns:
        :param dtype:
        :return:
        """
        assert number_of_rows_per_rank >= 1
        assert number_of_columns >= 1

        # Create empty buffers for C tile (if not already initialized) Else, set to zero.
        tile_shape = (number_of_rows_per_rank, number_of_columns)
        if self.C_i is None:
            self.C_i = np.zeros(tile_shape, dtype=dtype)
        else:
            assert self.C_i.shape == tile_shape
            self.C_i[:] = 0

        # Do the same for C_0, C_0_buf, X_i, X_0
        if self.C_0 is None:
            self.C_0 = np.zeros(tile_shape, dtype=dtype)
        else:
            assert self.C_0.shape == tile_shape
            self.C_0[:] = 0
        if self.C_0_buf is None:
            self.C_0_buf = np.zeros(tile_shape, dtype=dtype)
        else:
            assert self.C_0_buf.shape == tile_shape
            self.C_0_buf[:] = 0
        if self.X_i is None:
            self.X_i = np.zeros(tile_shape, dtype=dtype)
        else:
            assert self.X_i.shape == tile_shape
            self.X_i[:] = 0
        if self.X_0 is None:
            self.X_0 = np.zeros(tile_shape, dtype=dtype)
        else:
            assert self.X_0.shape == tile_shape
            self.X_0[:] = 0

    @staticmethod
    def column_subgroup(tiles_per_side, group: MPI.Group) -> MPI.Group:
        """
        Given a group that contains all ranks for this matrix, return the column subgroup.
        :param group: contains all ranks for this matrix
        :return:
        """
        return group

    @staticmethod
    def row_subgroup(tiles_per_side, group: MPI.Group) -> MPI.Group:
        """
        Given a group that contains all ranks for this matrix, return the row subgroup
        :param group: contains all ranks for this matrix
        :return:
        """
        return group


    def allgather_result(self, C: np.array) -> np.ndarray:
        """
        All-gathers the result.
        :param C:
        :return:
        """
        assert C is not None
        assert self.C_i is not None
        self.comm.Gather(self.C_i, C, root=0)
        self.comm.Bcast(C, root=0)
        return C

    def is_column_rank(self) -> bool:
        """
        Returns true if this rank is a column rank
        :return:
        """
        return True

    # Deprecated
    def set_features_slice_from_features(self, X: np.ndarray) -> None:
        column_rank = self.comm.Get_rank()
        # corresponding X block
        assert X.shape[0] % self.tiles_per_side == 0
        rows_per_rank = X.shape[0] // self.tiles_per_side
        self.set_features(X[column_rank * rows_per_rank:(column_rank + 1) * rows_per_rank, :])

    def _distribute_arrays(self, A: sparse.csr_matrix, X: np.array, b: int, root: int) -> None:
        """
        Tiles and distributes X and A to the correct ranks.
        :param A: Buffer for A. Assumes that A is padded to a multiple of b
        :param X: Buffer for X
        :param b: arrow-width of A
        :param root: root rank
        :return: None
        """
        # Assumes that A is padded to a multiple of b
        assert root == 0
        assert A.shape[0] % b == 0
        assert A.shape[1] == A.shape[0]
        assert X.shape[0] == A.shape[1]
        assert b > 0

        self.tiles_per_side = int(A.shape[0] / b)
        assert self.tiles_per_side == self.comm.Get_size()

        k = X.shape[1]

        rank = self.comm.Get_rank()

        self.zero_rhs(b, k)

        # Send and receive X
        if rank == 0:
            self.X_0 = X[0:b, :].copy()
            self.X_i = self.X_0
            for i in range(1, self.tiles_per_side):
                self.comm.send(X[i * b:(i + 1) * b, :].copy(), dest=i, tag=0)
        else:
            self.X_i = self.comm.recv(source=0, tag=0)

        if rank == 0:
            self.A_0i = A[0:b, 0:b]

        # If we have a single rank, special case, nothing to distribute
        if self.tiles_per_side == 1:
            return

        # If we are the root, tile A and distribute it
        # Distribute A
        # Distribute A_0i
        if rank == 0:
            for i in range(1, self.tiles_per_side):
                self.comm.send(A[0:b, i * b:(i + 1) * b].copy(), dest=i, tag=0)
        else:
            # Init sparse csr_matrix
            self.A_0i = self.comm.recv(source=0, tag=0)
        # Distribute A_i0
        if rank == 0:
            for i in range(1, self.tiles_per_side):
                self.comm.send(A[i * b:(i + 1) * b, 0:b].copy(), dest=i, tag=1)
        else:
            self.A_i0 = self.comm.recv(source=0, tag=1)

        # Distribute A_ii
        if rank == 0:
            for i in range(1, self.tiles_per_side):
                self.comm.send(A[i * b:(i + 1) * b, i * b:(i + 1) * b].copy(), dest=i, tag=2)
        else:
            self.A_ii = self.comm.recv( source=0, tag=2)

        self.comm.Barrier()