import time
from typing import List, Union
import numpy
import numpy as np
from mpi4py.MPI import Request

from mpi4py import MPI
from scipy import sparse

from arrow.common import wb_logging
from arrow.common.sp2cp import _sp2cp
from arrow.arrow_matrix import ArrowMatrix


try:
    import cupy as cp
except Exception as e:
    print(e)

try:
    import wandb
except Exception as e:
    pass


class ArrowMPI(ArrowMatrix):

    # The number of tiles per side
    tiles_per_side: int

    # For the row processors (i>=0)
    A_0i: Union[sparse.csr_matrix, None]
    # For the column processors (i>0)
    A_i0: Union[sparse.csr_matrix, None]
    # For the column processors (i>0) where j in i-1, i, i+1
    A_ij: List[numpy.array]

    # For the column processors
    X_0: Union[numpy.array, None]
    # For the column processors (i>0) where j in i-1, i, i+1
    X_ij: List[numpy.array]

    # For the row processors
    C_0: Union[numpy.array, None]
    # For the column processors (i>0)
    C_i: Union[numpy.array, None]

    # If true, only the diagonal blocks are not None
    is_block_diagonal: bool

    comm: MPI.Comm
    row_comm: MPI.Comm
    column_comm: MPI.Comm

    def is_column_rank(self):
        return self.comm.rank == 0 or self.comm.rank >= self.tiles_per_side

    def result_tile(self):
        return self.C_i

    def __init__(self, comm: MPI.Comm, is_block_diagonal: bool = False):
        self.comm = comm
        # Initializes the class
        comm_size = comm.Get_size()
        assert comm_size % 2 == 1
        assert comm_size >= 1

        self.is_block_diagonal = is_block_diagonal
        self.tiles_per_side = (comm_size+1) // 2
        comm = self.comm

        # construct the row and column sub-communicators
        total_group = comm.Get_group()
        row_group = self.row_subgroup(self.tiles_per_side, total_group)
        column_group = self.column_subgroup(self.tiles_per_side, total_group)

        assert column_group.size == self.tiles_per_side
        assert row_group.size == self.tiles_per_side

        self.row_comm = MPI.Comm.Create(comm, row_group)
        self.column_comm = MPI.Comm.Create(comm, column_group)

        self.A_ij = [None, None, None]
        self.X_ij = [None, None, None]
        self.X_0 = None
        self.C_i = None
        self.C_0 = None
        self.A_i0 = None
        self.A_0i = None

        self.A_ij_1_nnz_columns = None
        self.A_i0_nnz_columns = None
        self.A_0i_nnz_columns = None

    def feature_tile(self):
        return self.X_ij[1]

    def spmm(self, device: str = 'cpu'):
        """
         Compute the SpMM
        :param device: 'cpu' for CPU and 'gpu' for GPU
        :return:
        """
        self._arrow_spmm(self.comm, self.row_comm, self.column_comm, device=device)

    def _send_xi_tiles(self, comm: MPI.Comm, rank: int) -> None:
        if self.tiles_per_side == 1:
            return

        assert self.is_block_diagonal
        assert self.X_ij[1] is not None

        if rank >= self.tiles_per_side:
            # Send to top "diagonal" row neighbor (rank+comm_size/2)
            assert rank - self.tiles_per_side + 1 > 0
            comm.Send(self.X_ij[1], rank - self.tiles_per_side + 1)

        elif rank != 0:
            # If row tile, receive from "diagonal" column neighbor (rank-comm_size/2)
            assert rank < self.tiles_per_side
            comm.Recv(self.X_ij[1], rank + self.tiles_per_side - 1)

    def _isend_xi_tiles(self, comm: MPI.Comm, rank: int) -> None:
        # TODO Support "block-diagonal mode"
        # TODO In block-diagonal mode, we could do synchronous sends
        if self.tiles_per_side == 1:
            return

        assert self.X_ij[1] is not None
        # ASSUMES THAT THE COLUMN TILES HOLD THE X_ij[1]
        # TODO assertions for other j's
        requests: List[MPI.Request] = []
        p = comm.Get_size()
        if rank == 0 or rank >= self.tiles_per_side:
            # If column tile, send to top and bottom neighbor
            # If "block-diagonal" mode, not necessary
            previous_column_rank = None
            next_column_rank = None
            if rank == 0:
                next_column_rank = self.tiles_per_side
            elif rank < p-1:
                assert rank < comm.Get_size()-1
                next_column_rank = rank+1

            if rank == self.tiles_per_side:
                previous_column_rank = 0
            elif rank >= self.tiles_per_side:
                previous_column_rank = rank-1

            # Send to top neighbor and receive from top neighbor
            # NOTE: Need to use non-blocking sends to avoid deadlock
            if previous_column_rank is not None and not self.is_block_diagonal:
                assert self.X_ij[0] is not None and not self.is_block_diagonal
                requests.append(comm.Isend(self.X_ij[1], previous_column_rank, 1))
                requests.append(comm.Irecv(self.X_ij[0], previous_column_rank, 2))

            # Send to bottom neighbor and receive from bottom neighbor
            # NOTE: Need to use non-blocking sends to avoid deadlock
            if next_column_rank is not None and not self.is_block_diagonal:
                assert self.X_ij[2] is not None and not self.is_block_diagonal
                requests.append(comm.Isend(self.X_ij[1], next_column_rank, 2))
                requests.append(comm.Irecv(self.X_ij[2], next_column_rank, 1))

            # Send to top "diagonal" row neighbor (rank+comm_size/2)
            if rank != 0:
                assert rank-self.tiles_per_side+1 > 0
                requests.append(comm.Isend(self.X_ij[1], rank-self.tiles_per_side+1))

        elif rank != 0:
            # If row tile, receive from "diagonal" column neighbor (rank-comm_size/2)
            assert rank < self.tiles_per_side
            requests.append(comm.Irecv(self.X_ij[1], rank + self.tiles_per_side-1))

        # Wait for nonblocking sends to finish
        MPI.Request.Waitall(requests)

    def _ad_spmm_column_tile(self, column_comm, rank: int, bcast_request: Request) -> None:
        """
        Computation of the i-th column tile (i > 0).
        :param column_comm: Communicator that contains the processors responsible for the column computation
        :param rank: index into the column tiles (i>0). Index 1 corresponds to C_1.
        :return:
        """
        if rank == 0:
            return
        else:
            assert self.X_0 is not None
            assert self.C_i is not None
            assert self.A_i0 is not None
            assert self.A_ij[1] is not None
            assert self.X_ij[1] is not None
            assert self.A_ij[0] is None or self.X_ij[0] is not None
            assert self.A_ij[2] is None or self.X_ij[2] is not None
            assert column_comm is not None

            tic = time.perf_counter()
            if self.A_ij_1_nnz_columns is not None:
                self.C_i = self.A_ij[1] @ self.X_ij[1][self.A_ij_1_nnz_columns, :]
            else:
                self.C_i = self.A_ij[1] @ self.X_ij[1]

            # Wait for the broadcast to complete
            if bcast_request is not None:
                MPI.Request.Wait(bcast_request)

            if self.A_i0_nnz_columns is not None:
                self.C_i += self.A_i0 @ self.X_0[self.A_i0_nnz_columns, :]
            else:
                self.C_i += self.A_i0 @ self.X_0

            if self.A_ij[0] is not None:
                assert self.C_i.shape == self.X_ij[0].shape
                assert rank > 1
                assert self.A_ij[0].shape[1] == self.X_ij[0].shape[0]
                self.C_i += self.A_ij[0] @ self.X_ij[0]

            if self.A_ij[2] is not None:
                assert rank < self.tiles_per_side - 1
                self.C_i += self.A_ij[2] @ self.X_ij[2]

            toc = time.perf_counter()
            wb_logging.log({"spmm_kernel_time": toc - tic})

    
    def _ad_spmm_column_tile_gpu(self, column_comm, rank: int, bcast_request: Request) -> None:
        """
        Computation of the i-th column tile (i > 0). GPU version.
        :param column_comm: Communicator that contains the processors responsible for the column computation
        :param rank: index into the column tiles (i>0). Index 1 corresponds to C_1.
        :return:
        """
        if rank == 0:
            return
        else:
            assert self.X_0 is not None
            assert self.C_i is not None
            assert self.A_i0 is not None
            assert self.A_ij[1] is not None
            assert self.X_ij[1] is not None
            assert self.A_ij[0] is None or self.X_ij[0] is not None
            assert self.A_ij[2] is None or self.X_ij[2] is not None
            assert column_comm is not None

            tic = time.perf_counter()
            A = _sp2cp(self.A_ij[1])
            if self.A_ij_1_nnz_columns is not None:
                X = cp.asarray(self.X_ij[1][self.A_ij_1_nnz_columns, :])
            else:
                X = cp.asarray(self.X_ij[1])
            C = A @ X

            if bcast_request is not None:
                MPI.Request.Wait(bcast_request)

            A = _sp2cp(self.A_i0)
            if self.A_i0_nnz_columns is not None:
                X = cp.asarray(self.X_0[self.A_i0_nnz_columns, :])
            else:
                X = cp.asarray(self.X_0)
            C[:] += A @ X

            if self.A_ij[0] is not None:
                A = _sp2cp(self.A_ij[0])
                X = cp.asarray(self.X_ij[0])
                C[:] += A @ X
            if self.A_ij[2] is not None:
                A = _sp2cp(self.A_ij[2])
                X = cp.asarray(self.X_ij[2])
                C[:] += A @ X
            self.C_i = cp.asnumpy(C)
            toc = time.perf_counter()
            wb_logging.log({"spmm_kernel_time": toc - tic})

    def _ad_spmm_row_tile(self, row_comm, row_rank) -> None:
        """
        Computation of the i-th row tile (indexed from 0).
        When invoked for all row tiles, compute the result of the first output tile C_0.
        C_0 will be updated through a reduce in row_rank 0, as follows:
        C_0 = C_0 + \sum_i A_0i X_i.
        :return:
        """
        assert self.A_0i is not None
        assert self.X_ij[1] is not None
        assert row_rank > 0 or self.C_0 is not None
        assert row_comm is not None

        tic = time.perf_counter()
        if self.A_0i_nnz_columns is not None:
            self.C_i = self.A_0i @ self.X_ij[1][self.A_0i_nnz_columns, :]
        else:
            self.C_i = self.A_0i @ self.X_ij[1]
        toc = time.perf_counter()
        wb_logging.log({"spmm_kernel_time": toc - tic})
        tic = time.perf_counter()
        row_comm.Reduce(self.C_i, self.C_0, MPI.SUM, root=0)
        if row_rank == 0:
            self.C_i = self.C_0
        toc = time.perf_counter()
        wb_logging.log({"spmm_row_reduce": toc - tic})

    def _ad_spmm_row_tile_gpu(self, row_comm, row_rank) -> None:
        """
        Computation of the i-th row tile (indexed from 0). GPU version.
        When invoked for all row tiles, compute the result of the first output tile C_0.
        C_0 will be updated through a reduce in row_rank 0, as follows:
        C_0 = C_0 + \sum_i A_0i X_i.
        :return:
        """
        assert self.A_0i is not None
        assert self.X_ij[1] is not None
        assert row_rank > 0 or self.C_0 is not None
        assert row_comm is not None

        # TODO When doing multiple SPMMs, we might want to keep A on GPU
        tic = time.perf_counter()
        tic_conv = time.perf_counter()
        A = _sp2cp(self.A_0i)
        if self.A_0i_nnz_columns is not None:
            X = cp.asarray(self.X_ij[1][self.A_0i_nnz_columns, :])
        else:
            X = cp.asarray(self.X_ij[1])
        toc_conv = time.perf_counter()
        C = A @ X
        self.C_i = cp.asnumpy(C)
        toc = time.perf_counter()
        wb_logging.log({"spmm_to_gpu_time": toc_conv - tic_conv})
        wb_logging.log({"spmm_kernel_time": toc - tic})

        # self.C_i = self.A_0i @ self.X_ij[1]
        tic = time.perf_counter()
        row_comm.Reduce(self.C_i, self.C_0, MPI.SUM, root=0)

        if row_rank == 0:
            self.C_i[:] = self.C_0
        toc = time.perf_counter()
        wb_logging.log({"spmm_row_reduce": toc - tic})

    def _arrow_spmm(self, comm, row_comm, column_comm, device: str = 'cpu'):
        """
        The layout is as follows: For odd p processors, the first (p+1)/2 ranks are row ranks
        The next (p-1)/2 ranks are column ranks.
        rank 0 is both a column and row rank.
        The column_comm contains all the column tiles, where rank 0 is also column rank 0
        The row_comm contains all the row tiles. The row ranks equal the rank in comm
        Precondition: The row ranks contain X_i
        Precondition: All ranks have the 'correct' slices of A
        Precondition: The column ranks have C_0 initialized
        Precondition: The row ranks have C_i initialized
        Post-condition: The i-th column rank has C_i computed. Note that at the moment, the initial state of C_i is discarded.
        :param comm: Communicator that contains all the ranks
        :param row_comm: Communicator on the row ranks
        :param column_comm: Communicator on the column ranks
        :param device: 'cpu' or 'gpu'
        :return:
        """
        assert comm is not None
        assert self.tiles_per_side > 0

        comm_size = comm.Get_size()
        rank = comm.Get_rank()
        assert comm_size % 2 == 1

        tic = time.perf_counter()
        if self.is_block_diagonal:
            self._send_xi_tiles(comm, rank)
        else:
            self._isend_xi_tiles(comm, rank)
        toc = time.perf_counter()
        wb_logging.log({"spmm_x_send_time": toc - tic})

        bcast_request = None
        if self.is_column_rank():
            tic = time.perf_counter()
            assert column_comm is not None
            assert column_comm.Get_rank() == self._column_rank(rank)

            if rank == 0:
                assert self.X_ij[1] is not None
                self.X_0 = self.X_ij[1]

            assert self.X_0 is not None

            column_comm.Bcast(self.X_0, root=0)
            toc = time.perf_counter()
            wb_logging.log({"spmm_x_bcast_time": toc - tic})

        if rank < self.tiles_per_side:
            assert row_comm is not None
            # row tile
            if device == 'cpu':
                self._ad_spmm_row_tile(row_comm, rank)
            else:
                self._ad_spmm_row_tile_gpu(row_comm, rank)

            if bcast_request is not None:
                assert rank == 0
                MPI.Request.Wait(bcast_request)

        else:
            assert column_comm is not None
            assert column_comm.Get_rank() >= 0
            # column tile
            if device == 'cpu':
                self._ad_spmm_column_tile(column_comm, self._column_rank(rank), bcast_request)
            else:
                self._ad_spmm_column_tile_gpu(column_comm, self._column_rank(rank), bcast_request)

    def _column_rank(self, rank):
        """
        Given a rank within the matrix communicator, return the rank in the column communicator,
        assuming that the rank corresponds to a column tile.
        :param rank:
        :return:
        """
        assert self.tiles_per_side >= 1

        if rank == 0:
            # Rank 0 is both a row and column, hence extra treatment
            column_rank = 0
        else:
            column_rank = rank - self.tiles_per_side + 1

        return column_rank



    def set_features(self, X: np.ndarray):
        """
        Sets a SLICE of features.
        NOTE: Does not copy the data, but instead sets a reference to X.
        :param X:
        :return:
        """
        assert X is not None

        self.X_ij[1] = X

    def load_sparse_matrix_from_blocks(self, blocks: List[List[sparse.csr_matrix]]):
        assert len(blocks) == self.tiles_per_side

        #print("LOAD SPARSE MATRIX ", self.comm.Get_rank())
        rank = self.comm.Get_rank()
        column_rank = self._column_rank(rank)

        if column_rank > 0:
            # first block of row
            self.A_i0 = blocks[column_rank][0]

            assert self.A_i0.has_sorted_indices
            assert self.A_i0.has_canonical_format
            self.A_i0.check_format()

            assert self.A_i0.shape[0] == self.A_i0.shape[1]

            # j=i exists
            self.A_ij[1] = blocks[column_rank][column_rank]
            assert self.A_ij[1].shape == self.A_i0.shape

            if column_rank > 1:
                # j=i-1 exists if column index is at least 2
                self.A_ij[0] = blocks[column_rank][column_rank-1]
                assert self.A_ij[0] is None or self.A_ij[0].shape == self.A_i0.shape
            else:
                assert self.A_ij[0] is None

            if column_rank < self.tiles_per_side-1:
                # j=i+1 exists
                self.A_ij[2] = blocks[column_rank][column_rank+1]
                assert self.A_ij[2] is None or self.A_ij[2].shape == self.A_i0.shape
        else:
            # Row tile
            self.A_0i = blocks[0][rank]

        self._optimize_Ai_slices()

    def zero_rhs(self, number_of_rows_per_rank: int, number_of_columns: int, dtype=np.float32):
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

        # Create empty buffers for C tile (if applicable)
        tile_shape = (number_of_rows_per_rank, number_of_columns)
        self.C_i = np.zeros(tile_shape, dtype=dtype)
        self.X_ij = [None, np.zeros(tile_shape, dtype=dtype), None]
        if self.comm.Get_rank() < self.tiles_per_side:
            self.C_0 = np.zeros(tile_shape, dtype=dtype)

        if self.is_column_rank():
            self.X_ij[0] = np.zeros(tile_shape, dtype=dtype)
            self.X_ij[2] = np.zeros(tile_shape, dtype=dtype)
            self.X_0 = np.zeros(tile_shape, dtype=dtype)


    @staticmethod
    def column_subgroup(tiles_per_side: int, group: MPI.Group) -> MPI.Group:
        """
        Given a group that contains all ranks for this matrix, return the column subgroup.
        :param group: contains all ranks for this matrix
        :return:
        """
        if tiles_per_side == 1:
            return group
        else:
            return group.Range_excl([(1, tiles_per_side - 1, 1)])

    @staticmethod

    def row_subgroup(tiles_per_side: int, group: MPI.Group) -> MPI.Group:
        """
        Given a group that contains all ranks for this matrix, return the row subgroup
        :param tiles_per_side: number of ranks per side of the matrix (the matrix has 2*tiles_per_side-1 ranks)
        :param group: contains all ranks for this matrix
        :return:
        """
        if tiles_per_side == 1:
            return group
        else:
            return group.Range_excl([(tiles_per_side, 2 * tiles_per_side - 2, 1)])

    def allgather_result(self, C: np.array):
        """
        All-gathers the result.
        :param C:
        :return:
        """
        assert C is not None

        column_rank = self._column_rank(self.comm.Get_rank())
        rank = self.comm.Get_rank()
        if rank == 0 or rank >= self.tiles_per_side:
            assert self.C_i is not None
            assert C.size == self.C_i.size * self.tiles_per_side
            assert self.column_comm.Get_rank() >= 0
            assert C.dtype == self.C_i.dtype
            assert column_rank >= 0
            assert self.column_comm is not None
            assert self.column_comm.Get_size() == self.tiles_per_side
            self.column_comm.Gather(self.C_i, C, root=0)

        #self.comm.Barrier()
        self.comm.Bcast(C, 0)
        assert C is not None
        return C

    # Deprecated
    def set_features_slice_from_features(self, X: np.ndarray):
        if self.is_column_rank():
            column_rank = self._column_rank(self.comm.Get_rank())
            # corresponding X block
            assert X.shape[0] % self.tiles_per_side == 0
            rows_per_rank = X.shape[0] // self.tiles_per_side
            self.set_features(X[column_rank * rows_per_rank:(column_rank + 1) * rows_per_rank, :])

    def _optimize_Ai_slices(self, threshold = 0.3):
        def nonzero_column_indices(matrix):
            return np.unique(matrix.nonzero()[1])

        # For each matrix that has fewer that threshold * n columns, we can optimize the slice
        if self.A_i0 is not None:
            nnz_columns = nonzero_column_indices(self.A_i0)
            if len(nnz_columns) < threshold * self.A_i0.shape[1]:
                self.A_i0 = self.A_i0[:, nnz_columns]
                self.A_i0_nnz_columns = nnz_columns

        if self.A_ij[1] is not None:
            nnz_columns = nonzero_column_indices(self.A_ij[1])
            if len(nnz_columns) < threshold * self.A_ij[1].shape[1]:
                self.A_ij[1] = self.A_ij[1][:, nnz_columns]
                self.A_ij_1_nnz_columns = nnz_columns

        if self.A_0i is not None:
            nnz_columns = nonzero_column_indices(self.A_0i)
            if len(nnz_columns) < threshold * self.A_0i.shape[1]:
                self.A_0i = self.A_0i[:, nnz_columns]
                self.A_0i_nnz_columns = nnz_columns



    def _distribute_arrays(self, A: np.array, X: np.array, b: int, root: int = 0):
        """
        Tiles and X and distributes it to the correct ranks.
        TODO USE THIS MAINLY FOR DEBUGGING.
        This method is very slow as it broadcasts A and X and tiles them locally.
        :param A: Buffer for A. Assumes that A is padded to a multiple of b
        :param X: Buffer for X
        :param b: arrow-width of A
        :param root: The rank of the root that holds the data to distribute
        :return:
        """
        # Assumes that A is padded to a multiple of b
        assert A.shape[0] % b == 0
        assert A.shape[1] == A.shape[0]
        assert X.shape[0] == A.shape[1]

        self.comm.Bcast(A, root)
        self.comm.Bcast(X, root)

        k = X.shape[1]
        A_tile_shape = (b, b)
        X_tile_shape = (b, k)

        rank = self.comm.Get_rank()
        column_rank = self._column_rank(rank)

        self.X_ij = [np.zeros(X_tile_shape, dtype=np.float32), np.zeros(X_tile_shape, dtype=np.float32), np.zeros(X_tile_shape, dtype=np.float32)]

        # Create empty buffers for C tile (if applicable)
        self.C_i = np.zeros(X_tile_shape, dtype=np.float32)
        if rank < self.tiles_per_side:
            self.C_0 = np.zeros(X_tile_shape, dtype=np.float32)

        if column_rank > 0:
            # corresponding X block
            self.X_ij[1] = X[column_rank*b:(column_rank+1)*b, :]
            # empty buffer
            self.X_0 = np.zeros(X_tile_shape, dtype=np.float32)
            # first block of row
            self.A_i0 = sparse.csr_matrix(A[column_rank*b:(column_rank+1)*b, 0:b], shape=(b, b))
            assert self.A_i0.shape == A_tile_shape

            # j=i exists
            self.A_ij[1] = sparse.csr_matrix(A[column_rank*b:(column_rank+1)*b, column_rank*b:(column_rank+1)*b], shape=(b, b))
            assert self.A_ij[1].shape == A_tile_shape

            if column_rank > 1:
                # j=i-1 exists if column index is at least 2
                self.A_ij[0] = sparse.csr_matrix(A[column_rank * b:(column_rank + 1) * b, (column_rank - 1) * b:column_rank * b], shape=(b, b))
                assert self.A_ij[0].shape == A_tile_shape

            if column_rank < self.tiles_per_side-1:
                # j=i+1 exists
                self.A_ij[2] = sparse.csr_matrix(A[column_rank * b:(column_rank + 1) * b, (column_rank + 1) * b:(column_rank + 2) * b], shape=(b, b))
                assert self.A_ij[2].shape == A_tile_shape

        else:
            # Row tile
            self.A_0i = sparse.csr_matrix(A[0:b, rank*b:(rank+1)*b], shape=(b, b))
            assert self.A_0i.shape == A_tile_shape

        # Initialize X_0
        if rank == 0:
            self.X_ij[1] = X[0:b, :].copy()
            assert self.X_ij[1].shape == X_tile_shape

        self.comm.Barrier()
