import sys
import time
from typing import List, Union
import numpy
import numpy as np
from mpi4py.MPI import Request

from mpi4py import MPI
from scipy import sparse
from arrow.common import wb_logging, graphio
from arrow.arrow_mpi import ArrowMPI
from arrow.arrow_matrix import ArrowMatrix
from arrow.arrow_slim_mpi import ArrowSlimMPI

try:
    import wandb
except Exception as e:
    pass


class ArrowDecompositionMPI:
    # Holds the slices and communicators for the 'own' part of the decomposition
    # Specifically, the matrix at position self.matrix_index
    B: ArrowMPI

    # Index of the matrix that the current rank is responsible for
    matrix_index: int
    # Number of matrices in the decomposition
    decomposition_length: int

    # Communicator for the whole computation (including other matrices)
    comm: MPI.Comm

    # Device to use (cpu or gpu)
    device: str

    # Number of rows per rank (MUST be equal in all ranks) - use padding. Must match B.
    _n_rows_per_rank: int
    # Number of columns in B.C_i
    _n_feature_columns: int

    # Aggregator that contains the column processors of two neighboring matrices (self.matrix_index and self.matrix_index-1)
    _backward_aggregation_comm:  Union[MPI.Comm, None] = None
    _recvbuf: Union[np.ndarray, None] = None
    _sendbuf: Union[np.ndarray, None] = None
    _backward_recvcounts = None
    _backward_rdispls = None
    _backward_sendcounts = None
    _backward_sdispls = None

    # Maps the current order of C_i to one that is ordered by backwards receiving processor
    _back_send_permutation: Union[np.ndarray, None] = None
    _back_receive_permutation: Union[np.ndarray, None] = None

    # Aggregator that contains the column processors of two neighboring matrices (self.matrix_index and self.matrix_index+1)
    _forward_aggregation_comm: Union[MPI.Comm, None] = None
    _forward_recvcounts = None
    _forward_rdispls = None
    _forward_sendcounts = None
    _forward_sdispls = None

    # Maps the current order of C_i to one that is ordered by forwards receiving processor
    _forward_send_permutation: Union[np.ndarray, None] = None
    # A list of indexes into the own X slice that can be updated by the received data (in this order)
    _forward_receive_permutation: Union[np.ndarray, None] = None

    _empty_buf: np.ndarray = None
    _back_zero_list: List[int] = None
    _forward_zero_list: List[int] = None

    def __init__(self, comm: MPI.Comm,
                 B: ArrowMatrix,
                 matrix_index: int,
                 number_of_rows_per_rank: int,
                 number_of_feature_columns: int,
                 groups: list[MPI.Group],
                 to_previous_permutation,
                 to_next_mapping,
                 device='cpu',
                 slim=False):
        """
        :param comm: Communicator containing *all* participating matrices' ranks
        :param B: Own rank's matrix. Does not have to contain any data yet.
        :param matrix_index: The index of the own matrix in the decomposition
        :param number_of_rows_per_rank: The number of rows per rank. Must match B.
        :param number_of_feature_columns: The number of columns in the result of the multiplication
        :param to_previous_permutation: Permutation that maps the own rows to rows in the previous matrix
        :param to_next_mapping: Permutation that maps the own rows to rows in the next matrix
        :param device: Either 'cpu' or 'gpu'.
        :param slim: If true, uses one rank per block. Else, row and column arrow is split.
        """
        assert comm.Get_size() >= sum([g.Get_size() for g in groups])

        self.comm = comm
        self.B = B
        self.matrix_index = matrix_index
        self.decomposition_length = len(groups)
        self._n_rows_per_rank = number_of_rows_per_rank
        self._n_feature_columns = number_of_feature_columns
        self.device = device
        self.slim = slim

        self._initialize_communicators(comm, groups)
        self._initialize_all_to_all_tables(to_previous_permutation, to_next_mapping)

    @staticmethod
    def initialize(comm: MPI.Comm,
                   n_blocks: np.ndarray,
                   to_prev_permutation: np.ndarray,
                   to_next_permutation: np.ndarray,
                   rows_per_rank: int,
                   feature_columns: int,
                   device='cpu',
                   block_diagonal: bool = True,
                   slim: bool = False):
        """
        Factory method to create a usable ArrowDecompositionMPI instance.
        This is the preferred way to instantiate instances as it takes care of the splitting of the communicator.
        If None is returned, the current process does not participate in the computation.
        :param block_diagonal: If true, assumes a arrow-block-diagonal shape. if false, assumes a arrow-banded shape
        :param device: Either 'cpu' or 'gpu'.
        :param to_next_permutation: For each row of this processor, maps to the row-index of the next matrix that corresponds to it.
        :param to_prev_permutation: For each row of this processor, maps to the row-index of the previous matrix that corresponds to it.
        :param comm: communicator for the computation.
        :param n_blocks: For each i, contains the number of blocks per side of the i-th matrix in the decomposition.
        :param rows_per_rank: number of rows in each processor
        :param feature_columns: number of feature columns (dense right-hand side columns)
        :param slim: If true, uses one rank per block. Else, row and column arrow is split.
        :return: Either A usable ArrowDecompositionMPI instance or None if the current rank does not participate.
        """
        assert not slim or block_diagonal

        # Adjust group size based on if the underlying arrow matrix is slim or not
        group_sizes = list(n_blocks * 2 - 1) if not slim else list(n_blocks)
        total_blocks = sum(group_sizes)

        assert comm.Get_size() >= total_blocks
        assert total_blocks > 0

        total_subgroup = comm.Get_group().Range_incl([(0, total_blocks - 1, 1)])
        subcomm = comm.Create(total_subgroup)

        if comm.Get_rank() > total_blocks - 1:
            return None

        groups = []
        so_far = 0
        rank = subcomm.Get_rank()
        own_matrix_index = -1
        for i, count in enumerate(group_sizes):
            groups.append(total_subgroup.Range_incl([(so_far, so_far + count - 1, 1)]))
            so_far += count
            if so_far > rank and own_matrix_index == -1:
                own_matrix_index = i
                assert groups[i].Get_rank() >= 0

        assert own_matrix_index >= 0
        assert all([g.Get_size() == x for (g, x) in zip(groups, group_sizes)])

        own_group = groups[own_matrix_index]

        if slim:
            B: ArrowMatrix = ArrowSlimMPI(MPI.Comm.Create(subcomm, own_group))
        else:
            B: ArrowMatrix = ArrowMPI(MPI.Comm.Create(subcomm, own_group), block_diagonal)

        arrow: ArrowDecompositionMPI = ArrowDecompositionMPI(subcomm,
                                                             B,
                                                             own_matrix_index,
                                                             rows_per_rank,
                                                             feature_columns,
                                                             groups,
                                                             to_prev_permutation,
                                                             to_next_permutation,
                                                             device=device)

        return arrow

    def load_data_from_blocks(self, blocked: List[List[List[sparse.csr_matrix]]]):
        assert len(blocked) == self.decomposition_length
        self.B.load_sparse_matrix_from_blocks(blocked[self.matrix_index])

    def _initialize_communicators(self, comm: MPI.Comm, groups: list[MPI.Group]):
        """
        Initializes self._forward_aggregation_comm and self._backward_aggregation_comm
        :param comm: communicator that holds all the ranks in the groups
        :param groups: The groups that correspond to the matrices.
        :return:
        """
        assert len(groups) == self.decomposition_length

        group_sizes = [g.Get_size() for g in groups]

        tiles_per_side = (np.asarray(group_sizes) + 1) // 2. if not self.slim else np.asarray(group_sizes)

        self._forward_aggregation_comm = None
        self._backward_aggregation_comm = None

        for matrix_idx in range(len(groups) - 1):
            current_col_group = self.B.column_subgroup(tiles_per_side[matrix_idx], groups[matrix_idx])
            next_col_group = self.B.column_subgroup(tiles_per_side[matrix_idx + 1], groups[matrix_idx + 1])

            aggregation_group = MPI.Group.Union(current_col_group, next_col_group)
            aggregation_comm = comm.Create(aggregation_group)
            if self.matrix_index == matrix_idx:
                self._forward_aggregation_comm = aggregation_comm
            elif self.matrix_index == matrix_idx + 1:
                self._backward_aggregation_comm = aggregation_comm

    def _initialize_all_to_all_tables(self,
                                      to_previous_permutation: Union[np.ndarray, None],
                                      to_next_mapping: Union[np.ndarray, None],
                                      ) -> None:
        """
        Initializes the necessary communicators to aggregate sub-results.
        :param to_previous_permutation: Permutation that maps the own rows to rows in the previous matrix
        :param to_next_mapping: Permutation that maps the own rows to rows in the next matrix
        :return:
        """
        if not self.B.is_column_rank():
            return

        # Set up all-to-all-tables to previous matrix
        if self.matrix_index > 0:
            assert to_previous_permutation is not None
            assert to_previous_permutation.size == self._n_rows_per_rank
            assert self._backward_aggregation_comm.Get_rank() >= 0

            back_group_size = self._backward_aggregation_comm.Get_size()

            self._back_zero_list = [0 for _ in range(back_group_size)]

            self._backward_sendcounts, self._backward_sdispls, self._back_send_permutation, self._back_receive_permutation = self._all_to_all_tables(
                to_previous_permutation,
                self._n_rows_per_rank,
                self._n_feature_columns,
                back_group_size,
                0)

            assert len(self._back_zero_list) == self._backward_aggregation_comm.Get_size()
            assert self._backward_sdispls is not None
            assert self._backward_sendcounts is not None
            assert self._back_send_permutation is not None
            assert self._back_receive_permutation is not None
            assert self._backward_sendcounts[self._backward_aggregation_comm.rank] == 0

        else:
            self._backward_sendcounts, self._backward_sdispls, self._back_send_permutation, self._back_receive_permutation = None, None, None, None

        # Set up all-to-all-tables to next matrix
        if self.matrix_index < self.decomposition_length - 1:

            assert self._forward_aggregation_comm.Get_rank() >= 0
            assert to_next_mapping is not None
            assert to_next_mapping.size == self._n_rows_per_rank

            forward_group_size = self._forward_aggregation_comm.Get_size()
            own_group_size = self.B.column_comm.Get_size()

            self._forward_zero_list = [0 for _ in range(forward_group_size)]

            self._forward_recvcounts, self._forward_rdispls, self._forward_send_permutation, self._forward_receive_permutation = self._all_to_all_tables(
                to_next_mapping,
                self._n_rows_per_rank,
                self._n_feature_columns,
                forward_group_size,
                own_group_size)

            assert self._forward_recvcounts is not None
            assert self._forward_rdispls is not None
            assert self._forward_send_permutation is not None
            assert self._forward_receive_permutation is not None
            assert self._forward_recvcounts[self._forward_aggregation_comm.rank] == 0
            assert np.sum(self._forward_recvcounts) <= self._n_rows_per_rank * self._n_feature_columns

        else:
            self._forward_recvcounts, self._forward_rdispls, self._forward_send_permutation, self._forward_receive_permutation = None, None, None, None

        self._empty_buf = np.empty(1, dtype=np.float32)
        self._recvbuf = np.empty((self._n_rows_per_rank, self._n_feature_columns), dtype=np.float32)
        self._sendbuf = np.empty((self._n_rows_per_rank, self._n_feature_columns), dtype=np.float32)

    def step(self):
        """
        Performs one SpMM iteration.
        Note that only the column processors of the first matrix hold C_i and X_ij[1] after completion.
        Precondition: A is properly distributed in all ranks
        Precondition: X is properly distributed in matrix index 0
        Postcondition: X := AX has been executed
        Postcondition: C_i has been updated (in the matrix at index 0 in the decomposition)
        Postcondition: X_ij[1] has been updated (in the matrix at index 0 in the decomposition)
        :return:
        """

        request = self._propagate_features()

        tic = time.perf_counter()
        # Performs one step of the SpMM iteration
        self.B.spmm(device=self.device)

        toc = time.perf_counter()
        wb_logging.log({'spmm_arrow_time': toc - tic})

        if request is not None:
            request.wait()

        self._aggregate()

    def _propagate_features(self) -> Union[Request or None]:
        """
        Propagates the features from the first matrix to all other matrices
        :return:
        """
        request = None
        if self.B.is_column_rank():
            # Broadcasts the new C into the X at matrices[i] with i>0.
            # Essentially the reverse pattern from the backwards
            tic = time.perf_counter()
            request = self._propagate_features_forwards()
            toc = time.perf_counter()
            wb_logging.log({"spmm_bcast_time": toc - tic})

        return request

    @staticmethod
    def _all_to_all_tables(out_permutation: numpy.ndarray, rows_per_rank: int, n_columns: int,
                           total_ranks: int, put_offset: int = 0) -> (List[int], List[int], np.ndarray, np.ndarray):
        """
        Computes tables for sending rows along a permutation.
        :param put_offset: The output send counts and displacement are put at the given offset. This makes sense
        if the "own" ranks come first.
        :param total_ranks: Number of ranks participating in the all to all.
        :param out_permutation: Indexes to route for. The i-th row is sent to / received from the permutation[i] row
        Note that values that "overflow" (are larger than rows_per_rank*total_ranks-1) are taken as dummy values that are ignored.
        :param rows_per_rank: Number of rows per rank. All ranks except the last *must* have this many rows.
        :param n_columns: Number of columns per row.
        :return: counts, displacements, and the permutations to apply to permutation to ensure it's sorted by rank, and the permutation to apply when inverting the communication pattern
        """
        assert out_permutation.size == rows_per_rank
        assert put_offset < total_ranks
        assert n_columns > 0
        assert total_ranks > 0

        ranks = np.floor_divide(out_permutation, rows_per_rank).astype(np.intp)

        # Send counts
        # TODO Vectorize (low priority)
        sendcounts = np.zeros(total_ranks, dtype=np.int64)

        for i in range(rows_per_rank):
            if ranks[i] + put_offset < total_ranks:
                sendcounts[int(put_offset + ranks[i])] += n_columns

        # Computes the displacements. These are just the cumulative sums of the sendcount, shifted by 1, and multiplied
        # by the feature width
        sdispls = np.zeros(total_ranks, dtype=np.int64)
        np.cumsum(sendcounts, dtype=np.int64, out=sdispls)
        sdispls = np.roll(sdispls, 1)
        sdispls[0] = 0

        # Sorting by rank is the permutation we need
        # Make it stable though!
        rank_permutation = np.argsort(ranks, kind='stable')

        aggregation_permutation = ArrowDecompositionMPI._aggregation_permutation(out_permutation, ranks, total_ranks)

        return list(sendcounts), list(sdispls), rank_permutation, aggregation_permutation

    @staticmethod
    def _aggregation_permutation(out_permutation: np.ndarray, out_ranks: np.ndarray, total_ranks: int):

        grouped = [[] for _ in range(total_ranks)]

        for i in range(len(out_permutation)):
            assert i < out_ranks.size
            if out_ranks[i] < total_ranks:
                grouped[int(out_ranks[i])].append((out_permutation[i], i))

        result = []
        for l in grouped:
            l.sort(key=lambda x: x[0])
            result.extend([x[1] for x in l])

        return np.asarray(result, np.intp)

    def _aggregate(self):
        """
        Aggregates the C_i into the next X_i (Logically and AllReduce of the rows of X, but through the permutations).
        Precondition: _initialize_all_to_all_tables has been called once
        :return:
        """
        if not self.B.is_column_rank():
            return

        assert self.matrix_index == 0 or self._backward_aggregation_comm is not None
        assert self.matrix_index == self.decomposition_length - 1 or self._forward_aggregation_comm is not None

        tic = time.perf_counter()
        # Reduces the partial sums of C into the C at the matrices[0]
        self._aggregate_features_backwards()
        toc = time.perf_counter()
        wb_logging.log({"spmm_reduce_time": toc - tic})

    def _aggregate_features_backwards(self):
        """
        Send the results from matrix_index i to the matrix_index i-1 and aggregates them, for decreasing i
        Precondition: _initialize_all_to_all_tables has been called once
        Precondition: The result is in self.B.C_i
        Postcondition: The aggregated (partial) result is in self.B.C_i and self.B.X_ij[1]
        :return:
        """
        assert self.B.is_column_rank()
        assert self.matrix_index == 0 or self._backward_aggregation_comm is not None

        for i in reversed(range(1, self.decomposition_length)):

            if i == self.matrix_index:

                # Group C_i by receiving processor
                tic = time.perf_counter()
                self._sendbuf = self.B.C_i[self._back_send_permutation]
                toc = time.perf_counter()
                wb_logging.log({'back_agg_send': toc - tic})
                # All-to-all-v
                tic = time.perf_counter()
                self._send_to_previous_matrix()
                toc = time.perf_counter()
                wb_logging.log({'back_agg_all_to_all_v': toc - tic})
            elif i - 1 == self.matrix_index:
                # All-to-all-v
                tic = time.perf_counter()
                self._receive_from_next_matrix()
                toc = time.perf_counter()
                wb_logging.log({'back_agg_all_to_all_v': toc - tic})
                tic = time.perf_counter()
                # Permute to correct position in C_i
                self.B.C_i[self._forward_receive_permutation] += self._recvbuf[0:self._forward_receive_permutation.size]
                self.B.set_features(self.B.C_i)
                toc = time.perf_counter()
                wb_logging.log({'back_agg_recv': toc - tic})

    def _send_to_previous_matrix(self):
        # NOTE: Because we are sending to the previous matrix, we can use the backward send tables and comm

        # If the current matrix has a single rank, use scatter
        if self.B.comm.Get_size() == 1:
            self._backward_aggregation_comm.Scatterv([
                self._sendbuf,
                self._backward_sendcounts,
                self._backward_sdispls,
                MPI.FLOAT
            ], [
                self._empty_buf,
                0,
                MPI.FLOAT
            ],
                root=self._backward_aggregation_comm.Get_size()-1
            )

        # Else, use all-to-all
        else:
            self._backward_aggregation_comm.Alltoallv([
                self._sendbuf,
                self._backward_sendcounts,
                self._backward_sdispls,
                MPI.FLOAT
            ], [
                # Note used here
                self._empty_buf,
                # These are 0
                self._back_zero_list,
                # These are 0
                self._back_zero_list,
                MPI.FLOAT
            ])

    def _receive_from_next_matrix(self):
        # NOTE: Because we are receiving from the next matrix, we use the forward aggregation comm and receive tables
        # If the next matrix has a single rank, use scatter
        if self._forward_aggregation_comm.Get_size() == self.B.column_comm.Get_size() + 1:
            self._forward_aggregation_comm.Scatterv(
                None,
                [
                    self._recvbuf,
                    self._forward_recvcounts[-1],
                    MPI.FLOAT
                ],
                root=self._forward_aggregation_comm.Get_size() - 1
            )
        # Else, use all-to-all
        else:
            self._forward_aggregation_comm.Alltoallv([
                # Note used here
                self._empty_buf,
                # These are 0
                self._forward_zero_list,
                # These are 0
                self._forward_zero_list,
                MPI.FLOAT
            ], [
                self._recvbuf,
                self._forward_recvcounts,
                self._forward_rdispls,
                MPI.FLOAT
            ])

    def _propagate_features_forwards(self) -> Union[Request or None]:
        """
        Send the X features forward from matrix i to the matrix_index i+1, for increasing i.
        Precondition: _initialize_all_to_all_tables has been called once
        Precondition: The matrix at index 0 has the features in self.B.X_ij[1]
        Postcondition: self.B.X_ij[1] contains the features from the matrix at index 0, in the right permutation
        :return:
        """
        assert self.matrix_index == self.decomposition_length - 1 or self._forward_aggregation_comm is not None
        assert self.B.C_i is not None
        assert self.B.is_column_rank()

        request = None
        for i in range(self.decomposition_length - 1):

            if i == self.matrix_index:
                assert self.B.feature_tile() is not None
                # Group C_i by receiving processor
                tic = time.perf_counter()
                self._sendbuf = self.B.feature_tile()[self._forward_send_permutation]
                toc = time.perf_counter()
                wb_logging.log({'forward_agg_send': toc - tic})
                # All-to-all-v
                tic = time.perf_counter()
                request = self._send_to_next_matrix()
                toc = time.perf_counter()
                wb_logging.log({'forward_agg_all_to_all_v': toc - tic})
            elif i + 1 == self.matrix_index:
                # All-to-all-v
                tic = time.perf_counter()
                request = self._receive_from_previous_matrix()
                if request is not None:
                    request.wait()
                toc = time.perf_counter()
                wb_logging.log({'forward_agg_all_to_all_v': toc - tic})
                # Permute to correct position in X_ij
                tic = time.perf_counter()
                self.B.C_i[self._back_receive_permutation] = self._recvbuf[0:self._back_receive_permutation.size]
                self.B.set_features(self.B.C_i)
                toc = time.perf_counter()
                wb_logging.log({'forward_agg_recv': toc - tic})
                request = None

        return request

    def _receive_from_previous_matrix(self) -> Union[Request, None]:
        # NOTE: Receiving from the previous matrix is like sending to the previous matrix
        # Hence, we can use the back aggregation comms and back send tables
        if self.B.comm.Get_size() == 1:
            # Fast path: if the current matrix has a single rank, use Gatherv
            return self._backward_aggregation_comm.Igatherv(
                [
                    self._empty_buf,
                    0,
                    MPI.FLOAT
                ], [
                    self._recvbuf,
                    self._backward_sendcounts,
                    self._backward_sdispls,
                    MPI.FLOAT
                ],
                self._backward_aggregation_comm.Get_size()-1
            )
        else:
            # Slow path: Use alltoallv
            return self._backward_aggregation_comm.Ialltoallv([
                self._empty_buf,
                self._back_zero_list,
                self._back_zero_list,
                MPI.FLOAT
            ], [
                self._recvbuf,
                self._backward_sendcounts,
                self._backward_sdispls,
                MPI.FLOAT
            ])

    def _send_to_next_matrix(self) -> Union[Request, None]:
        # Note: sending to the next matrix is like receiving from the next matrix
        # Hence, we can use the forward aggregation comm and receive tables
        if self._forward_aggregation_comm.Get_size() == self.B.column_comm.Get_size() + 1:
            # Fast path: if the next matrix has a single rank, use Gatherv
            return self._forward_aggregation_comm.Igatherv(
                [
                    self._sendbuf,
                    self._forward_recvcounts[-1],
                    MPI.FLOAT
                ],
                None,
                root=self._forward_aggregation_comm.Get_size()-1
            )
        else:
            # Slow path: Use Alltoallv
            return self._forward_aggregation_comm.Ialltoallv([
                self._sendbuf,
                self._forward_recvcounts,
                self._forward_rdispls,
                MPI.FLOAT
            ], [
                self._empty_buf,
                self._forward_zero_list,
                self._forward_zero_list,
                MPI.FLOAT
            ])

    @staticmethod
    def number_of_blocks(adjacency, width: int) -> int:
        if isinstance(adjacency, tuple):
            # Tuple of memory-mapped files
            indptr = adjacency[2]
            nnz_per_row = indptr[1:] - indptr[:-1]
            num_rows = indptr.size - 1
        else:
            nnz_per_row = adjacency.getnnz(1)
            num_rows = adjacency.shape[0]
        nnz_per_row = np.flip(nnz_per_row)
        # This works for symmetric matrices. Otherwise we also need to consider columns
        zero_rows = next(j for j, x in enumerate(nnz_per_row) if x > 0)
        nonzero_rows = num_rows - zero_rows
        number_of_blocks: int = int(np.ceil(nonzero_rows / width))
        return number_of_blocks

    @staticmethod
    def load_decomposition_new(comm: MPI.Comm,
                               filename: str,
                               width: int,
                               is_block_diagonal: bool,
                               datatype=np.float32,
                               slim=False,
                               use_mmap=False):
        """
        Loads a decomposed graph from file and distributes it to the processors in comm.
        The graph is loaded at the root and distributed from there.
        :param comm: communicator, needs enough ranks to hold the decomposition.
        :param filename: path to the decomposition, excluding with and block diagonal suffixes
        :param width: width of the arrow decomposition
        :param is_block_diagonal: if true, the decomposition has a single block around the diagonal.
        :param datatype: datatype of the matrix
        :param slim: if true, uses one rank per block. Else, row and column arrow is split. Note that slim implies it must be block diagonal.
        :param use_mmap: if true, memory-maps the graph from disk using npy files of the CSR matrix. Else, loads it into memory.
        If use_mmap is true, the graph should be stored as a set of files representing the CSR matrix (See graphio.py)
        :return: list_of_blocks, n_blocks, to_previous_permutation, to_next_permutation
        """
        assert not slim or is_block_diagonal

        rank = comm.Get_rank()

        decomposition_length = np.zeros(1, dtype=np.int32)

        decomposition = None
        to_previous_p = None
        to_next_p = None

        if rank == 0:
            if use_mmap:
                decomposition = graphio.load_decomposition_new(filename, width, block_diagonal=is_block_diagonal)
            else:
                decomposition = graphio.load_decomposition(filename, width, block_diagonal=is_block_diagonal)
            if len(decomposition) == 0:
                print("ERROR: decomposition with name ", filename, " and width ", width, "not found", flush=True, file=sys.stderr)

            decomposition_length[0] = len(decomposition)

        comm.Bcast(decomposition_length)

        n_blocks = np.zeros(decomposition_length[0], dtype=np.int32)

        if decomposition_length == 0:
            return None, n_blocks, None, None

        if rank == 0:
            for i, (adjacency, permutation) in enumerate(decomposition):
                # NOTE: This forces zero blocks to be cut
                print(f"Cutting the permutation; Part {i}")
                number_of_blocks = ArrowDecompositionMPI.number_of_blocks(adjacency, width)
                n_blocks[i] = number_of_blocks

        comm.Bcast(n_blocks)

        if slim:
            if np.sum(n_blocks) > comm.Get_size():
                return None, n_blocks, None, None
        elif np.sum(2 * n_blocks - 1) > comm.Get_size():
            return None, n_blocks, None, None

        if rank == 0:

            destination = 0

            permutations = [p for _, p in decomposition]

            one_based = np.min(permutations[0]) > 0

            # NEED TO PAD THE PERMUTATIONS!!!!
            number_of_rows = n_blocks[0] * width
            for i in range(len(permutations)):

                if one_based:
                    permutations[i] -= 1

                if permutations[i].size < number_of_rows:
                    old_size = permutations[i].size
                    permutations[i] = np.pad(permutations[i], (0, number_of_rows - old_size), constant_values=0)
                    permutations[i][old_size:] = np.arange(old_size, number_of_rows)
                assert permutations[i].size == number_of_rows

            inverse_permutations = [np.argsort(p) for p in permutations]

            rank_0_blocks: List[List[Union[sparse.csr_matrix, None]]] = [[None for _ in range(n_blocks[0])] for _ in
                                                                         range(n_blocks[0])]

            for i, (adjacency, permutation) in enumerate(decomposition):

                if use_mmap:
                    list_of_blocks: List[List[sparse.csr_matrix]] = graphio.split_matrix_to_blocks_new(adjacency, width)
                else:
                    list_of_blocks: List[List[sparse.csr_matrix]] = graphio.split_matrix_to_blocks(adjacency, width)

                # TRUNCATE LIST OF BLOCKS
                list_of_blocks = list_of_blocks[0: n_blocks[i]]
                for j in range(len(list_of_blocks)):
                    list_of_blocks[j] = list_of_blocks[j][0:n_blocks[i]]

                # Prepare the global permutations
                if i > 0:
                    # GENERATING PREVIOUS PERMUTATION

                    to_previous_permutation = (inverse_permutations[i - 1])[permutations[i]]
                    # assert all([permutations[i][j] == permutations[i - 1][to_previous_permutation[j]] for j in range(len(to_previous_permutation))])

                    to_previous_permutation = np.where(to_previous_permutation >= width * n_blocks[i - 1],
                                                       2 * width * n_blocks[0], to_previous_permutation)

                if i < len(decomposition) - 1:
                    # GENERATING NEXT PERMUTATION
                    to_next_permutation = (inverse_permutations[i + 1])[permutations[i]]
                    # assert all([permutations[i][j] == permutations[i + 1][to_next_permutation[j]] for j in range(len(to_next_permutation))])

                    to_next_permutation = np.where(to_next_permutation >= width * n_blocks[i + 1],
                                                   2 * width * n_blocks[0], to_next_permutation)

                assert len(list_of_blocks[0]) == len(list_of_blocks)

                # Send to appropriate ranks
                # ROW NUMBER 0 IS SPECIAL
                destination_save = destination
                row = 0
                for column in range(len(list_of_blocks)):
                    # send A_0i to destination

                    block = list_of_blocks[row][column]
                    if use_mmap:
                        block = graphio.load_block_from_bslice(adjacency, block, width)

                    if destination == 0:
                        block.sort_indices()
                        block.sum_duplicates()
                        block.has_canonical_format = True

                        rank_0_blocks[row][column] = block
                        # Slice my own permutation
                        if i < len(decomposition) - 1:
                            to_next_p = to_next_permutation[row * width: (row + 1) * width]

                    else:
                        ArrowDecompositionMPI._send_block(comm, block, destination)
                        # Slice and send permutations
                        if column == 0:
                            if i > 0:
                                ArrowDecompositionMPI._send_permutation(comm, to_previous_permutation[row * width: (row + 1) * width], destination)
                            if i < len(decomposition) - 1:
                                ArrowDecompositionMPI._send_permutation(comm, to_next_permutation[row * width: (row + 1) * width], destination)

                    destination += 1

                if slim:
                    assert destination > destination_save
                    destination = destination_save + 1
                # REMAINING ROWS
                for row in range(1, len(list_of_blocks)):

                    # send A_i0 to destination
                    block = list_of_blocks[row][0]
                    if use_mmap:
                        block = graphio.load_block_from_bslice(adjacency, block, width)
                    ArrowDecompositionMPI._send_block(comm, block, destination)

                    # send A_ii to destination
                    block = list_of_blocks[row][row]
                    if use_mmap:
                        block = graphio.load_block_from_bslice(adjacency, block, width)
                    ArrowDecompositionMPI._send_block(comm, block, destination)

                    # send A_ii-1 to destination
                    if row > 1 and not is_block_diagonal:
                        block = list_of_blocks[row][row - 1]
                        if use_mmap:
                            block = graphio.load_block_from_bslice(adjacency, block, width)
                        ArrowDecompositionMPI._send_block(comm, block, destination)

                    # send A_ii+1 to destination
                    if row < len(list_of_blocks) - 1 and not is_block_diagonal:
                        block = list_of_blocks[row][row + 1]
                        if use_mmap:
                            block = graphio.load_block_from_bslice(adjacency, block, width)
                        ArrowDecompositionMPI._send_block(comm, block, destination)

                    # Slice and send permutations
                    if i > 0:
                        ArrowDecompositionMPI._send_permutation(comm, to_previous_permutation[row * width: (row + 1) * width], destination)
                    if i < len(decomposition) - 1:
                        ArrowDecompositionMPI._send_permutation(comm, to_next_permutation[row * width: (row + 1) * width], destination)

                    destination += 1

            return rank_0_blocks, n_blocks, to_previous_p, to_next_p

        else:
            ###
            # RECEIVING
            ###

            destination = 0
            for i in range(decomposition_length[0]):

                list_of_blocks: List[List[Union[sparse.csr_matrix, None]]] = [[None for _ in range(n_blocks[i])] for _
                                                                              in range(n_blocks[i])]

                destination_save = destination
                row = 0
                for column in range(n_blocks[i]):
                    # send A_0i to destination
                    if rank == destination:
                        list_of_blocks[row][column] = ArrowDecompositionMPI._receive_block(comm, datatype)

                        if column == 0:
                            # Receive permutations
                            if i > 0:
                                to_previous_p = ArrowDecompositionMPI._receive_permutation(comm, list_of_blocks[row][0].shape[0])
                            if i < decomposition_length[0] - 1:
                                to_next_p = ArrowDecompositionMPI._receive_permutation(comm, list_of_blocks[row][0].shape[0])

                        if not slim or column == 0:
                            return list_of_blocks, n_blocks, to_previous_p, to_next_p

                    destination += 1

                if slim:
                    assert destination > destination_save
                    destination = destination_save + 1

                for row in range(1, n_blocks[i]):
                    if rank == destination:
                        # receive A_i0
                        list_of_blocks[row][0] = ArrowDecompositionMPI._receive_block(comm, datatype)

                        # receive A_ii
                        list_of_blocks[row][row] = ArrowDecompositionMPI._receive_block(comm, datatype)

                        # receive A_ii-1
                        if row > 1 and not is_block_diagonal:
                            list_of_blocks[row][row - 1] = ArrowDecompositionMPI._receive_block(comm, datatype)

                        # receive A_ii+1
                        if row < len(list_of_blocks) - 1 and not is_block_diagonal:
                            list_of_blocks[row][row + 1] = ArrowDecompositionMPI._receive_block(comm, datatype)

                        # Receive permutations
                        if i > 0:
                            to_previous_p = ArrowDecompositionMPI._receive_permutation(comm, list_of_blocks[row][0].shape[0])
                        if i < decomposition_length[0] - 1:
                            to_next_p = ArrowDecompositionMPI._receive_permutation(comm, list_of_blocks[row][0].shape[0])

                        return list_of_blocks, n_blocks, to_previous_p, to_next_p

                    destination += 1

        return None, n_blocks, to_previous_p, to_next_p

    @staticmethod
    def _send_permutation(comm, perm_slice, destination):
        buf = np.asarray(perm_slice, dtype=np.int64)
        comm.Send(buf, destination, tag=4)

    @staticmethod
    def _send_block(comm, block, destination):
        meta = np.zeros(2, dtype=np.int64)
        meta[0] = block.shape[0]
        meta[1] = block.getnnz()
        comm.Send(meta, destination, tag=0)
        comm.Send(block.data, destination, tag=1)
        comm.Send(block.indices, destination, tag=2)
        comm.Send(block.indptr, destination, tag=3)

    @staticmethod
    def _receive_block(comm, datatype, source: int = 0):
        meta = np.zeros(2, dtype=np.int64)
        # try:
        comm.Recv(meta, source, tag=0)
        # Create bufs
        data_buf = np.empty(meta[1], dtype=datatype)
        indices_buf = np.empty(meta[1], dtype=np.int32)
        pointer_buf = np.empty(meta[0] + 1, dtype=np.int32)

        comm.Recv(data_buf, source, tag=1)
        comm.Recv(indices_buf, source, tag=2)
        comm.Recv(pointer_buf, source, tag=3)

        # Crate sparse array
        block = sparse.csr_matrix((data_buf, indices_buf, pointer_buf), shape=(meta[0], meta[0]),
                                  dtype=datatype)
        block.sort_indices()
        block.sum_duplicates()
        block.has_canonical_format = True
        return block

    @staticmethod
    def _receive_permutation(comm, n_rows, source=0):
        buf = np.zeros(n_rows, dtype=np.int64)
        comm.Recv(buf, source=source, tag=4)
        return buf