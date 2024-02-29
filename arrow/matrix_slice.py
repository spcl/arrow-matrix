import numpy as np
from mpi4py import MPI


class MatrixSlice:
    """
    A 1D slice of a matrix for the spmmPETSc implementation. This entails both a local and a non-local slice.
    Encapsulates the communication tables needed to communicate the non-local slice.
    Also encapsulates the number of columns in each rank.
    """
    def __init__(self,
                 A_i_local,
                 A_i_nonlocal,
                 x_index_in: np.ndarray,
                 rank_in: np.ndarray,
                 x_index_out: np.ndarray,
                 rank_out: np.ndarray,
                 all_n_i: np.ndarray,
                 start_col: int,
                 end_col: int,
                 send_count: np.ndarray,
                 recv_count: np.ndarray) -> None:
        """
        @param A_i_local: The local slice of A_i
        @param A_i_nonlocal: The non-local slice of A_i
        @param x_index_in: The indices of the rows of X that need to be received (sorted)
        @param rank_in: The ranks from which the rows of X_i need to be received (sorted by corresponding x_index_in)
        @param x_index_out: The indices of the rows of X_i that need to be sent (sorted)
        @param rank_out: The ranks to which the rows of X_i need to be sent (sorted by corresponding x_index_out)
        @param all_n_i: The number of rows in each rank's slice
        @param start_col: The start column of the local slice of A_i
        @param end_col: The end column of the local slice of A_i
        @param send_count: The number of rows sent to each rank
        @param recv_count: The number of rows received from each rank
        """
        # assert A_i_local and A_i_nonlocal have the same number of rows
        assert A_i_local.shape[0] == A_i_nonlocal.shape[0]
        # assert A_i_local is square
        assert A_i_local.shape[0] == A_i_local.shape[1]
        # Assert the x_index_in and rank_in are the same size
        assert x_index_in.shape[0] == rank_in.shape[0]
        # Assert the x_index_out and rank_out are the same size
        assert x_index_out.shape[0] == rank_out.shape[0]
        # Assert the x_index_in and number of columns in A_i_nonlocal is the same
        assert x_index_in.shape[0] == A_i_nonlocal.shape[1]
        # assert sorted
        assert MatrixSlice._is_sorted(x_index_in)
        assert MatrixSlice._is_sorted(rank_out)
        assert MatrixSlice._is_sorted(rank_in)

        self.A_i_local = A_i_local
        self.A_i_nonlocal = A_i_nonlocal
        # These are global indices into the overall X matrix
        self.x_index_in = x_index_in
        self.rank_in = rank_in
        # These are global indices into the overall X_i matrix
        self.x_index_out = x_index_out
        self.rank_out = rank_out
        self.all_n_i = all_n_i
        self.start_col = start_col
        self.end_col = end_col

        # These are local indices into the local X_i matrix
        self.x_index_out_localized = x_index_out - start_col

        start_col_all_ranks = np.cumsum(all_n_i)
        # prepend 0
        start_col_all_ranks = np.insert(start_col_all_ranks, 0, 0)

        # These are local indices into the local X matrix
        # Using numpy:
        self.x_index_in_localized = x_index_in - start_col_all_ranks[rank_in]

        self.send_count = send_count
        self.recv_count = recv_count

        self.send_sdispl = np.cumsum(send_count)
        self.send_sdispl = np.insert(self.send_sdispl, 0, 0)
        self.recv_sdispl = np.cumsum(recv_count)
        self.recv_sdispl = np.insert(self.recv_sdispl, 0, 0)

    @staticmethod
    def get_local_matrix_dimensions(comm, A_i) -> np.ndarray:
        """
        All-gather the number of rows in each rank's slice to compute local slice range.
        @param comm: The MPI communicator
        @param A_i: The local slice of the matrix
        @return: A list of the number of rows in each rank's slice
        """
        n_i = A_i.shape[0]
        all_n_i = comm.allgather(n_i)
        return np.asarray(all_n_i)

    @staticmethod
    def identify_local_slice(rank, all_n_i) -> tuple[int, int]:
        """
        Compute the start and end indices for the local slice of columns.
        @param rank: The rank of the local slice
        @param all_n_i: A list of the number of rows in each rank's slice
        @return: The start and end indices for the local slice of columns
        """
        start_col = sum(all_n_i[:rank])
        end_col = sum(all_n_i[:rank + 1])
        return start_col, end_col

    @classmethod
    def initialize(cls, comm, A_i) -> 'MatrixSlice':
        """
        Initialize the communication tables for the given rank.
        @param comm: The MPI communicator
        @param A_i: The local slice of the matrix
        @return: A MatrixSlice object encapsulating the communication tables
        """
        rank = comm.Get_rank()

        # Get the local matrix dimensions
        all_n_i = cls.get_local_matrix_dimensions(comm, A_i)
        total_rows = sum(all_n_i)
        if total_rows != A_i.shape[1]:
            raise ValueError(
                f"Matrix not square: Rank {rank} has {A_i.shape[1]} columns, but the total number of rows is {total_rows}")

        # Identify local slice
        start_col, end_col = cls.identify_local_slice(rank, all_n_i)

        # Extract the local slice of A_i
        A_i_local = A_i[:, start_col:end_col]
        A_i_local.sort_indices()
        A_i_local.sum_duplicates()
        A_i_local.eliminate_zeros()

        # Construct receive communication arrays
        non_local_columns, x_index_in, rank_in = MatrixSlice.construct_receive_tables(A_i, start_col, end_col, all_n_i)

        # Compute the receive counts using numpy:
        recv_counts = np.bincount(rank_in, minlength=comm.Get_size())

        # Construct send tables
        x_index_out, rank_out, send_counts = MatrixSlice.construct_send_tables(comm, rank_in, x_index_in, recv_counts)

        assert cls.check_comm_tables(comm, x_index_in, rank_in, x_index_out, rank_out)

        # Construct the non-local slice of A_i
        # It is given by the columns of the non-local columns:
        A_i_nonlocal = A_i[:, non_local_columns]
        A_i_local.sort_indices()
        A_i_local.sum_duplicates()
        A_i_local.eliminate_zeros()

        assert A_i_nonlocal.shape[1] == len(x_index_in)

        comm.Barrier()

        return cls(A_i_local, A_i_nonlocal, x_index_in, rank_in, x_index_out, rank_out, all_n_i, start_col, end_col, send_counts, recv_counts)


    @staticmethod
    def check_comm_tables(comm, x_index_in, rank_in, x_index_out, rank_out) -> bool:
        """
        Check the communication tables for consistency
        @param comm: The MPI communicator
        @param x_index_in: The indices of the rows of X that need to be received (sorted)
        @param rank_in: The ranks from which the rows of X_i need to be received (sorted by corresponding x_index_in)
        @param x_index_out: The indices of the rows of X_i that need to be sent (sorted)
        @param rank_out: The ranks to which the rows of X_i need to be sent (sorted by corresponding x_index_out)
        """

        # Count the number of rows sent to each rank
        send_counts = np.bincount(rank_out, minlength=comm.Get_size())
        # Count the number of rows received from each rank
        recv_counts = np.bincount(rank_in, minlength=comm.Get_size())

        # All-to-all communication to share counts
        recv_counts_c = comm.alltoall(list(send_counts))
        # All-to-all communication to share counts
        send_counts_c = comm.alltoall(list(recv_counts))

        # Check that the counts are the same
        assert list(send_counts) == send_counts_c
        assert list(recv_counts) == recv_counts_c

        return list(send_counts) == send_counts_c and list(recv_counts) == recv_counts_c

    @staticmethod
    def construct_receive_tables(A_i, start_col: int, end_col: int, all_n_i: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Construct x_index_in and rank_in arrays for non-local columns.
        @param A_i: The local slice of the matrix
        @param start_col: The start column of the local slice of the matrix
        @param end_col: The end column of the local slice of the matrix
        @param all_n_i: A list of the number of rows in each rank's slice
        @return: nonlocal_cols, x_index_in, rank_in,
        where x_index_in is the index of the row in the local slice of X_i that needs to be received and
        rank_in is the rank to which the row needs to be sent.
        The arrays are sorted by x_index_in.
        """
        # Identify the non-local columns, which are the columns that are not in the local slice
        # Do the same, but in two steps, slice A_i up to start col, then from end col to end
        first_slice = A_i[:, :start_col].nonzero()[1]
        assert np.all(first_slice < start_col)
        second_slice = A_i[:, end_col:].nonzero()[1] + end_col
        assert np.all(second_slice >= end_col)
        assert np.all(second_slice < A_i.shape[1])
        both_slices = np.concatenate((first_slice, second_slice))
        nonlocal_cols = np.unique(both_slices)

        x_index_in = np.zeros(nonlocal_cols.size, dtype=np.int64)
        rank_in = np.zeros(nonlocal_cols.size, dtype=np.int64)

        # Identify the rank and x_index for each non-local column
        # An alternative way would be to vectorize the comparisons (but then p many comparisons are needed)
        cumulative_sum = np.cumsum(all_n_i)
        r = 0
        i = 0
        for col in nonlocal_cols:
            while col >= cumulative_sum[r]:
                r += 1
            assert col < cumulative_sum[r]
            x_index_in[i] = col
            rank_in[i] = r
            i += 1
            assert col == x_index_in[i-1]

        assert x_index_in.size == rank_in.size
        assert x_index_in.size == nonlocal_cols.size

        return nonlocal_cols, x_index_in, rank_in

    @staticmethod
    def _is_sorted(a):
        return np.all(a[:-1] <= a[1:])

    @staticmethod
    def construct_send_tables(comm, rank_in: np.ndarray, x_index_in: np.ndarray, recv_counts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Construct the send tables for all-to-all-v communication
        @param comm: The MPI communicator
        @param rank_in: The ranks from which the rows of X_i need to be received (sorted)
        @param x_index_in: The indices of the rows of X that need to be received (sorted)
        @param recv_counts: The number of rows received from each rank
        @return: x_index_out, rank_out (sorted by rank_out), send_counts
        """
        assert MatrixSlice._is_sorted(x_index_in)
        assert MatrixSlice._is_sorted(rank_in)

        # All-to-all communication to share counts
        send_counts = np.zeros(comm.Get_size(), dtype=np.int64)
        comm.Alltoall(recv_counts, send_counts)

        send_buffer = np.zeros(np.sum(send_counts), dtype=np.int64)

        comm.Alltoallv([x_index_in, recv_counts, MPI.INT64_T], [send_buffer, send_counts, MPI.INT64_T])

        # Convert recv_counts + recv_buffer in a form x_index_out, rank_out
        # Where x_index_out is the index of the row in the local slice of X_i that needs to be sent to rank_out
        # and rank_out is the rank to which the row needs to be sent
        x_index_out = []
        rank_out = []
        j = 0
        for r, count in enumerate(send_counts):
            for i in range(count):
                x_index_out.append(send_buffer[j+i])
                rank_out.append(r)
            j += count

        # Sort the x_index_out and rank_out arrays by first element, then second element
        s = sorted(zip(x_index_out, rank_out), key=lambda x: (x[1], x[0]))
        if len(s) > 0:
            x_index_out, rank_out = zip(*s)

        assert MatrixSlice._is_sorted(rank_out)

        return np.asarray(x_index_out, dtype=np.int64), np.asarray(rank_out, dtype=np.int64), send_counts
