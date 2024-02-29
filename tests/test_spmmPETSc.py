import unittest
from mpi4py import MPI
import numpy as np
import scipy.sparse
from arrow import matrix_slice
from arrow.baseline.spmm_petsc import spmm_cpu
from arrow.matrix_slice import MatrixSlice

class TestSpmm(unittest.TestCase):

    def helper_run_test(self, comm, A, X, start_row, end_row):
        A_mine = A[start_row:end_row, :]
        X_i = X[start_row:end_row, :]
        k = X.shape[1]
        assert A_mine.shape[0] == X_i.shape[0]
        assert A_mine.shape[0] == end_row - start_row

        rank = comm.Get_rank()

        Y_i = np.zeros((A_mine.shape[0], k), dtype=X.dtype)

        # Create a MatrixSlice object
        try:
            A_slice = MatrixSlice.initialize(comm, A_mine)
        except ValueError as e:
            assert False, f"RANK {rank} FAILED TO INITIALIZE MATRIX SLICE"

        X_i_nonlocal = np.zeros((len(A_slice.rank_in), k), dtype=X.dtype)

        comm.Barrier()

        #result = A_slice.A_i_local @ X_i + A_slice.A_i_nonlocal @ X_i_nonlocal
        result = spmm_cpu(comm, A_slice, X_i, Y_i, X_i_nonlocal)

        result_full = A @ X
        result_full_mine = result_full[start_row:end_row, :]

        # Assert result is close to X_i
        if not np.allclose(result, result_full_mine):
            print(f"RANK {rank}", result_full_mine - result)

        self.assertTrue(np.allclose(result, result_full_mine))

    def test_spmm_unequal(self):
        np.random.seed(0)
        # Tests spmmPETSc.py with an identity matrix
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        rows_per_rank_0 = 33
        first_half = round(size / 2)

        for rows_per_rank_1 in range (0, 34):

            n = first_half * rows_per_rank_0 + (size - first_half) * rows_per_rank_1

            all_ni = np.array([rows_per_rank_0 if i < first_half else rows_per_rank_1 for i in range(size)])
            all_ni_prefix_sum = np.cumsum(all_ni)
            # prepend 0
            all_ni_prefix_sum = np.insert(all_ni_prefix_sum, 0, 0)

            for seed in range(10):
                for density in [0, 0.02, 0.05, 0.1]:
                    # Create a random matrix
                    A = scipy.sparse.rand(n, n, density=density, format='csr', random_state = 42 + seed)
                    # Create a random X matrix
                    k = 4
                    X = np.round(5*np.random.rand(n, k))
                    # Test spmm
                    self.helper_run_test(comm, A, X, all_ni_prefix_sum[rank], all_ni_prefix_sum[rank + 1])


    def test_spmm(self):
        np.random.seed(0)
        # Tests spmmPETSc.py with an identity matrix
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        rows_per_rank = 4
        n = size * rows_per_rank

        for seed in range(10):
            # Create a random matrix
            A = np.round(1 * scipy.sparse.rand(n, n, density=0.1, format='csr', random_state = 42 + seed))

            # Create a random X matrix
            k = 4
            X = np.round(5*np.random.rand(n, k))

            self.helper_run_test(comm, A, X, rank * rows_per_rank, (rank + 1) * rows_per_rank)


    def test_spmm_eye(self):
        # Tests spmmPETSc.py with an identity matrix
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        rows_per_rank = 16
        n = size * rows_per_rank

        # Create an identity matrix
        eye = scipy.sparse.eye(n, dtype=np.float64, format='csr')
        eye_mine = eye[rank * rows_per_rank:(rank + 1) * rows_per_rank, :]

        # Create a random matrix
        k = 8
        X_i = np.random.rand(rows_per_rank, k)
        Y_i = np.zeros((rows_per_rank, k))

        # Create a MatrixSlice object
        eye_mine_slice = MatrixSlice.initialize(comm, eye_mine)

        X_i_nonlocal = np.zeros((len(eye_mine_slice.rank_in), k))

        result = spmm_cpu(comm, eye_mine_slice, X_i, Y_i, X_i_nonlocal)

        # Assert result is close to X_i
        self.assertTrue(np.allclose(result, X_i))




if __name__ == '__main__':
    unittest.main()
