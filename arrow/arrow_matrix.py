"""
Abstract class for arrow matrix
"""
from abc import abstractmethod, ABC
from typing import List

import numpy as np
from mpi4py import MPI
from scipy import sparse


class ArrowMatrix(ABC):

    # The number of tiles per side
    tiles_per_side: int

    @abstractmethod
    def result_tile(self):
        """
        Returns the result tile
        :return:
        """

    @abstractmethod
    def feature_tile(self):
        """
        Returns the feature tile
        :return:
        """

    @abstractmethod
    def spmm(self, device: str = 'cpu'):
        """
         Compute the SpMM
        :param device: 'cpu' for CPU and 'gpu' for GPU
        :return:
        """

    @abstractmethod
    def set_features(self, X: np.ndarray):
        """
        Sets a SLICE of features.
        NOTE: Does not copy the data, but instead sets a reference to X.
        :param X:
        :return:
        """

    @abstractmethod
    def load_sparse_matrix_from_blocks(self, blocks: List[List[sparse.csr_matrix]]):
        """
        Loads the sparse matrix from blocks
        :param blocks:
        :return:
        """

    @abstractmethod
    def is_column_rank(self) -> bool:
        """
        Returns true if this rank is a column rank
        :return:
        """

    @abstractmethod
    def zero_rhs(self, number_of_rows_per_rank: int, number_of_columns: int, dtype=np.float32):
        """
        Clears the feature matrix X and the result matrix C.
        You must call this before the first SpMM iteration to initialize the right buffers.
        :param number_of_rows_per_rank:
        :param number_of_columns:
        :param dtype:
        :return:
        """
    pass

    @abstractmethod
    def allgather_result(self, C: np.array):
        """
        All-gathers the result.
        :param C:
        :return:
        """
    pass

    @abstractmethod
    def set_features_slice_from_features(self, X: np.ndarray):
        """
        Sets a SLICE of features.
        NOTE: Does not copy the data, but instead sets a reference to X.
        :param X:
        :return:
        """
    pass

    @staticmethod
    @abstractmethod
    def column_subgroup(tiles_per_side, group: MPI.Group) -> MPI.Group:
        """
        Given a group that contains all ranks for this matrix, return the column subgroup.
        :param group: contains all ranks for this matrix
        :return:
        """

    @staticmethod
    @abstractmethod
    def row_subgroup(tiles_per_side, group: MPI.Group) -> MPI.Group:
        """
        Given a group that contains all ranks for this matrix, return the row subgroup
        :param tiles_per_side: number of ranks per side of the matrix
        :param group: contains all ranks for this matrix
        :return:
        """
