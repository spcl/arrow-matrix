import argparse
import numpy as np

from scipy import sparse
from typing import Dict, Tuple, Union


# From https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool(v: Union[str, bool]) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def relabel_nodes(g: Union[sparse.csr_array, sparse.csr_matrix], mapping: Dict[int, int]) -> Union[sparse.csr_array, sparse.csr_matrix]:
    """ Relabels the nodes of the graph g according to a given mapping.
    
    The graph must be given as a SciPy-compatible CSR array or matrix.
    The labels of the nodes must be integers, corresponding to the rows of the matrix.

    :param g: The graph to relabel.
    :param mapping: The mapping from old to new labels.
    """

    # Validate the graph.
    if not isinstance(g, (sparse.csr_array, sparse.csr_matrix)):
        raise TypeError("The graph must be a SciPy-compatible CSR array or matrix.")
    if g.shape[0] != g.shape[1]:
        raise ValueError("The matrix must be square.")

    # Validate the relabeling mapping.
    labels = list(range(g.shape[0]))
    if sorted(mapping.keys()) != labels:
        raise ValueError("The keys of the mapping must be the rows of the graph's matrix representation.")
    if sorted(mapping.values()) != labels:
        raise ValueError("The values of the mapping must be the rows of the graph's matrix representation.")
    
    # Prepare the permutation matrix.
    order = [mapping[i] for i in range(g.shape[0])]
    I: sparse.coo_matrix = sparse.eye(g.shape[0], format='coo', dtype=np.int32)
    I.row = I.row[order]
    I = I.tocsr()
    # Relabel the nodes.
    g_prime = I @ g @ I.T

    return g_prime


def time_to_ms(runtime: float) -> int:
    return int(runtime * 1000)


def mpi_print(rank: int, msg: str):
    if rank == 0:
        print(msg, flush=True)


def generate_sparse_matrix(rows: int, cols: int, nnz: int, dtype: np.dtype,
                           rng: np.random.Generator) -> sparse.csr_matrix:
    """ Generates a sparse matrix in CSR format with the given number of non-zero elements.

    NOTE: The number of non-zero elements may be slightly larger than the requested number due to rounding.

    :param rows: The number of rows in the matrix.
    :param cols: The number of columns in the matrix.
    :param nnz: The number of non-zero elements in the matrix.
    :param dtype: The data type of the matrix.
    :param rng: The random number generator to use.
    :return: The generated sparse matrix.
    """
    # density = nnz / (rows * cols)
    # return sparse.random(rows, cols, density=density, format='csr', dtype=dtype, random_state=rng)
    # NOTE: The following avoids issues with sparse.random failing due to overflowing indices
    nnzpr = int(np.ceil(nnz / rows))
    actual_nnz = nnzpr * rows
    data = rng.random((actual_nnz, ), dtype=dtype)
    indptr = np.arange(0, actual_nnz + 1, nnzpr, dtype=np.int64)
    indices = rng.integers(0, cols, size=(actual_nnz, ), dtype=np.int64)
    tmp = sparse.csr_matrix((data, indices, indptr), shape=(rows, cols), dtype=dtype)
    tmp.sum_duplicates()
    tmp.sort_indices()
    return tmp


def generate_dense_matrix(rows: int, cols: int, dtype: np.dtype, rng: np.random.Generator) -> np.ndarray:
    """ Generates a dense matrix.

    :param rows: The number of rows in the matrix.
    :param cols: The number of columns in the matrix.
    :param dtype: The data type of the matrix.
    :param rng: The random number generator to use.
    :return: The generated dense matrix.
    """
    return 2*rng.random((rows, cols), dtype=dtype)-1
