import enum
import os

import igraph
import numpy as np
import pickle

from tqdm import tqdm

import arrow.decomposition
from numpy import typing as npt
from scipy import sparse
from typing import List, Union, Any


def load_graph(filename: str) -> igraph.Graph:
    """
    :param filename: 
    :return: 
    """
    return pickle.load(open(f"{filename}_graph.pickle", "rb"))


# Decomposition File Type enum:
#   - npz: compressed sparse row matrix
#   - indptr npy: numpy array
#   - indices npy: numpy array
#   - data npy: numpy array
class DecompositionFileType(enum.Enum):
    npz = 1
    indptr_npy = 2
    indices_npy = 3
    data_npy = 4
    permutation_npy = 5
    nonzero_rows_npy = 6


def format_path(base_path: str, width: int, index: Union[int, None], block_diagonal: bool,
                file_type: DecompositionFileType) -> str:
    """
    Formats a path for a matrix with the given parameters
    :param base_path: the base path
    :param width: the width of the matrix
    :param index: the index of the matrix
    :param block_diagonal: whether the matrix is in block diagonal format
    :param file_type: the type of the file
    :return: the formatted path
    """
    path = f"{base_path}_B"
    path += f"_{width}"
    if index is not None:
        path += f"_{index}"
    if block_diagonal:
        path += "_bd"

    # Append extension based on file_type
    if file_type == DecompositionFileType.npz:
        path += ".npz"
    elif file_type == DecompositionFileType.indptr_npy:
        path += "_indptr.npy"
    elif file_type == DecompositionFileType.indices_npy:
        path += "_indices.npy"
    elif file_type == DecompositionFileType.data_npy:
        path += "_data.npy"
    elif file_type == DecompositionFileType.permutation_npy:
        path += "_permutation.npy"
    elif file_type == DecompositionFileType.nonzero_rows_npy:
        path += "_nnzrows.npy"

    return path


def save_decomposition(graph: igraph.Graph,
                       decomposition: list[arrow.decomposition.ArrowGraph],
                       filename: str,
                       dtype: npt.DTypeLike = np.float32,
                       use_width: bool = True,
                       block_diagonal: bool = True,
                       save_graph: bool = True) -> None:
    """
    Saves the decomposition to files in scipy csr format
    The i-th part of the decomposition is stored as {filename}_B_{width}_{bd}_{i}.npz
    The permutation that maps the original id's to the id's of B_i is in {filename}_B_{width}_{bd}_{i}_permutation.py
    as a numpy array.
    :param graph: the graph to store (if save_graph is True)
    :param decomposition: the decomposition to store
    :param filename: prefix to use for storing the files
    :param dtype: the data type to use for the sparse matrices
    :param use_width: ignored. exists for backwards compatibility.
    :param block_diagonal: whether the decomposition uses the block diagonal in the filename
    :return: None
    """

    if save_graph:
        # Save graph
        with open(f"{filename}_graph.pickle", "wb") as f:
            pickle.dump(graph, f)

        # Save A
        A = graph.get_adjacency_sparse().astype(dtype)
        sparse.save_npz(f"{filename}_A.npz", A)

    # Save B
    width = 0
    for i, arrow in enumerate(decomposition):
        A = arrow.graph.get_adjacency_sparse().astype(dtype)
        width = arrow.arrow_width
        matrix_path = format_path(filename, arrow.arrow_width, i, block_diagonal, DecompositionFileType.npz)
        sparse.save_npz(matrix_path, A)
        permutation_path = format_path(filename, arrow.arrow_width, i, block_diagonal,
                                       DecompositionFileType.permutation_npy)
        np.save(permutation_path, arrow.permutation)

    # Save nonzeros (for convenience)
    nonzero_rows = np.asarray([a.nonzero_rows for a in decomposition], dtype=np.int64)
    nnz_path = format_path(filename, width, 0, block_diagonal, DecompositionFileType.nonzero_rows_npy)
    np.save(nnz_path, nonzero_rows)


def decomposition_size(filename, width, block_diagonal):
    i = 0
    while True:
        path = format_path(filename, width, i, block_diagonal, DecompositionFileType.permutation_npy)
        # Check if path exists:
        if not os.path.exists(path):
            break
        i += 1
    return i


def save_decomposition_new(graph: igraph.Graph,
                           decomposition: list[arrow.decomposition.ArrowGraph],
                           filename: str,
                           dtype: npt.DTypeLike = np.float32,
                           use_width: bool = True,
                           block_diagonal: bool = True,
                           save_graph: bool = True) -> None:
    """
    Saves the decomposition to files in scipy csr format
    each part of the decomposition is stored
    in three separate arrays: *_indptr.npy, *_indices.npy, *_data.npy

    The i-th part of the decomposition is stored as three arrays:
    {filename}_B_{width}_{i}_{bd}_indptr.npy,
    {filename}_B_{width}_{i}_{bd}_indices.npy,
     {filename}_B_{width}_{i}_{bd}_data.npy
    The permutation that maps the original id's to the id's of B_i is in
    {filename}_B_{width}_{i}_{bd}_permutation.py

    :param decomposition: the decomposition to store
    :param filename: prefix to use for storing the files
    :param dtype: the data type to use for the sparse matrices
    :param use_width: ignored. exists for backwards compatibility.
    :param block_diagonal: whether the decomposition uses the block diagonal in the filename
    :param save_graph: whether to save the input graph to a pickled file
    :return: None
    """

    if save_graph:
        # Save graph
        with open(f"{filename}_graph.pickle", "wb") as f:
            pickle.dump(graph, f)

        # Save A
        A = graph.get_adjacency_sparse().astype(dtype)
        # sparse.save_npz(f"{filename}_A.npz", A)
        np.save(f"{filename}_A_indptr.npy", A.indptr)
        np.save(f"{filename}_A_indices.npy", A.indices)
        np.save(f"{filename}_A_data.npy", A.data)

    # Save B
    arrow_width = 0
    for i, arrow_m in enumerate(decomposition):
        A = arrow_m.graph.get_adjacency_sparse().astype(dtype)

        path = format_path(filename, arrow_m.arrow_width, i, block_diagonal, DecompositionFileType.indptr_npy)
        np.save(path, A.indptr)
        path = format_path(filename, arrow_m.arrow_width, i, block_diagonal, DecompositionFileType.indices_npy)
        np.save(path, A.indices)
        path = format_path(filename, arrow_m.arrow_width, i, block_diagonal, DecompositionFileType.data_npy)
        np.save(path, A.data)
        path = format_path(filename, arrow_m.arrow_width, i, block_diagonal, DecompositionFileType.permutation_npy)
        np.save(path, arrow_m.permutation)

        if i == 0:
            arrow_width = arrow_m.arrow_width

    # Save nonzeros (for convenience)
    nonzero_rows = np.asarray([a.nonzero_rows for a in decomposition], dtype=np.int64)
    path = format_path(filename, arrow_width, 0, block_diagonal, DecompositionFileType.nonzero_rows_npy)
    np.save(path, nonzero_rows)


def load_decomposition(filename: str, width: int = None, block_diagonal: bool = True, no_permutation=False) \
        -> list[(sparse.csr_matrix, Union[None, npt.NDArray[np.integer]])]:
    """
    Loads the decomposition from files in scipy csr format
    The i-th part of the decomposition is stored as {filename}_B_{width}_{bd}_{i}.npz
    The permutation that maps the original id's to the id's of B_i is in {filename}_B_{width}_{bd}_{i}_permutation.py
    as a numpy array.
    :param no_permutation: If true, the permutation matrix is not loaded (None is at its place)
    :param filename: prefix to use for loading the files
    :param width: The width of the arrow to load. If None, it is assume that it is not part of the filename.
    :param block_diagonal: whether the decomposition uses the block diagonal format
    :return: the decomposition
    """

    # Load B
    i = 0
    decomposition = []

    print("Loading decomposition", filename, width, f"bd: {block_diagonal}...")

    for i in range(decomposition_size(filename, width, block_diagonal)):
        path = format_path(filename, width, i, block_diagonal, DecompositionFileType.npz)
        B = sparse.load_npz(path)
        if no_permutation:
            permutation = None
        else:
            path = format_path(filename, width, i, block_diagonal, DecompositionFileType.permutation_npy)
            permutation = np.load(path)
        decomposition.append((B, permutation))

    if len(decomposition) == 0:
        # THIS IS THE OLD NAMING SCHEME.
        # TO SUPPORT THE OLD NAMING SCHEME, WE SEARCH FOR IT ALSO IF THE PREVIOUS BREAKS
        while True:
            try:
                # mawi_201512020130_B_5000000_0_bd
                # mawi_201512020130_B_5000000_0_bd_permutation.npy
                basename = f"{filename}_B"
                if width:
                    basename += f"_{width}"
                basename += f"_{i}"
                if block_diagonal:
                    basename += "_bd"
                B = sparse.load_npz(f"{basename}.npz")
                print("matrix found:", B.nnz)
                if no_permutation:
                    permutation = None
                else:
                    permutation = np.load(f"{basename}_permutation.npy")
            except FileNotFoundError:
                break
            decomposition.append((B, permutation))
            i += 1

    return decomposition


def load_decomposition_new(filename: str, width: int = None, block_diagonal: bool = True, no_permutation=False,
                           mem_map=False) \
        -> list[(Any, Union[None, npt.NDArray[np.integer]])]:
    """
    Loads the decomposition from files in csr format
    Can either construct the scipy csr matrix or use memory mapping to load it.
    The i-th part of the decomposition is stored as three arrays:
    {filename}_B_{width}_{i}_{bd}_indptr.npy,
    {filename}_B_{width}_{i}_{bd}_indices.npy,
     {filename}_B_{width}_{i}_{bd}_data.npy
    The permutation that maps the original id's to the id's of B_i is in
    {filename}_B_{width}_{i}_{bd}_permutation.py
    as a numpy array.
    :param no_permutation: If true, the permutation matrix is not loaded (None is at its place)
    :param filename: prefix to use for loading the files
    :param width: The width of the arrow to load. If None, it is assume that it is not part of the filename.
    :param block_diagonal: whether the decomposition uses the block diagonal format
    :param mem_map: whether to use memory mapping for the files
    :return: the decomposition
    """

    # Load B
    i = 0
    decomposition = []

    basename = get_pathname(filename, width, block_diagonal)
    print("Loading decomposition", basename, "...", flush=True)
    while True:

        try:
            f_name = format_path(filename, width, i, block_diagonal, DecompositionFileType.indptr_npy)
            if mem_map:
                indptr = np.lib.format.open_memmap(f_name, mode='r')
            else:
                indptr = np.load(f_name)
            f_name = format_path(filename, width, i, block_diagonal, DecompositionFileType.indices_npy)
            if mem_map:
                indices = np.lib.format.open_memmap(f_name, mode='r')
            else:
                indices = np.load(f_name)
            f_name = format_path(filename, width, i, block_diagonal, DecompositionFileType.data_npy)
            if os.path.exists(f_name):
                if mem_map:
                    data = np.lib.format.open_memmap(f_name, mode='r')
                else:
                    data = np.load(f_name)
            else:
                data = np.ones(indices.size, dtype=np.float32)
            if mem_map:
                B = (data, indices, indptr)
            else:
                B = sparse.csr_matrix((data, indices, indptr))

            if no_permutation:
                permutation = None
            else:
                f_name = format_path(filename, width, i, block_diagonal, DecompositionFileType.permutation_npy)
                permutation = np.load(f_name)
        except FileNotFoundError:
            break
        decomposition.append((B, permutation))
        i += 1

    return decomposition


def convert_decomposition(filename: str, width: int = None, block_diagonal: bool = True) \
        -> list[(sparse.csr_matrix, Union[None, npt.NDArray[np.integer]])]:
    # Load B
    i = 0
    decomposition = []

    basename = get_pathname(filename, width, block_diagonal)
    print("Loading decomposition", basename, "...")
    while True:

        try:
            B = sparse.load_npz(f"{basename}_{i}.npz")
            # Instead use the new format_path function
            np.save(format_path(filename, width, i, block_diagonal, DecompositionFileType.indptr_npy), B.indptr)
            np.save(format_path(filename, width, i, block_diagonal, DecompositionFileType.indices_npy), B.indices)
            np.save(format_path(filename, width, i, block_diagonal, DecompositionFileType.data_npy), B.data)

        except FileNotFoundError:
            break
        decomposition.append((B, None))
        i += 1

    if len(decomposition) == 0:
        # THIS IS THE OTHER NAMING SCHEME.
        # TO SUPPORT THE OLD NAMING SCHEME, WE SEARCH FOR IT ALSO IF THE PREVIOUS BREAKS
        while True:
            try:
                # mawi_201512020130_B_5000000_0_bd
                # mawi_201512020130_B_5000000_0_bd_permutation.npy
                f_name = format_path(filename, width, i, block_diagonal, DecompositionFileType.npz)
                B = sparse.load_npz(f_name)
                # Instead use the new format_path function
                np.save(format_path(filename, width, i, block_diagonal, DecompositionFileType.indptr_npy), B.indptr)
                np.save(format_path(filename, width, i, block_diagonal, DecompositionFileType.indices_npy), B.indices)
                np.save(format_path(filename, width, i, block_diagonal, DecompositionFileType.data_npy), B.data)

            except FileNotFoundError:
                break
            decomposition.append((B, None))
            i += 1

    return decomposition


def split_matrix_to_blocks(A: sparse.csr_matrix,
                           block_size: int,
                           dtype: npt.DTypeLike = None,
                           use_min_shape: bool = False) -> List[List[Union[sparse.csr_matrix, None]]]:
    """
    Splits the matrix A into blocks of size block_size x block_size
    :param A: the matrix to split
    :param block_size: the size of the blocks
    :param dtype: the data type to use for the blocks. If None, the data type of A is used.
    :param use_min_shape: whether to use the minimum shape of the blocks or keep it fixed at block_size
    :return: a list of the blocks
    """
    rows, cols = A.shape
    dtype = dtype or A.dtype

    # Generate blocks
    blocks_per_col = int(np.ceil(rows / block_size))
    blocks_per_row = int(np.ceil(cols / block_size))
    blocks = [[None for _ in range(blocks_per_row)] for _ in range(blocks_per_col)]
    for i in range(blocks_per_col):
        for j in range(blocks_per_row):
            if i > 0 and not j in (0, i - 1, i, i + 1):
                continue
            #

            shape = (min(rows - i * block_size, block_size), min(cols - j * block_size, block_size))
            slice = A[i * block_size:min(rows, (i + 1) * block_size),
                    j * block_size:min(cols, (j + 1) * block_size)]
            pad_width = block_size - shape[0]

            if use_min_shape or pad_width == 0:
                block = sparse.csr_matrix(slice, shape=shape, dtype=dtype)
            else:
                # We need to pad the index pointer so that there are enough rows
                shape2 = (block_size, block_size)
                indx_ptr = np.pad(slice.indptr, (0, pad_width), mode='edge')
                block = sparse.csr_matrix((slice.data, slice.indices, indx_ptr),
                                          shape=shape2,
                                          dtype=dtype)

            block.sum_duplicates()
            block.sort_indices()
            assert block.has_canonical_format
            blocks[i][j] = block

    return blocks


def split_matrix_to_blocks_new(A: sparse.csr_matrix,
                               block_size: int,
                               dtype: npt.DTypeLike = None,
                               use_min_shape: bool = False) -> List[List[Union[sparse.csr_matrix, None]]]:
    """
    Splits the matrix A into blocks of size block_size x block_size

    :param A: the matrix to split
    :param block_size: the size of the blocks
    :param dtype: the data type to use for the blocks. If None, the data type of A is used.
    :param use_min_shape: whether to use the minimum shape of the blocks or keep it fixed at block_size
    :return: a list of the blocks
    """
    # A[2]: indptr
    # A[1]: indices
    # A[0]: data
    # NOTE: Assuming matrix is an adjacency matrix, i.e., square
    # Assert square
    assert A.shape[0] == A.shape[1]

    rows, cols = A[2].size - 1, A[2].size - 1
    dtype = dtype or A[0].dtype

    # Generate blocks
    blocks_per_col = int(np.ceil(rows / block_size))
    blocks_per_row = int(np.ceil(cols / block_size))
    blocks = [[None for _ in range(blocks_per_row)] for _ in range(blocks_per_col)]
    for i in range(blocks_per_col):
        for j in range(blocks_per_row):
            if i > 0 and not j in (0, i - 1, i, i + 1):
                continue

            bslice = (slice(i * block_size, min(rows, (i + 1) * block_size)),
                      slice(j * block_size, min(cols, (j + 1) * block_size)))

            blocks[i][j] = bslice

    return blocks


def load_block_from_bslice(A: tuple,
                           bslice: tuple,
                           block_size: int,
                           dtype: npt.DTypeLike = None,
                           use_min_shape: bool = False) -> sparse.csr_matrix:
    if bslice is None:
        return None

    # Slicing
    row_slice = bslice[0]
    col_slice = bslice[1]
    num_rows = row_slice.stop - row_slice.start
    num_cols = col_slice.stop - col_slice.start
    shape = (num_rows, num_cols)

    # Load data
    indptr = np.empty(num_rows + 1, dtype=A[2].dtype)
    indptr[:-1] = A[2][row_slice]
    indptr[-1] = A[2][row_slice.stop]
    indices = A[1][indptr[0]:indptr[-1]]
    data = A[0][indptr[0]:indptr[-1]]
    # Fix indptr
    indptr -= indptr[0]
    indptr[-1] = data.size
    # Load full rows
    row_block = sparse.csr_matrix((data, indices, indptr), shape=(num_rows, A[2].size - 1), dtype=A[0].dtype)
    # Slice columns
    block = row_block[:, col_slice]

    # Padding and others
    pad_width = block_size - shape[0]

    if use_min_shape or pad_width == 0:
        block = sparse.csr_matrix(block, shape=shape, dtype=dtype)
    else:
        # We need to pad the index pointer so that there are enough rows
        shape2 = (block_size, block_size)
        indx_ptr = np.pad(block.indptr, (0, pad_width), mode='edge')
        block = sparse.csr_matrix((block.data, block.indices, indx_ptr),
                                  shape=shape2,
                                  dtype=dtype)

    block.sum_duplicates()
    block.sort_indices()
    assert block.has_canonical_format

    return block


def get_pathname(basename: str, width: int, is_block_diagonal: bool):
    basename = f"{basename}_B"
    if width:
        basename += f"_{width}"
    if is_block_diagonal:
        basename += "_bd"
    return basename
