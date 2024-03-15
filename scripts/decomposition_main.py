import argparse
from pathlib import Path

import igraph
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random

from arrow.common import graphio
from arrow.decomposition import arrow_decomposition
import scipy
import scipy.io
from scipy.io import mmread
import mat73


def load_graph_matlab(filename, directed=False) -> igraph.Graph:
    """
    :param filename: filename of the matlab matrix file
    :param directed: whether the graph is undirected
    :return: the graph as an igraph object
    """
    try:
        dataset = mat73.loadmat(filename)
        mat = dataset['Problem']['A']
        mat = mat.tocoo()
        edgelist = np.stack((mat.row, mat.col), axis=1)
    except TypeError:
        dataset = scipy.io.loadmat(filename)
        # Extract the edgelist from the matlab matrix file
        edgelist = dataset['Problem'][0][0][1]

    return igraph.Graph.TupleList(edgelist, directed=directed)


def load_graph_matrix_market(file, directed=False) -> igraph.Graph:
    """
    Loads a graph from a matrix market file.

    For example, the following is a matrix market file:

    %%MatrixMarket matrix coordinate real general
    %%RBCode matrix
    %%RBMatrixID EXAMPLE1
    %%RBTitle Small general matrix used as Example 1
    5  5  11
    1  1   1.0
    3  1   2.0
    5  1   3.0
    1  2  -4.0
    4  2   5.0
    2  3  -6.0
    5  3  -7.0
    1  4  -8.0
    4  4  -9.0
    2  5  10.0
    5  5  11.0

    :param file: file of the matrix market file (Path or string)
    :param directed: whether the graph is undirected
    :return: the graph as an igraph object
    """

    # Load the matrix market file into a sparse matrix
    matrix = mmread(file)

    # Convert the sparse matrix to COO format, which makes it easier to iterate over non-zero elements
    coo_matrix = matrix.tocoo()

    # Prepare edge list and weights
    edges = [(i, j) for i, j in zip(coo_matrix.row, coo_matrix.col)]
    weights = coo_matrix.data

    # Create an igraph graph from the edges
    g = igraph.Graph(edges=edges, directed=directed)
    g.es['weight'] = weights
    print("Number of vertices:", g.vcount())
    print("Number of edges:", g.ecount())
    return g


def visualize_banded_decomposition(file, banded_decomposition: list) -> None:

    fig, axs = plt.subplots(1, len(banded_decomposition), constrained_layout=True)
    fig.set_figheight(5)
    fig.set_figwidth(5*len(banded_decomposition))

    i = 0
    for (B_i, permutation) in banded_decomposition:

        B_i: scipy.sparse.csr_matrix

        nzz_rows = np.diff(B_i.indptr)
        # largest nonzero row index:
        nzz_row_count = np.argmax(np.where(nzz_rows > 0)) + 1
        print(f"Number of non-zero rows in B_{i}: {nzz_row_count}")
        # Remove the zero rows to ensure square shape:

        B_i = B_i[0:max(B_i.shape[1], nzz_row_count), :]

        axs[i].spy(B_i, aspect='auto')
        axs[i].set_title("B_" + str(i))
        i = i+1

    plt.savefig(file)


def main() -> None:
    """
    Expects a dataset directory and a dataset name. The dataset is expected to be in the form of a matlab file.
    The script will convert the graph to a sparse matrix and save it in the dataset directory.
    The file structure should be (for --format=matlab):
    dataset_dir/dataset_name/dataset_name.mat
    or for matrix market --format=mtx:
    dataset_dir/dataset_name/dataset_name.mtx
    """
    random.seed(0)
    np.random.seed(0)

    parser = argparse.ArgumentParser()

    parser.add_argument('--width', type=int, default=5000000)
    parser.add_argument('--dataset_dir', type=str, default='~/Desktop/')
    # Allow one or more dataset names:
    parser.add_argument('--dataset_name', nargs='+', type=str, default=['kmer_V2a'])

    parser.add_argument('--format', type=str, default='matlab', help="The format of the graph file (matlab or mtx).")

    parser.add_argument('--visualize', type=bool, default=False, help="Visualize the decomposition.")

    parser.add_argument('--directed', type=bool, default=False, help="Whether the graph is directed or not.")

    parser.add_argument('--save_input_graph',
                        type=bool,
                        default=False,
                        help='Whether to save the graph as a pickle to speed up loading the matrix in the future.')

    args = parser.parse_args()
    print(args)
    datasets_directory = Path(args.dataset_dir).expanduser()
    # If args.dataset_name is not a list, make dataset_names a list:
    if not isinstance(args.dataset_name, list):
        dataset_names = [args.dataset_name]
    else:
        dataset_names = args.dataset_name

    width = args.width

    if width <= 0:
        raise ValueError("Width must be positive")

    block_diagonal = True

    for dataset_name in dataset_names:
        dataset_dir = datasets_directory / dataset_name
        pickle_file = dataset_dir / f"{dataset_name}_graph.pickle"

        if pickle_file.exists():
            print(f"Loading {dataset_name}'s graph...")
            with open(dataset_dir / f"{dataset_name}_graph.pickle", "rb") as pickle_file:
                graph = pickle.load(pickle_file)
        else:
            if args.format == 'matlab':
                dataset_mat_file = dataset_name + '.mat'
                if not (dataset_dir / dataset_mat_file).exists():
                    raise ValueError(f"File {dataset_mat_file} does not exist in {dataset_dir}")

                print(f"Loading {dataset_name}'s graph from matlab file...")
                graph = load_graph_matlab(dataset_dir / dataset_mat_file, directed=args.directed)
            elif args.format == 'mtx':
                dataset_mm_file = dataset_name + '.mtx'
                if not (dataset_dir / dataset_mm_file).exists():
                    raise ValueError(f"File {dataset_mm_file} does not exist in {dataset_dir}")

                print(f"Loading {dataset_name}'s graph from matrix market file...")
                graph = load_graph_matrix_market(dataset_dir / dataset_mm_file, directed=args.directed)
            else:
                raise ValueError(f"Unknown format {args.format}")

        print(f"Converting {dataset_name} to arrow decomposition with width {width}...")

        if width > 0:
            B = arrow_decomposition(graph, arrow_width=width, max_number_of_levels=10, block_diagonal=block_diagonal)
            print(f"Successfully decomposed into {len(B)} matrices.")

            print(f"Saving {dataset_name}'s arrow decomposition...")
            graphio.save_decomposition_new(graph,
                                           B,
                                           dataset_dir / dataset_name,
                                           block_diagonal=block_diagonal,
                                           save_graph=args.save_input_graph)

            if args.visualize:
                print(f"Visualizing {dataset_name}'s arrow decomposition...")

                decomposition = graphio.load_decomposition_new(dataset_dir / dataset_name,
                                                                width,
                                                                block_diagonal,
                                                                mem_map=False)

                visualize_banded_decomposition(dataset_dir / f"{dataset_name}_B_{width}_decomposition.png", decomposition)

                assert len(B) == len(decomposition)

        else:
            # Skip, do nothing:
            print(f"Skipping {dataset_name} because width is 0")


if __name__ == "__main__":
    main()
