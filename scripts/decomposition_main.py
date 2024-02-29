import argparse
import igraph
import numpy as np
import os
import pickle
import random
from arrow.decomposition import arrow_decomposition
from scipy import sparse
import scipy
import scipy.io
import mat73
from tqdm import tqdm


def load_graph_matlab(filename: str, undirected=False) -> igraph.Graph:
    """
    :param filename: filename of the matlab matrix file
    :param undirected: whether the graph is undirected
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

    return igraph.Graph.TupleList(edgelist, directed=not undirected)


def main() -> None:
    """
    Expects a dataset directory and a dataset name. The dataset is expected to be in the form of a matlab file.
    The script will convert the graph to a sparse matrix and save it in the dataset directory.
    The file structure should be:
    dataset_dir/dataset_name/dataset_name.mat
    """
    random.seed(0)
    np.random.seed(0)

    parser = argparse.ArgumentParser()

    parser.add_argument('--width', type=int, default=0)
    parser.add_argument('--dataset_dir', type=str, default='~/Desktop')
    # Allow one ore more dataset names:
    parser.add_argument('--dataset_name', nargs='+', type=str, default='kmer_V2a')

    args = parser.parse_args()
    print(args)
    datasets_directory = os.path.expanduser(args.dataset_dir)
    # If args.dataset_name is not a list, make dataset_names a list:
    if not isinstance(args.dataset_name, list):
        dataset_names = [args.dataset_name]
    else:
        dataset_names = args.dataset_name

    width = args.width

    block_diagonal = True

    for dataset_name in dataset_names:
        dataset_dir = os.path.join(datasets_directory, dataset_name)
        dataset_mat_file = dataset_name + '.mat'
        pickle_file = os.path.join(dataset_dir, f"{dataset_name}_graph.pickle")

        decomposition_dir = os.path.join(datasets_directory, dataset_name)

        if os.path.exists(pickle_file):
            print(f"Loading {dataset_name}'s graph...")
            graph = pickle.load(open(os.path.join(dataset_dir, f"{dataset_name}_graph.pickle"), "rb"))
        else:
            print(f"Loading {dataset_name}'s graph from matlab file...")
            graph = load_graph_matlab(os.path.join(dataset_dir, dataset_mat_file), undirected=False)
            pickle.dump(graph, open(os.path.join(dataset_dir, f"{dataset_name}_graph.pickle"), "wb"))

        if not os.path.exists(os.path.join(dataset_dir, f"{dataset_name}_A.npz")):
            # save the sparse matrix from the graph:
            print(f"Saving {dataset_name}'s graph as sparse matrix...")
            matrix = graph.get_adjacency_sparse().astype(np.float32)
            sparse.save_npz(os.path.join(dataset_dir, f"{dataset_name}_A.npz"), matrix)

        if not os.path.exists(os.path.join(dataset_dir, f"{dataset_name}_A_permuted.npz")):
            # Save a randomly permuted adjacency matrix:
            print(f"Saving {dataset_name}'s randomly permuted graph as sparse matrix...")
            permutation = list(np.random.permutation(graph.vcount()))
            g_permuted = graph.permute_vertices(permutation)
            matrix = g_permuted.get_adjacency_sparse().astype(np.float32)
            sparse.save_npz(os.path.join(dataset_dir, f"{dataset_name}_A_permuted.npz"), matrix)

        print(f"Converting {dataset_name} to arrow decomposition with width {width}...")

        if width > 0:
            B = arrow_decomposition(graph, arrow_width=width, max_number_of_levels=5, block_diagonal=block_diagonal)

            for i, arrow in tqdm(enumerate(B)):
                print(f"Converting the {i}-th arrow matrix to sparse matrix...")
                matrix = arrow.graph.get_adjacency_sparse().astype(np.float32)
                basename = f"{dataset_name}_B_{arrow.arrow_width}_{i}"
                if block_diagonal:
                    basename = f"{basename}_bd"
                sparse.save_npz(os.path.join(dataset_dir, f"{basename}.npz"), matrix)
                np.save(os.path.join(dataset_dir, f"{basename}_permutation.npy"), arrow.permutation)
        else:
            # Skip, do nothing:
            print(f"Skipping {dataset_name} because width is 0")


if __name__ == "__main__":
    main()
