import os

from arrow import decomposition
import igraph
import numpy as np
import unittest
from scipy import sparse
from typing import List
from arrow.common import graphio


class TestArrowDecomposition(unittest.TestCase):

    def generate_datasets(self):
        datasets_synthetic: List[igraph.Graph] = [igraph.Graph.Barabasi(2 ** i, 4, 503) for i in range(4, 8)]
        datasets_synthetic.extend([igraph.Graph.Barabasi(2 ** i, 8, 3434) for i in range(5, 8)])
        datasets_synthetic.extend([igraph.Graph.Erdos_Renyi(2 ** i, 0.1) for i in range(5, 8)])
        datasets_synthetic.extend([igraph.Graph.Barabasi(2 ** i, 3, directed=True) for i in range(10, 13)])

        width_counts = [4, 8, 10]

        return datasets_synthetic, width_counts

    def test_arrow(self):

        # Initialize random number generator
        rng = np.random.default_rng(42)

        datasets_synthetic, width_count = self.generate_datasets()

        for (g_index, g) in enumerate(datasets_synthetic):

            # Generate random dense matrix (B in A @ B = C)
            B = rng.random((g.vcount(), 16), dtype=np.float32)

            for width_c in width_count:
                g.vs["original_id"] = range(0, g.vcount())

                g_copy = g.copy()
                width = int(g.vcount() / width_c) + 1
                decomp = decomposition.arrow_decomposition(g, width, max_number_of_levels=100, block_diagonal=True)

                # The permutations are actually permutations
                for i in range(len(decomp)):
                    self.assertEqual(len(decomp[i][1]), g_copy.vcount())
                    vertex_set = set(decomp[i][1])
                    for v in range(g_copy.vcount()):
                        self.assertIn(v, vertex_set)

                inverse_permutation = [np.argsort(decomp[i][1]) for i in range(len(decomp))]
                un_permuted_graphs = [decomp[i][0].permute_vertices(list(decomp[i][1])) for i in range(len(decomp))]

                g_union = un_permuted_graphs[0]
                for i in range(1, len(decomp)):
                    g_union = g_union.union(un_permuted_graphs[i])

                # This tests that each edge occurs in both g AND g_union
                self.assertEqual(g_union.ecount(), g_copy.ecount())
                edge_set = set([e.tuple for e in g_copy.es])
                for e in g_union.es:
                    self.assertIn(e.tuple, edge_set)

                # They are edge-disjoint
                for i in range(len(decomp)):
                    for j in range(i + 1, len(decomp)):
                        g_intersection = un_permuted_graphs[i].intersection(un_permuted_graphs[j])
                        self.assertEqual(g_intersection.ecount(), 0)

                # The bandwidth is actually correct (based on edges)
                for i in range(len(decomp)):
                    width = decomp[i].arrow_width
                    band_edges = decomp[i][0].es.select(
                        lambda e: abs(e.source - e.target) <= width
                                  or e.source < width
                                  or e.target < width)

                    self.assertEqual(len(decomp[i][0].es), len(band_edges))

                # Verify matrix product
                A: sparse.csr_matrix = g.get_adjacency_sparse().astype(np.float32)
                ref_C = A @ B
                val_C = np.zeros((A.shape[0], B.shape[1]), dtype=np.float32)
                val_A = np.zeros(A.shape, dtype=np.float32)

                for i, arrow in enumerate(decomp):
                    matrix = arrow.graph.get_adjacency_sparse().astype(np.float32)
                    print("Arrow width", arrow.arrow_width)
                    print(matrix.toarray())

                    partial_sum = matrix @ B[arrow.permutation]
                    val_C += partial_sum[inverse_permutation[i]]

                    P = np.eye(A.shape[0])[inverse_permutation[i]]

                    val_A += P @ matrix @ P.T

                print(f"Relative error: {np.linalg.norm(ref_C - val_C) / np.linalg.norm(ref_C)}")

                self.assertTrue(np.allclose(val_A, A.toarray()))
                self.assertTrue(np.allclose(val_C, ref_C))

                # Create a tmp directory ./tmp
                os.makedirs("./tmp", exist_ok=True)

                # Save the decomposition to a file
                filename = f"./tmp/arrow_decomposition_{g_index}"
                graphio.save_decomposition(g, decomp, filename, block_diagonal=True)
                self.decomposition_matches(g.get_adjacency_sparse().astype(np.float32), filename, width, True)

        # Remove all files in the tmp directory
        for file in os.listdir("./tmp"):
            os.remove(os.path.join("./tmp", file))

    def decomposition_matches(self, A: sparse.csr_matrix, decomposition_filename: str, width:int, is_block_diagonal: bool):
        """
        Tests if a decomposition (saved to file) matches a given sparse matrix A.
        Note: modifies A in place. So A becomes unusable after running this test. Make a copy prior to calling
        if you need A later.
        :param A: The true matrix.
        :param decomposition_filename: path to the file (same arg as the one passed to save_decomposition)
        :param width: the width of the decomposition
        :param is_block_diagonal: true if the decomposition is block diagonal
        :return:
        """
        decomp = graphio.load_decomposition(decomposition_filename, width, is_block_diagonal)

        for i, (matrix, permutation) in enumerate(decomp):
            inverse_permutation = np.argsort(permutation)

            P_val = np.ones(inverse_permutation.shape[0])
            P_indptr = np.arange(inverse_permutation.shape[0]+1)

            P = sparse.csr_matrix((P_val, inverse_permutation, P_indptr), dtype=np.float32)
            A -= P @ matrix @ P.T

        print("DECOMPOSITION ERROR", np.max(np.abs(A)))
        self.assertTrue(np.isclose(0.0, np.max(np.abs(A))))

    @staticmethod
    def compute_spmm(decomposition: list, X: np.ndarray):
        """
        Computes the product of the decomposition with X
        :param decomposition:
        :param X:
        :return:
        """
        assert X is not None

        val_C = np.zeros((X.shape[0], X.shape[1]), dtype=np.float32)
        for i, (adjacency, permutation) in enumerate(decomposition):
            # print(graph.get_adjacency_sparse().astype(np.float32).toarray())
            inverse_permutation = np.argsort(permutation)
            partial_sum = adjacency @ X[permutation]
            val_C += partial_sum[inverse_permutation]

        return val_C

    def test_arrow_mm(self):

        # Initialize random number generator
        rng = np.random.default_rng(42)

        datasets_synthetic, width_count = self.generate_datasets()

        for g in datasets_synthetic:

            # Generate random dense matrix (B in A @ B = C)
            B = rng.random((g.vcount(), 16), dtype=np.float32)

            for width_c in width_count:
                g.vs["original_id"] = range(0, g.vcount())

                width = int(g.vcount() / width_c) + 1
                decomp = decomposition.arrow_decomposition(g, width, width)

                # Verify matrix product
                A: sparse.csr_matrix = g.get_adjacency_sparse().astype(np.float32)
                ref_C = A @ B
                val_C = np.zeros((A.shape[0], B.shape[1]), dtype=np.float32)
                val_A = np.zeros(A.shape, dtype=np.float32)

                for i, arrow in enumerate(decomp):

                    inverse_permutation = np.argsort(arrow.permutation)
                    permutation = arrow.permutation

                    matrix = arrow.graph.get_adjacency_sparse().astype(np.float32)
                    partial_sum = matrix @ B[permutation]
                    val_C += partial_sum[inverse_permutation]

                    P = np.eye(A.shape[0])[inverse_permutation]

                    val_A += P @ matrix @ P.T

                print(f"Relative error: {np.linalg.norm(ref_C - val_C) / np.linalg.norm(ref_C)}")


if __name__ == '__main__':
    unittest.main()
