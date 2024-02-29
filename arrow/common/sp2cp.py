from scipy import sparse

try:
    import cupy as cp

    def _sp2cp(matrix: sparse.csr_matrix) -> cp.sparse.csr_matrix:
        """ Converts a SciPy CSR matrix to a CuPy CSR matrix.

        :param matrix: The SciPy CSR matrix.
        :return: The CuPy CSR matrix.
        """
        tmp = cp.sparse.csr_matrix((cp.asarray(matrix.data), cp.asarray(matrix.indices), cp.asarray(matrix.indptr)),
                                   shape=matrix.shape,
                                   dtype=matrix.dtype)
        tmp._has_canonical_format = True
        return tmp

except Exception as e:
    print(e)
    def _sp2cp(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
        """ Converts a SciPy CSR matrix to a CuPy CSR matrix.

        :param matrix: The SciPy CSR matrix.
        :return: The CuPy CSR matrix.
        """
        # raise a new exception to stop the execution (not supported)
        raise Exception("CuPy not available. Check your installation")
