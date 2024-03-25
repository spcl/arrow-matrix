# Arrow Matrix Decomposition - Fast SpMM for Tall-Skinny Matrices

We propose a novel approach to iterated sparse matrix dense matrix multiplication, a fundamental computational kernel in scientific computing and graph neural network training. In cases where matrix sizes exceed the memory of a single compute node, data transfer becomes a bottleneck. An approach based on dense matrix multiplication algorithms leads to sub-optimal scalability and fails to exploit the sparsity in the problem. To address these challenges, we propose decomposing the sparse matrix into a small number of highly structured matrices called arrow matrices, which are connected by permutations. Our approach enables communication-avoiding multiplications, achieving a polynomial reduction in communication volume per iteration for matrices corresponding to planar graphs and other minor-excluded families of graphs. Our evaluation demonstrates that our approach outperforms a state-of-the-art method for sparse matrix multiplication on matrices with hundreds of millions of rows, offering near-linear strong and weak scaling.

This project contains the code for the paper
[Arrow Matrix Decomposition: A Novel Approach for Communication-Efficient Sparse Matrix Multiplication, Gianinazzi et al., PPoPP 2024](https://dl.acm.org/doi/10.1145/3627535.3638496)

## Key Features

**Scalable and Distributed Computing**: With support for mpi4py and Cray-MPICH, the module is designed for scalability, facilitating distributed computing across multiple nodes and GPUs.

**Efficient SpMM Operations**: By integrating CSRMM kernels and leveraging GPU acceleration, our module offers highly efficient SpMM operations suitable for large-scale scientific computing tasks.

**Advanced Decomposition Techniques**: The use of linear arrangement frameworks and pruning, coupled with the innovative decomposition algorithm, ensures optimal performance and resource utilization in SpMM operations.

**Compatibility and Versatility**: The implementation's reliance on widely-used and well-supported libraries and frameworks ensures broad compatibility and application across various computing environments and use cases.

## Installation

The package can be installed using pip:
```
pip install -e .
```
To enable gpu support you additionally need to install [cupy](https://docs.cupy.dev/en/stable/install.html)

For example:
```commandline
pip install cupy-cuda11x
```
or:
```commandline
pip install cupy-cuda12x
```


To verify the installation, you can run the tests:
```
cd scripts
chmod +x run_tests.sh
./run_tests.sh
```

## Quick Start

Using the arrow matrix spmm requires two steps:

1) decompose the matrix
2) perform the spmm

We provide two implementations for 1., one in python and one in Julia.
The python implementation may be called from the `arrow_decompose` commandline call.

Example Usage (.mat input):
```commandline
arrow_decompose --dataset_dir ~/data --dataset_name graph1 graph2 --format 'matlab' --width 10000
```

Example Usage Matrix Market (.mtx) input:
```commandline
arrow_decompose --dataset_dir ~/data --dataset_name graph1 graph2 --format 'mtx' --width 10000
```
Options:
* For a directed graph, pass `--directed True`.
* To visualize the arrow matrices, pass `--visualize True`. 
* Pass `save_input_graph True` to save the input graph in order to speed up later invocations of the script.

The Julia implementation may be called from the `ArrowDecompositionMain.jl` script.
It is necessary to convert its output to the npy format using the `convert_to_csr.jl` scripy

To multiply 10 times with the decomposed matrix on random right-hand sides, you can use the `spmm_arrow` commandline call.
```commandline
mpiexec -n 8 spmm_arrow --path ./data/graph1_B --width 10000 --features 16 --device cpu --iterations 10
```

To use your custom right-hand sides, you need to 
use the `ArrowDecompositionMPI` class directly, as defined in `arrow_dec_mpi.py`.
To see how to use that class, refer to `arrow_bench.py`.

## Provided SpMM Implementations

### Arrow Matrix

The arrow matrix decomposition-based kernel can be invoked via the spmm_arrow entry point.
It requires that the arrow matrices have been decomposed and are available in the specified directory.

Example usage:
```commandline
mpiexec -n 8 ./scripts/spmm_arrow_main.py --path ./data/graph1_B --width 10000 --features 16 --device gpu --iterations 10
```

### 1.5D A-Stationary

The 1.5D A-Stationary-based kernel can be invoked via the spmm_15d entry point.

Example usage:
```commandline
mpiexec -n 8 ./scripts/spmm_15d_main.py --dataset file --file /path/to/matrix.mat --iterations 10 --device gpu
```

To run a benchmark with a random float32 sparse matrix having 100,000 vertices and 1,000,000 edges on a GPU:
```commandline
mpiexec -n 8 ./scripts/spmm_15d_main.py --vertices 100000 --edges 1000000 --device gpu
```

### Hypergraph-Partitioning-Based PeTSc-style

The hypergraph partitioning-based kernel can be invoked via the spmm_petsc entry point.

## Usage
```commandline
python ./scripts/spmm_petsc_main.py --type float64 --file matrix.part.1.slice.2.npz --gpu-tiling True
```
