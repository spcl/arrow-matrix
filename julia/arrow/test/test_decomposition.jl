include("../ArrowDecomposition.jl")
using Test
using SparseArrays
using Random

@testset "Undirected Tests" begin
    
    @testset "random_undirected" begin
        Random.seed!(1234)  # Fix the seed for reproducibility
        for density in [0.1, 0.2, 0.3, 0.4, 0.5]
            matrix = sprand(20, 20, density)
            matrix = matrix + matrix'
            num_vertices = size(matrix, 1)
            original_id = Vector{Int}(1:num_vertices)
            order::Vector{Int} = Vector{Int}()
            linearize_with_random_forest(matrix, order, original_id, 4, nothing)
        end
    end

    @testset "clique" begin
        Random.seed!(1234)  # Fix the seed for reproducibility
        matrix = sprand(7, 7, 1.0)
        num_vertices = size(matrix, 1)
        original_id = Vector{Int}(1:num_vertices)
        order::Vector{Int} = Vector{Int}()
        linearize_with_random_forest(matrix, order, original_id, 4, nothing)
    end

    @testset "undirected" begin
        # a matrix using sparse
        symmetric_matrix::SparseMatrixCSC = sparse([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [2, 3, 4, 5, 6, 7, 8, 9, 10, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        symmetric_matrix = symmetric_matrix + symmetric_matrix'
        num_vertices = size(symmetric_matrix, 1)
        original_id = Vector{Int}(1:num_vertices)
        order::Vector{Int} = Vector{Int}()
        linearize_with_random_forest(symmetric_matrix, order, original_id, 4, nothing)
    end
end

# Test arrow decomposition
@testset "End to End Test" begin
    
    @testset "Random Directed" begin
        # Generate  a random matrix
        for density in [0.01, 0.002, 0.005, 0.1, 0.15, 0.2]
            Random.seed!(1234)  # Fix the seed for reproducibility
            matrix = sprand(1000, 1000, density)
            arrow_decomposition(matrix, 100, 10, true, true, false)
        end
    end

    @testset "Random Un-Directed" begin
        # Generate  a random matrix
        for density in [0.1, 0.2, 0.3, 0.4, 0.5]
            Random.seed!(1234)  # Fix the seed for reproducibility
            matrix = sprand(100, 100, density)
            matrix = matrix + matrix'
            arrow_decomposition(matrix, 10, 10, true, true, true)
        end
    end

end