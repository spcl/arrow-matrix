include("../GraphAlgorithms.jl")
using Test
using SparseArrays

@testset "DisjointSet Tests" begin
    
    @testset "Initialization" begin
        ds = DisjointSet(5)
        @test ds.parent == [1, 2, 3, 4, 5]
        @test ds.rank == zeros(Int, 5)
    end

    @testset "Single Element" begin
        ds = DisjointSet(1)
        @test find_root(ds, 1) == 1
    end

    @testset "Union of Distinct Sets" begin
        ds = DisjointSet(5)
        @test union!(ds, 1, 2)
        @test find_root(ds, 1) == find_root(ds, 2)
        @test union!(ds, 3, 4)
        @test find_root(ds, 3) == find_root(ds, 4)
        @test union!(ds, 2, 3)
        @test find_root(ds, 1) == find_root(ds, 3)
    end

    @testset "Union of Same Sets" begin
        ds = DisjointSet(5)
        union!(ds, 1, 2)
        @test !union!(ds, 1, 2)  # It should return false since they are already in the same set
    end

    @testset "Path Compression" begin
        ds = DisjointSet(5)
        union!(ds, 1, 2)
        union!(ds, 2, 3)
        find_root(ds, 3)  # After this operation, 3 should directly point to 1
        @test ds.parent[3] == 1
    end

    @testset "Union by Rank" begin
        ds = DisjointSet(5)
        union!(ds, 1, 2)
        # We assume that the union prefers to put the first argument into the root
        @test (ds.rank[1] == 1 && ds.rank[2] == 0)
        union!(ds, 2, 3)
        @test (ds.rank[1] == 1 && ds.rank[2] == 0 && ds.rank[3] == 0 )

        union!(ds, 4, 5)
        # Here, either 4 becomes the root and its rank becomes 1
        @test (ds.rank[4] == 1 && ds.rank[5] == 0)
        
        # Now, 1 becomes the root and increases its rank
        union!(ds, 1, 4)
        @test (ds.rank[1] == 2)
    end
end



using Test, SparseArrays
using SparseArrays  # For SparseMatrixCSC
using DataStructures  # For PriorityQueue
using Test  # For writing test cases

# Assuming the minimum_spanning_forest function is defined here or in an imported module

function known_graph_1()
    # A small graph with 4 vertices and 5 edges
    edges = [(1, 2, 1.0), (2, 3, 2.0), (3, 4, 3.0), (4, 1, 4.0), (1, 3, 0.5)]
    adj_matrix = zeros(4, 4)
    weights = Float32[]
    for (u, v, w) in edges
        adj_matrix[u, v] = w
        push!(weights, w)
    end
    return sparse(adj_matrix), weights
end

@testset "Minimum Spanning Forest Tests" begin
    # Test 1: Known small graph 1
    adj_matrix, weights = known_graph_1()
    mst_edges = minimum_spanning_forest(adj_matrix, weights, nothing, false)
    @test length(mst_edges[1]) == length(mst_edges[2]) == 3  # n - 1 edges for a tree

    # Test 2: Single vertex graph
    singleton = spzeros(1,1)
    weights = Float32[]
    mst_edges = minimum_spanning_forest(singleton, weights, nothing, false)
    @test isempty(mst_edges[1])  # No edges as it's a single vertex graph

    # Test 3: Empty graph
    adj_matrix = spzeros(5, 5)
    weights = Float32[]
    mst_edges = minimum_spanning_forest(adj_matrix, weights)
    @test isempty(mst_edges[1]) && isempty(mst_edges[2])  # No edges as it's an empty graph

    # Test 4: Fully connected graph
    adj_matrix = ones(Float32, 4, 4)
    adj_matrix = sparse(adj_matrix)
    weights = ones(6)
    mst_edges = minimum_spanning_forest(adj_matrix, weights)
    @test length(mst_edges[1]) == length(mst_edges[2]) == 3  # n - 1 edges for a tree

    # Additional tests can be added to test other aspects of the function
end
