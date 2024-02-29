using SparseArrays
using Base.Iterators: flatten
using MAT
using NPZ
using DataStructures  # for PriorityQueue
using ArgParse

include("GraphAlgorithms.jl")

function linearize_with_ck(g_adj::SparseMatrixCSC, order::Vector{Int}, original_id::Vector{Int}, base_size::Int=2, vert_mask::Union{BitVector, Nothing}=nothing)
    num_vertices = size(g_adj, 1)
    visited = falses(num_vertices)
    components_array, offsets = connected_components(g_adj, visited, vert_mask)
    components_size = offsets[2:end] .- offsets[1:end-1]
    components_ind = sortperm(components_size, rev=true)
    
    for ind in components_ind
        components = components_array[offsets[ind]:offsets[ind+1]-1]
        if components_size[ind] <= base_size
            append!(order, original_id[components])
            continue
        end
        
        if components_size[ind] > 10000
            visited .= false
        else
            visited[components] .= false
        end
        visited[components[1]] = true
        bfs_order = bfs(components[1], g_adj, visited, vert_mask)
        
        append!(order, original_id[bfs_order])
    end
end

function linearize_with_random_forest(g_adj::SparseMatrixCSC, order::Vector{Int}, original_id::Vector{Int}, base_size::Int=4, vert_mask::Union{BitVector, Nothing}=nothing)
    num_vertices = size(g_adj, 1)
    num_vertices_masked = isnothing(vert_mask) ? num_vertices : sum(vert_mask)
    num_edges = nnz(g_adj)
    factor = 2

    # TODO: merge MST, CC, BFS to one pass, DP as another pass
    weights = rand(Float32, num_edges)
    spanning_forest_edges = minimum_spanning_forest(g_adj, weights, vert_mask)
    spanning_forest = sparse(spanning_forest_edges[1], spanning_forest_edges[2], ones(length(spanning_forest_edges[1])), num_vertices, num_vertices)

    visited = falses(num_vertices)
    g_adj_symmetric = g_adj

    components_array, offsets = connected_components(g_adj_symmetric, visited, vert_mask)
    components_size = offsets[2:end] .- offsets[1:end-1]
    components_ind = sortperm(components_size, rev=true)

    @assert (length(components_size) <= num_vertices_masked) "Number of components $(length(components_size)) should be at most the number of vertices $(num_vertices_masked)"

    @assert (nnz(spanning_forest) == (num_vertices_masked - length(components_size))*factor)  "Spanning forest has $(nnz(spanning_forest)) edges, but should have $(num_vertices_masked - length(components_size)*factor) edges"

    subtree_size = ones(Int, num_vertices)
    
    for ind in components_ind
        components = components_array[offsets[ind]:offsets[ind+1]-1]
        if components_size[ind] <= base_size
            append!(order, original_id[components])
            continue
        end
        
        if components_size[ind] > 10000
            visited .= false
        else
            visited[components] .= false
        end
        visited[components[1]] = true
        return_levels = true
        bfs_order, levels = bfs(components[1], spanning_forest, visited, vert_mask, return_levels)
        level_dict = Dict{Int, Int}(bfs_order[i] => levels[i] for i in eachindex(bfs_order))
        
        # linearize_tree
        for v in reverse(bfs_order)
            for idx in nzrange(spanning_forest, v)
                neighbor = rowvals(spanning_forest)[idx]
                if level_dict[v] < level_dict[neighbor]
                    subtree_size[v] += subtree_size[neighbor]
                end
            end
        end
        @assert subtree_size[components[1]] == components_size[ind]

        # linearize_tree_stack
        stack = [components[1]]
        while length(stack) > 0
            current_vertex = pop!(stack)
            push!(order, original_id[current_vertex])

            # We want to visit the larger subtree vertices last, hence we push onto the stack into reverse order
            succ = [neighbor for neighbor in rowvals(spanning_forest)[nzrange(spanning_forest, current_vertex)] if level_dict[current_vertex] < level_dict[neighbor]]
            sortind = sortperm(subtree_size[succ], rev=true)
            append!(stack, succ[sortind])
        end
    end
end

function _arrow_linear_order(g_adj::SparseMatrixCSC, original_id::Vector{Int}, arrow_width::Int=512, deterministic::Bool=false, symmetric::Bool=true)
    num_vertices = size(g_adj, 1)
    costs = g_adj.colptr[2:end] .- g_adj.colptr[1:end-1]
    costs_ind = sortperm(costs, rev=true)
    base_size = 2

    high_cost_vertices = original_id[costs_ind[1:arrow_width]]
    middle_vertices = [costs_ind[i] for i in arrow_width+1:num_vertices if costs[costs_ind[i]] > 0]
    # TODO: optimize this
    singletons = [original_id[costs_ind[i]] for i in arrow_width+length(middle_vertices)+1:num_vertices if costs[costs_ind[i]] == 0]

    @assert length(high_cost_vertices) + length(middle_vertices) + length(singletons) == num_vertices

    order = high_cost_vertices
    vert_mask = falses(num_vertices)
    vert_mask[middle_vertices] .= true
    
    # If not symmetric, make a symmetrized version of g_adj
    if !symmetric
        g_adj_s = g_adj + g_adj'
    else
        g_adj_s = g_adj
    end

    if deterministic
        linearize_with_ck(g_adj_s, order, original_id, base_size, vert_mask)
    else
        linearize_with_random_forest(g_adj_s, order, original_id, base_size, vert_mask)
    end
    
    append!(order, singletons)
    @assert length(order) == num_vertices
    return order
end

function _select_edges_band(G_adj::SparseMatrixCSC, arrow_width::Int, inverse_permutation::Vector{Int}, prune::Bool)
    num_vertices = size(G_adj, 1)
    l1_edge_list = (Vector{Int}(), Vector{Int}())
    l2_edge_list = (Vector{Int}(), Vector{Int}())
    for v in 1:num_vertices
        start_idx = G_adj.colptr[v]
        end_idx = G_adj.colptr[v+1]-1
        for idx in start_idx:end_idx
            source = v
            target = G_adj.rowval[idx]
            inv_source = inverse_permutation[source] - 1 # convert to 0 based
            inv_target = inverse_permutation[target] - 1
            if abs(inv_source - inv_target) <= arrow_width || (prune && (inv_source < arrow_width || inv_target < arrow_width))
                push!(l1_edge_list[1], source)
                push!(l1_edge_list[2], target)
            end
            if abs(inv_source - inv_target) > arrow_width && (inv_source >= arrow_width || !prune) && (inv_target >= arrow_width || !prune)
                push!(l2_edge_list[1], source)
                push!(l2_edge_list[2], target)
            end
        end
    end
    return l1_edge_list, l2_edge_list
end

function _select_edges_block(G_adj::SparseMatrixCSC, arrow_width::Int, inverse_permutation::Vector{Int}, prune::Bool)
    num_vertices = size(G_adj, 1)
    l1_edge_list = (Vector{Int}(), Vector{Int}())
    l2_edge_list = (Vector{Int}(), Vector{Int}())
    for v in 1:num_vertices
        start_idx = G_adj.colptr[v]
        end_idx = G_adj.colptr[v+1]-1
        for idx in start_idx:end_idx
            source = v
            target = G_adj.rowval[idx]
            inv_source = inverse_permutation[source] - 1 # convert to 0 based
            inv_target = inverse_permutation[target] - 1
            if abs(div(inv_source, arrow_width) - div(inv_target, arrow_width)) == 0 || (prune && (inv_source < arrow_width || inv_target < arrow_width))
                push!(l1_edge_list[1], source)
                push!(l1_edge_list[2], target)
            end
            if abs(div(inv_source, arrow_width) - div(inv_target, arrow_width)) >= 1 && (inv_source >= arrow_width || !prune) && (inv_target >= arrow_width || !prune)
                push!(l2_edge_list[1], source)
                push!(l2_edge_list[2], target)
            end
        end
    end
    return l1_edge_list, l2_edge_list
end

function get_arrow_width(g_adj::SparseMatrixCSC, initial_width::Int)
    width = initial_width
    num_vertices = size(g_adj, 1)
    for v in 1:num_vertices
        start_idx = g_adj.colptr[v]
        end_idx = g_adj.colptr[v+1]-1
        for idx in start_idx:end_idx
            source = v - 1
            target = g_adj.rowval[idx] - 1
            if source > width && target > width
                width = max(width, abs(source - target))
            end
        end
    end
    return width
end

struct ArrowGraph
    graph_adj::SparseMatrixCSC
    permutation::Vector{Int}
    nonzero_rows::Int
    arrow_width::Int
end

function nonzero_rows(g_adj::SparseMatrixCSC)
    return sum(g_adj.colptr[2:end] - g_adj.colptr[1:end-1] .> 0)
end

function _arrow_decomposition(G_adj::SparseMatrixCSC, arrow_width::Int, decomposition::Vector{ArrowGraph}, max_level::Int,
                              original_id::Vector{Int}, block_diagonal::Bool=false, prune::Bool=true, symmetric::Bool=true)
    num_vertices = size(G_adj, 1)
    #deterministic = true
    deterministic = length(decomposition) + 1 >= max_level
    # Compute linearization of the current level
    l1_order = _arrow_linear_order(G_adj, original_id, arrow_width, deterministic, symmetric)

    # Maps from Vertex id to position in the order
    inverse_permutation = sortperm(l1_order)

    if length(decomposition) + 1 < max_level
        if block_diagonal
            l1_edge_list, l2_edge_list = _select_edges_block(G_adj, arrow_width, inverse_permutation, prune)
        else
            l1_edge_list, l2_edge_list = _select_edges_band(G_adj, arrow_width, inverse_permutation, prune)
        end
        if length(l1_edge_list[1]) == 0
            source_list, target_list, _ = findnz(G_adj)
            l1_edge_list = (source_list, target_list)
            l2_edge_list = (Vector{Int}(), Vector{Int}())
        end
    
        num_edges = nnz(G_adj)
        
        @assert length(l1_edge_list[1]) >= 1
        @assert length(l1_edge_list[1]) + length(l2_edge_list[1]) == num_edges
    
        l1_source_perm = inverse_permutation[l1_edge_list[1]]
        l1_target_perm = inverse_permutation[l1_edge_list[2]]
        l1_adj = sparse(l1_source_perm, l1_target_perm, ones(length(l1_source_perm)), num_vertices, num_vertices)
        actual_width = arrow_width
    else
        source_list, target_list, _ = findnz(G_adj)
        l1_source_perm = inverse_permutation[source_list]
        l1_target_perm = inverse_permutation[target_list]
        l1_adj = sparse(l1_source_perm, l1_target_perm, ones(length(l1_source_perm)), num_vertices, num_vertices)
        actual_width = get_arrow_width(l1_adj, arrow_width)
        print("Level ", length(decomposition), ": desired width: ", arrow_width, " - actual width: ", actual_width, "\n")
        l2_edge_list = (Vector{Int}(), Vector{Int}())
    end

    @assert size(l1_adj, 1) == num_vertices
    nzrow = nonzero_rows(l1_adj)
    push!(decomposition, ArrowGraph(l1_adj, l1_order, nzrow, actual_width))
    
    if length(l2_edge_list[1]) > 0
        l2_adj = sparse(l2_edge_list[1], l2_edge_list[2], ones(length(l2_edge_list[1])), num_vertices, num_vertices)
        original_id_l2 = Vector{Int}(1:num_vertices)
        _arrow_decomposition(l2_adj, arrow_width, decomposition, max_level, original_id_l2, block_diagonal, prune, symmetric)
    end

end

function arrow_decomposition(G_adj::SparseMatrixCSC, arrow_width::Int = 512, max_number_of_levels::Int = 2,
                             block_diagonal::Bool = false, prune::Bool = true, symmetric::Bool = true)

    num_vertices = size(G_adj, 1)
    original_id = Vector{Int}(1:num_vertices)
    decomposition = Vector{ArrowGraph}()
    _arrow_decomposition(G_adj, arrow_width, decomposition, max_number_of_levels, original_id, block_diagonal, prune, symmetric)

    return decomposition
end

function read_sparse_matrix(filename::String)
    mat_dict = matread(filename)
    sparse_adj = mat_dict["Problem"]["A"]
    return sparse_adj
end

function store_decomposition(path::String, decomposition::Vector{ArrowGraph}, block_diagonal::Bool, dataset::String)
    level = 0
    for decomp in decomposition
        basename = "$(dataset)_B_$(decomp.arrow_width)_$level"
        if block_diagonal
            basename = basename * "_bd"
        end
        #npzwrite(joinpath(path, basename*"_julia.npz"), Dict("rowval" => decomp.graph_adj.rowval, "colptr" => decomp.graph_adj.colptr, "perm" => decomp.permutation))
        file = matopen(joinpath(path, basename*"_julia.mat"), "w")
        write(file, "rowval", decomp.graph_adj.rowval)
        write(file, "colptr", decomp.graph_adj.colptr)
        write(file, "perm", decomp.permutation)
        close(file)
        level += 1
    end
end

