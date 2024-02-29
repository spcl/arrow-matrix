using DataStructures  # for PriorityQueue
using Base.Iterators: flatten
using SparseArrays
using NPZ

# Disjoint set data structure for union-find operations
mutable struct DisjointSet
    parent::Vector{Int}
    rank::Vector{Int}
end

function DisjointSet(n::Int)
    return DisjointSet(collect(1:n), zeros(Int, n))
end

function find_root(ds::DisjointSet, x::Int)
    if ds.parent[x] != x
        ds.parent[x] = find_root(ds, ds.parent[x])  # path compression
    end
    return ds.parent[x]
end

function union!(ds::DisjointSet, x::Int, y::Int)
    root_x = find_root(ds, x)
    root_y = find_root(ds, y)
    
    if root_x == root_y
        return false
    end
    
    # union by rank
    if ds.rank[root_x] < ds.rank[root_y]
        ds.parent[root_x] = root_y
    elseif ds.rank[root_x] > ds.rank[root_y]
        ds.parent[root_y] = root_x
    else
        ds.parent[root_y] = root_x
        ds.rank[root_x] += 1
    end
    return true
end

# Kruskal's algorithm
# TODO: replace this with Prim's algorithm referencing bfs + connected component function
function minimum_spanning_forest(adj_matrix::SparseMatrixCSC, weights::Vector{Float32}, vert_mask::Union{BitVector, Nothing}=nothing, symmetric::Bool=true)
    n = size(adj_matrix, 1)
    ds = DisjointSet(n)
    edges = PriorityQueue{Tuple{Int, Int}, Float32}()
    enable_mask = !isnothing(vert_mask)
    
    for col in 1:n
        for idx in nzrange(adj_matrix, col)
            row = rowvals(adj_matrix)[idx]
            weight = weights[idx]
            if enable_mask && (!vert_mask[row] || !vert_mask[col])
                continue
            end
            if row < col || !symmetric  # to ensure we don't double-count undirected edges
                enqueue!(edges, (row, col), weight)
            end
        end
    end
    
    mst_edges = (Vector{Int}(), Vector{Int}())
    
    while !isempty(edges) && length(mst_edges) < n - 1
        edge = dequeue!(edges)
        u, v = edge
        if union!(ds, u, v)
            push!(mst_edges[1], u)
            push!(mst_edges[2], v)
            if symmetric
                push!(mst_edges[1], v)
                push!(mst_edges[2], u)
            end
        end
    end
    
    return mst_edges
end


function bfs(node::Int, csc_matrix::SparseMatrixCSC, visited::BitVector, 
             vert_mask::Union{BitVector, Nothing}=nothing, 
             return_levels::Bool=false, return_edges::Bool=false, symmetric::Bool=true,
             weights::Union{Vector{Float32}, Nothing}=nothing)
    queue = Tuple{Int, Int}[]
    push!(queue, (node, 0))
    component = Int[]
    levels = Int[]
    edges = (Int[], Int[])
    enable_mask = !isnothing(vert_mask)
    weighted_search = !isnothing(weights)
    
    while !isempty(queue)
        current, level = popfirst!(queue) # BFS
        #current, level = pop!(queue) # DFS
        push!(component, current)
        if return_levels
            push!(levels, level)
        end
        
        start_idx = csc_matrix.colptr[current]
        end_idx = csc_matrix.colptr[current+1]-1

        neighbor_list = Int[]
        neighbor_weights = Float32[]
        
        for idx in start_idx:end_idx
            neighbor = csc_matrix.rowval[idx]
            if enable_mask && !vert_mask[neighbor]
                continue
            end
            if !visited[neighbor]
                visited[neighbor] = true
                if !weighted_search
                    push!(queue, (neighbor, level+1))
                    if return_edges
                        push!(edges[1], current)
                        push!(edges[2], neighbor)
                        if symmetric
                            push!(edges[1], neighbor)
                            push!(edges[2], current)
                        end
                    end
                else
                    push!(neighbor_list, neighbor)
                    push!(neighbor_weights, weights[idx])
                end
            end
        end

        # TODO: make this a priority queue
        if weighted_search
            sortind = sortperm(neighbor_weights)
            for idx in sortind
                neighbor = neighbor_list[idx]
                push!(queue, (neighbor, level+1))
                if return_edges
                    push!(edges[1], current)
                    push!(edges[2], neighbor)
                    if symmetric
                        push!(edges[1], neighbor)
                        push!(edges[2], current)
                    end
                end
            end
        end
    end
    if return_levels
        return component, levels
    elseif return_edges
        return component, edges
    else
        return component
    end
end

function connected_components(csc_matrix::SparseMatrixCSC, visited::BitVector, 
                              vert_mask::Union{BitVector, Nothing}=nothing, 
                              weights::Union{Vector{Float32}, Nothing}=nothing,
                              symmetric::Bool=true)    
    components_list = Int[]  # list to store individual components
    offsets = [1]  # start offset for the first component
    return_levels = false
    return_edges = !isnothing(weights)
    edges = (Int[], Int[])

    num_vertices = size(csc_matrix, 1)
    enable_mask = !isnothing(vert_mask)
    
    for v in 1:num_vertices
        if enable_mask && !vert_mask[v]
            continue
        end
        if !visited[v]
            visited[v] = true
            if !return_edges
                component = bfs(v, csc_matrix, visited, vert_mask, return_levels, return_edges, symmetric)
            else
                component, edges_cc = bfs(v, csc_matrix, visited, vert_mask, return_levels, return_edges, symmetric, weights)
                append!(edges[1], edges_cc[1])
                append!(edges[2], edges_cc[2])
            end
            append!(components_list, component)
            push!(offsets, offsets[end] + length(component))
        end
    end
    
    if return_edges
        return components_list, offsets, edges
    else
        return components_list, offsets
    end
end
