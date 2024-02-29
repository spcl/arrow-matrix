include("ArrowDecomposition.jl")

function main(parsed_args)
    dataset = parsed_args["dataset_name"]
    mat_path = joinpath(parsed_args["dataset_path"], "$dataset.mat")
    sparse_adj = read_sparse_matrix(mat_path)
    arrow_width = parsed_args["arrow_width"]
    symmetric = parsed_args["symmetric"]

    @assert (arrow_width > 0)
    @assert (arrow_width <= size(sparse_adj, 1)) "Arrow width cannot be greater than the number of vertices"

    block_diagonal = true
    prune = true
    decomposition = Vector{ArrowGraph}()

    print("Dataset: ", dataset, ", num_vertices: ", size(sparse_adj, 1), ", num_edges: ", nnz(sparse_adj), ", symmetric: ", symmetric, "\n")

    @time begin
        decomposition = arrow_decomposition(sparse_adj, arrow_width, parsed_args["max_decomp"], block_diagonal, prune, symmetric)
    end
    store_decomposition(parsed_args["save_path"], decomposition, block_diagonal, dataset)
    
    level = 0
    for decomp in decomposition
        print("Level ", level, ", nonzero_rows ", decomp.nonzero_rows, ", actual_width ", decomp.arrow_width, ", desired_width ", arrow_width,"\n")
        level += 1
    end
end

s = ArgParseSettings()
@add_arg_table s begin
    "--dataset_name"
        arg_type = String
        default = "685_bus"
    "--dataset_path"
        arg_type = String
        default = "./test/data"
    "--save_path"
        arg_type = String
        default = "./test/result"
    "--max_decomp"
        arg_type = Int
        default = 5
    "--arrow_width"
        arg_type = Int
        default = 5000000
    "--symmetric"
        arg_type = Bool
        default = true
end
parsed_args = parse_args(ARGS, s)
main(parsed_args)