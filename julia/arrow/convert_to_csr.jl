"""
The julia script takes files of the form {filename}_julia.mat
that contain a CSC matrix, stored as follows:
rowval: the row indices of the nonzero entries of the matrix
colptr: the column pointers of the matrix
perm: the permutation vector used to permute the matrix
optionally, it also contains:
data: the nonzero entries of the matrix

The script reads the data in the CSC matrix and converts it to a CSR matrix by transposing it.

The script exports three files:
{filename}_indptr.npy: the indptr of the CSR matrix
{filename}_indices.npy: the indices of the CSR matrix
{filename}_permutation.npy: the permutation vector used to permute the matrix

if data is provided, additionally:
{filename}_data.npy: the nonzero entries of the matrix
"""

using MAT
using NPZ
using DataStructures
using ArgParse
using SparseArrays

function main(file)
    # Assert that the file ends with _julia.mat suffix:
    @assert (length(file) > 10)
    @assert (file[end-9:end] == "_julia.mat")

    println("Reading data from file ...")

    mat = matread(file)
    rowval = mat["rowval"]
    colptr = mat["colptr"]
    perm = mat["perm"]
    # Check if data is provided:
    if haskey(mat, "data")
        data = mat["data"]
    else
        data = nothing
    end

    weighted = false
    println("Converting to 0-based indexing ...")

    # Get n:
    n = length(colptr) - 1

    # Create the sparse matrix CSR by transposing the CSC matrix:
    colptr = colptr .- 1
    rowval = rowval .- 1

    println("Saving data to files ...")

    # Extract file without _julia.mat suffix:
    file = file[1:end-10]

    # Save the data as npy files
    npzwrite(file * "_indptr.npy", colptr)
    npzwrite(file * "_indices.npy", rowval)
    npzwrite(file * "_permutation.npy", perm)
    if weighted
        npzwrite(file * "_data.npy", data)
    end
    println("Done!")
end


# Parse the arguments and call main:
s = ArgParseSettings()
@add_arg_table s begin
    "--file"
        arg_type = String
        default = ""
end
parsed_args = parse_args(ARGS, s)
main(parsed_args["file"])

