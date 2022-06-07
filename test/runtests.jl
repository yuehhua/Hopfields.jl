using scTransformer
using CUDA
using Flux
using Test
using Zygote

@testset "scTransformer.jl" begin
    include("hopfield.jl")
end
