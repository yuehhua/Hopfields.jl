using Hopfields
using CUDA
using Flux
using Test
using Zygote

cuda_tests = [
    "cuda.jl"
]

tests = [
    "operations.jl",
    "layer.jl",
]

if CUDA.functional()
    CUDA.allowscalar(false)
    append!(tests, cuda_tests)
else
    @warn "CUDA unavailable, not testing GPU support"
end

@testset "Hopfields.jl" begin
    for t in tests
        include(t)
    end
end
