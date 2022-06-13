module scTransformer

using CUDA
using Flux, NNlib, NNlibCUDA
using Flux: glorot_uniform
using Flux: @functor

export
    HopfieldCore

include("utils.jl")
include("hopfield.jl")

end
