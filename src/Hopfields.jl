module Hopfields

using CUDA
using ChainRulesCore
using ChainRulesCore: @non_differentiable
using Flux, NNlib, NNlibCUDA
using Flux: glorot_uniform
using Flux: @functor
using Tullio, KernelAbstractions, CUDAKernels

export
    HopfieldCore,
    Hopfield,
    HopfieldLayer,
    HopfieldPooling

include("utils.jl")
include("operations.jl")
include("functional.jl")
include("layer.jl")

end
