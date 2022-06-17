function batched_innerprod(A::AbstractArray{T,3}, B::AbstractArray{S,3}; dims::Int=1) where {T,S}
    A, B = unsqueeze_batch(A, B)
    C = batched_innerprod(A, B, Val(dims))
    return squeeze_batch(C)
end

batched_innerprod(A::AbstractArray{T,3}, B::AbstractArray{S,3}, ::Val{1}) where {T,S} =
    batched_mul(batched_transpose(A), B)

batched_innerprod(A::AbstractArray{T,4}, B::AbstractArray{S,3}, ::Val{1}) where {T,S} =
    @tullio C[i, j, l, b] := A[k, i, l, b] * B[k, j, l]

batched_innerprod(A::AbstractArray{T,3}, B::AbstractArray{S,4}, ::Val{1}) where {T,S} =
    @tullio C[i, j, l, b] := A[k, i, l] * B[k, j, l, b]

batched_innerprod(A::AbstractArray{T,3}, B::AbstractArray{S,3}, ::Val{2}) where {T,S} =
    batched_mul(A, batched_transpose(B))

batched_innerprod(A::AbstractArray{T,4}, B::AbstractArray{S,3}, ::Val{2}) where {T,S} =
    @tullio C[i, j, l, b] := A[i, k, l, b] * B[j, k, l]

batched_innerprod(A::AbstractArray{T,3}, B::AbstractArray{S,4}, ::Val{2}) where {T,S} =
    @tullio C[i, j, l, b] := A[i, k, l] * B[j, k, l, b]

function masked_select(mask, A, B)
    mask, A = unsqueeze_batch(mask, A)
    mask, B = unsqueeze_batch(mask, B)
    C = ifelse.(mask, A, B)
    return squeeze_batch(C)
end

norm2(A::AbstractArray; dims=1) = sqrt.(sum(abs2, A, dims=dims))
max_norm2(A::AbstractArray) = maximum(norm2(A; dims=(1,2)), dims=4)

function init_active_heads(A::AbstractArray, heads::Int)
    return fill!(similar(A, Bool, 1, 1, heads), true)
end

function update_active_heads!(active_heads, step::Int, max_iter::AbstractArray)
    active_heads .= ((step .< max_iter) .| (max_iter .< 0))
    return active_heads
end

function update_active_heads!(active_heads, old_Â::AbstractArray, new_Â::AbstractArray, ϵ::AbstractArray)
    ΔÂ = reshape(max_norm2(old_Â - new_Â), 1, 1, :)
    active_heads .&= (ΔÂ .> ϵ)
    return active_heads
end

@non_differentiable init_active_heads(x...)
@non_differentiable update_active_heads!(x...)
