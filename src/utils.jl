maybe_layer(l, x) = l, x
maybe_layer(l, ::Nothing) = identity, l.weight

function move_heads_to_last(X::AbstractArray, heads::Int)
    # (heads*head_dim, L, B) -> (head_dim, L, heads*B)
    _, len, batch_size = size(X)
    X = reshape(X, :, heads, len, batch_size)
    X = reshape(permutedims(X, (1, 3, 4, 2)), :, len, heads*batch_size)
    return X
end

function move_heads_to_last(X::AbstractMatrix, heads::Int)
    # (heads*head_dim, L) -> (head_dim, L, heads)
    _, len = size(X)
    X = reshape(X, :, heads, len)
    X = permutedims(X, (1, 3, 2))
    return X
end

function move_heads_to_first(X::AbstractArray, heads::Int)
    # (T, S, heads*B) -> (heads*T, S, B)
    dim1, dim2, hb = size(X)
    batch_size = hb ÷ heads
    X = reshape(X, dim1, dim2, heads, batch_size)
    X = reshape(permutedims(X, (3, 1, 2, 4)), heads*dim1, dim2, batch_size)
    return X
end

batch_size(q::AbstractArray, k, v) = size(q, 3)
batch_size(q::AbstractMatrix, k::AbstractArray, v) = size(k, 3)
batch_size(q::AbstractMatrix, k::AbstractMatrix, v::AbstractArray) = size(v, 3)

squeeze_batch(X::AbstractArray) = reshape(X, size(X)[1:2]..., :)

function unsqueeze_batch(A::AbstractArray{T,3}, B::AbstractArray{S,3}) where {T,S}
    a_dim = size(A, 3)
    b_dim = size(B, 3)

    if a_dim == b_dim
        return A, B
    elseif a_dim ÷ b_dim > 0 && a_dim % b_dim == 0
        return unsqueeze_batch(A, a_dim, b_dim), B
    elseif b_dim ÷ a_dim > 0 && b_dim % a_dim == 0
        return A, unsqueeze_batch(B, b_dim, a_dim)
    else
        throw(ArgumentError("not supported array size of $(size(A)) and $(size(B))"))
    end
end

function unsqueeze_batch(X::AbstractArray{T,3}, dim1::Int, dim2::Int) where {T}
    bch_sz = dim1 ÷ dim2
    return reshape(X, size(X)[1:2]..., dim2, bch_sz)
end

function init_tensor(A::AbstractArray{<:Real,3}, ::AbstractArray, heads::Int)
    @assert size(A) == (1, 1, heads)
    return A
end

function init_tensor(x::Real, S::AbstractArray, heads::Int)
    return fill!(similar(S, 1, 1, heads), x)
end

@non_differentiable init_tensor(x...)
