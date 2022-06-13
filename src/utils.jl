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
