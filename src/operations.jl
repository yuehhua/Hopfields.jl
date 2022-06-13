function batched_innerprod(A::AbstractArray{T,3}, B::AbstractArray{S,3}) where {T,S}
    a_dim = size(A, 3)
    b_dim = size(B, 3)

    if a_dim == b_dim
        return _batched_innerprod(A, B)
    elseif a_dim ÷ b_dim > 0 && a_dim % b_dim == 0
        dims = size(A, 2), size(B, 2)
        relaxed_A = relax_dims(A, a_dim, b_dim)
        C = _batched_innerprod(relaxed_A, B)
        return reshape(C, dims..., :)
    elseif b_dim ÷ a_dim > 0 && b_dim % a_dim == 0
        dims = size(A, 2), size(B, 2)
        relaxed_B = relax_dims(B, b_dim, a_dim)
        C = _batched_innerprod(A, relaxed_B)
        return reshape(C, dims..., :)
    else
        throw(ArgumentError("not supported array size of $(size(A)) and $(size(B))"))
    end
end

_batched_innerprod(A::AbstractArray{T,3}, B::AbstractArray{S,3}) where {T,S} =
    batched_mul(batched_transpose(A), B)

_batched_innerprod(A::AbstractArray{T,4}, B::AbstractArray{S,3}) where {T,S} =
    @tullio C[i, j, l, b] := A[k, i, l, b] * B[k, j, l]

_batched_innerprod(A::AbstractArray{T,3}, B::AbstractArray{S,4}) where {T,S} =
    @tullio C[i, j, l, b] := A[k, i, l] * B[k, j, l, b]
