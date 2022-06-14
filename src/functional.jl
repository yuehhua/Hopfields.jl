function hopfield_forward(Qt, Kt, Vt, out_proj, dropout, heads::Int,
    β::AbstractArray, query::AbstractArray, key::AbstractArray, value::AbstractArray)

    Q = project(Qt, heads, query)
    K = project(Kt, heads, key)
    V = project(Vt, heads, value)
    
    Â = attention_prob(Q, K, β)
    Â = move_heads_to_first(Â, heads)
    V = move_heads_to_first(V, heads)

    V = dropout(V)
    attn_out = batched_mul(V, batched_transpose(Â))
    return out_proj(attn_out)
end

attention_prob(Q::AbstractArray, K::AbstractArray, β::AbstractArray) =
    softmax(attention_score(Q, K, β), dims=2)

attention_score(Q::AbstractArray, K::AbstractArray, β::AbstractArray) =
    β .* batched_innerprod(Q, K)

function project(layer, heads::Int, X::AbstractArray)
    X = layer(X)
    X = move_heads_to_last(X, heads)
    return X
end
