"""
    hopfield_forward(Qt, Kt, Vt, out_proj, dropout, heads,
                     β, query, key, value, max_iter, ϵ)

Hopfield layer forward function.

# Arguments

- `β::AbstractArray`:
- `query::AbstractArray`:
- `key::AbstractArray`:
- `value::AbstractArray`:
- `max_iter`: `-1` for iteration until stable.
- `ϵ`: 
"""
function hopfield_forward(Qt, Kt, Vt, out_proj, dropout, heads::Int,
    β::AbstractArray, query::AbstractArray, key::AbstractArray, value::AbstractArray,
    max_iter, ϵ)
    max_iter = init_tensor(max_iter, query, heads)  # max_iter ∈ (1, 1, heads)
    ϵ = init_tensor(ϵ, query, heads)  # ϵ ∈ (1, 1, heads)

    Q = project(Qt, heads, query)
    K = project(Kt, heads, key)
    V = project(Vt, heads, value)
    
    Â = attention_prob(Q, K, β)
    Â = multiple_updates(Â, Q, K, β, heads, max_iter, ϵ)
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

function multiple_updates(Â, Q, K, β, heads::Int, max_iter::AbstractArray, ϵ::AbstractArray)
    bch_sz = size(Â, 3) ÷ heads
    step = 1
    active_heads = init_active_heads(Q, heads)  # (1, 1, heads)
    update_active_heads!(active_heads, step, max_iter)
    old_Â = reshape(Â, size(Â)[1:2]..., heads, bch_sz)
    while any(active_heads)
        if step == 0
            # some stuff
        else
            new_Q = batched_innerprod(K, Â; dims=2)
            new_Q = masked_select(active_heads, new_Q, Q)
            Â = attention_prob(new_Q, K, β)
        end

        # update active_heads and step
        new_Â = reshape(Â, size(Â)[1:2]..., heads, bch_sz)
        update_active_heads!(active_heads, step, max_iter)
        update_active_heads!(active_heads, old_Â, new_Â, ϵ)
        old_Â = new_Â
        step += 1
    end
    return Â
end
