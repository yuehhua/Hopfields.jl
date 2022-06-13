"""
    HopfieldCore(emb_dim, heads)

Hopfield layer core.

# Arguments

- `Q`: (E, T, B), E is the embedding dimension, T is the target sequence length, B is the batch size
- `K`: (E, S, B), E is the embedding dimension, S is the source sequence length, B is the batch size
- `V`: (E, S, B), E is the embedding dimension, S is the source sequence length, B is the batch size
"""
struct HopfieldCore{B,Q,K,V,O,D<:Real}
    β::B
    linear_q::Q
    linear_k::K
    linear_v::V
    out_proj::O
    heads::Int
    dropout::D
end

function HopfieldCore(emb_dim::Int, heads::Int=1;
    dropout::Real=0., bias::Bool=true, bias_kv=false,
    kdim::Int=0, vdim::Int=0, out_dim::Int=0,
    head_dim::Int=0, pattern_dim::Int=0,
    enable_out_proj::Bool=true,
    # normalize_pattern::Bool=false, normalize_pattern_affine::Bool=false
    init=glorot_uniform,
    )
    kdim = (kdim > 0) ? kdim : emb_dim
    vdim = (vdim > 0) ? vdim : emb_dim
    out_dim = (out_dim > 0) ? out_dim : emb_dim
    head_dim = (head_dim > 0) ? head_dim : emb_dim
    pattern_dim = (pattern_dim > 0) ? pattern_dim : head_dim
    virtual_hopfield_dim = heads * head_dim
    virtual_pattern_dim = heads * pattern_dim

    β = fill(1 / sqrt(emb_dim), heads)
    linear_q = Dense(emb_dim, virtual_hopfield_dim; init=init)
    linear_k = Dense(kdim, virtual_hopfield_dim; bias=bias_kv, init=init)
    linear_v = Dense(vdim, virtual_pattern_dim; bias=bias_kv, init=init)
    out_proj = enable_out_proj ? Dense(virtual_pattern_dim, out_dim; init=init, bias=bias) : identity
    return HopfieldCore(β, linear_q, linear_k, linear_v, out_proj, heads, dropout)
end

@functor HopfieldCore

Flux.trainable(l::HopfieldCore) = (l.linear_q, l.linear_k, l.linear_v, l.out_proj)

function (l::HopfieldCore)(query, key, value)
    Qt, q = maybe_layer(l.linear_q, query)
    Kt, k = maybe_layer(l.linear_k, key)
    Vt, v = maybe_layer(l.linear_v, value)
    bch_sz = batch_size(q, k, v)
    β = reshape(repeat(l.β, bch_sz), 1, 1, :)
    return hopfield_forward(Qt, Kt, Vt, l.out_proj, l.heads, β, q, k, v)
end

function hopfield_forward(Qt, Kt, Vt, out_proj, heads::Int, β::AbstractArray, query::AbstractArray, key::AbstractArray, value::AbstractArray)
    Q = project(Qt, heads, query)
    K = project(Kt, heads, key)
    V = project(Vt, heads, value)
    
    Â = attention_prob(Q, K, β)
    Â = move_heads_to_first(Â, heads)
    V = move_heads_to_first(V, heads)

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
