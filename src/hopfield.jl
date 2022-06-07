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

    β = fill(1 / sqrt(emb_dim), 1, 1, heads)
    linear_q = Dense(emb_dim, virtual_hopfield_dim; init=init)
    linear_k = Dense(kdim, virtual_hopfield_dim; bias=bias_kv, init=init)
    linear_v = Dense(vdim, virtual_pattern_dim; bias=bias_kv, init=init)
    out_proj = enable_out_proj ? Dense(virtual_pattern_dim, out_dim; init=init, bias=bias) : identity
    return HopfieldCore(β, linear_q, linear_k, linear_v, out_proj, heads, dropout)
end

@functor HopfieldCore

Flux.trainable(l::HopfieldCore) = (l.β, l.linear_q, l.linear_k, l.linear_v, l.out_proj)

function (l::HopfieldCore)(query::AbstractArray, key::AbstractArray, value::AbstractArray)
    Qt, q = maybe_layer(l.linear_q, query)
    Kt, k = maybe_layer(l.linear_k, key)
    Vt, v = maybe_layer(l.linear_v, value)
    return hopfield_forward(Qt, Kt, Vt, l.out_proj, l.heads, l.β, q, k, v)
end

maybe_layer(l, x) = isnothing(x) ? (identity, l.weight) : (l, x)

function attention_score(Qt, Kt, heads::Int, β::AbstractArray, query::AbstractArray, key::AbstractArray)
    Q = Qt(query)
    K = Kt(key)
    _, targ_len, batch_size = size(Q)
    src_len = size(K, 2)
    
    # (heads*head_dim, L, B) -> (head_dim, L, heads*B)
    Q = reshape(Q, :, heads, targ_len, batch_size)
    Q = reshape(permutedims(Q, (1, 3, 4, 2)), :, targ_len, heads*batch_size)
    K = reshape(K, :, heads, src_len, batch_size)
    K = reshape(permutedims(K, (1, 3, 4, 2)), :, src_len, heads*batch_size)

    return repeat(β, 1, 1, batch_size) .* batched_mul(batched_transpose(Q), K)
end

function attention_prob(Qt, Kt, heads::Int, β::AbstractArray, query::AbstractArray, key::AbstractArray)
    return softmax(attention_score(Qt, Kt, heads, β, query, key), dims=2)
end

function hopfield_forward(Qt, Kt, Vt, out_proj, heads::Int, β::AbstractArray, query::AbstractArray, key::AbstractArray, value::AbstractArray)
    Â = attention_prob(Qt, Kt, heads, β, query, key)
    targ_len, src_len, d = size(Â)
    batch_size = d ÷ heads

    # (T, S, heads*B) -> (heads*T, S, B)
    Â = reshape(Â, targ_len, src_len, heads, batch_size)
    Â = reshape(permutedims(Â, (3, 1, 2, 4)), heads*targ_len, src_len, batch_size)

    V = Vt(value)

    attn_out = batched_mul(Â, batched_transpose(V))
    
    return out_proj(batched_transpose(attn_out))
end
