"""
    HopfieldCore(emb_dim, heads)

Hopfield layer core.

# Arguments

- `Q`: (E, T, B), E is the embedding dimension, T is the target sequence length, B is the batch size
- `K`: (E, S, B), E is the embedding dimension, S is the source sequence length, B is the batch size
- `V`: (E, S, B), E is the embedding dimension, S is the source sequence length, B is the batch size
"""
struct HopfieldCore{B,Q,K,V,O,D,T}
    β::B
    linear_q::Q
    linear_k::K
    linear_v::V
    out_proj::O
    dropout::D
    heads::Int
    max_iter::Int
    ϵ::T
end

function HopfieldCore(emb_dim::Int, heads::Int=1;
    dropout::Real=0.f0, bias::Bool=true, bias_kv=false,
    kdim::Int=0, vdim::Int=0, out_dim::Int=0,
    head_dim::Int=0, pattern_dim::Int=0,
    enable_out_proj::Bool=true, init=glorot_uniform,
    normalize_pattern::Bool=false, normalize_pattern_affine::Bool=false,
    max_iter::Int=0, ϵ::Real=1f-4,
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
    dropout_l = (dropout == 0.) ? identity : Dropout(dropout; dims=3)
    return HopfieldCore(β, linear_q, linear_k, linear_v, out_proj,
        dropout_l, heads, max_iter, ϵ)
end

@functor HopfieldCore

Flux.trainable(l::HopfieldCore) = (l.linear_q, l.linear_k, l.linear_v, l.out_proj)

function (l::HopfieldCore)(query, key, value)
    Qt, q = maybe_layer(l.linear_q, query)
    Kt, k = maybe_layer(l.linear_k, key)
    Vt, v = maybe_layer(l.linear_v, value)
    bch_sz = batch_size(q, k, v)
    β = reshape(repeat(l.β, bch_sz), 1, 1, :)
    return hopfield_forward(
        Qt, Kt, Vt, l.out_proj, l.dropout, l.heads,
        β, q, k, v, l.max_iter, l.ϵ)
end

input_dim(l::HopfieldCore) = size(l.linear_q.weight, 2)
hidden_dim(l::HopfieldCore) = size(l.linear_k.weight, 1) ÷ heads(l)
function output_dim(l::HopfieldCore)
    if l.out_proj == identity
        return size(l.linear_v.weight, 1)
    else
        return size(l.out_proj.weight, 1)
    end
end
pattern_dim(l::HopfieldCore) = size(l.linear_v.weight, 1) ÷ heads(l)
heads(l::HopfieldCore) = l.heads
stored_pattern_dim(l::HopfieldCore) = size(l.linear_k.weight, 2)
state_pattern_dim(l::HopfieldCore) = size(l.linear_q.weight, 2)
pattern_projection_dim(l::HopfieldCore) = size(l.linear_v.weight, 2)
# association_matrix(l::HopfieldCore) = 
# projected_pattern_matrix(l::HopfieldCore) = 

function Hopfield(ch::Pair{Int,Int}, hidden_dim::Int, pattern_dim::Int, heads::Int;
    dropout::Real=0.f0, bias::Bool=true, pattern_bias::Bool=false,
    stored_pattern_dim::Int=0, pattern_projection_dim::Int=0,
    enable_out_proj::Bool=true, init=glorot_uniform,
    normalize_hopfield::Bool=false, normalize_hopfield_affine::Bool=false,
    max_iter::Int=0, ϵ::Real=1f-4,
    )

    in_ch, out_ch = ch
    l = HopfieldCore(in_ch, heads; dropout=dropout, bias=bias, bias_kv=pattern_bias,
        head_dim=hidden_dim, out_dim=out_ch, pattern_dim=pattern_dim,
        kdim=stored_pattern_dim, vdim=pattern_projection_dim,
        enable_out_proj=enable_out_proj, init=init,
        normalize_pattern=normalize_hopfield,
        normalize_pattern_affine=normalize_hopfield_affine,
        max_iter=max_iter, ϵ=ϵ,
    )
    return l
end

function HopfieldLayer(ch::Pair{Int,Int}, hidden_dim::Int, pattern_dim::Int, heads::Int;
    dropout::Real=0.f0, bias::Bool=true, pattern_bias::Bool=false,
    stored_pattern_dim::Int=0,
    enable_out_proj::Bool=true, init=glorot_uniform,
    normalize_hopfield::Bool=false, normalize_hopfield_affine::Bool=false,
    max_iter::Int=0, ϵ::Real=1f-4,
    )

    in_ch, out_ch = ch
    l = HopfieldCore(in_ch, heads; dropout=dropout, bias=bias, bias_kv=pattern_bias,
        head_dim=hidden_dim, out_dim=out_ch, pattern_dim=pattern_dim,
        kdim=stored_pattern_dim, vdim=stored_pattern_dim,
        enable_out_proj=enable_out_proj, init=init,
        normalize_pattern=normalize_hopfield,
        normalize_pattern_affine=normalize_hopfield_affine,
        max_iter=max_iter, ϵ=ϵ,
    )
    return l
end

function HopfieldPooling(ch::Pair{Int,Int}, hidden_dim::Int, pattern_dim::Int, heads::Int;
    dropout::Real=0.f0, bias::Bool=true, pattern_bias::Bool=false,
    stored_pattern_dim::Int=0, pattern_projection_dim::Int=0,
    enable_out_proj::Bool=true, init=glorot_uniform,
    normalize_hopfield::Bool=false, normalize_hopfield_affine::Bool=false,
    max_iter::Int=0, ϵ::Real=1f-4,
    )

    in_ch, out_ch = ch
    l = HopfieldCore(in_ch, heads; dropout=dropout, bias=bias, bias_kv=pattern_bias,
        head_dim=hidden_dim, out_dim=out_ch, pattern_dim=pattern_dim,
        kdim=stored_pattern_dim, vdim=pattern_projection_dim,
        enable_out_proj=enable_out_proj, init=init,
        normalize_pattern=normalize_hopfield,
        normalize_pattern_affine=normalize_hopfield_affine,
        max_iter=max_iter, ϵ=ϵ,
    )
    return l
end
