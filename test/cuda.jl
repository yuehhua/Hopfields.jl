@testset "cuda" begin
    heads = 2
    emb_dim = 3
    kdim = 5
    vdim = 7
    head_dim = 11
    pattern_dim = 13

    batch_size = 17
    target_len = 19
    source_len = 23

    Q = rand(emb_dim, target_len, batch_size) |> gpu
    K = rand(kdim, source_len, batch_size) |> gpu
    V = rand(vdim, source_len, batch_size) |> gpu

    l = HopfieldCore(emb_dim, heads; kdim=kdim, vdim=vdim,
        head_dim=head_dim, pattern_dim=pattern_dim) |> gpu
    Y = l(Q, K, V)
    @test size(Y) == (emb_dim, heads*target_len, batch_size)

    g = Zygote.gradient(() -> sum(l(Q, K, V)), Flux.params(l))
    @test length(g.grads) == 6
end
