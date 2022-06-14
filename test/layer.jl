@testset "layer" begin
    heads = 2
    emb_dim = 3
    kdim = 5
    vdim = 7
    head_dim = 11
    pattern_dim = 13

    batch_size = 17
    target_len = 19
    source_len = 23

    in_channel = emb_dim
    hidden_dim = head_dim
    out_channel = 29
    stored_pattern_dim = kdim
    pattern_projection_dim = vdim

    Q = rand(emb_dim, target_len, batch_size)
    K = rand(kdim, source_len, batch_size)
    V = rand(vdim, source_len, batch_size)

    @testset "HopfieldCore" begin
        l = HopfieldCore(emb_dim, heads; kdim=kdim, vdim=vdim,
            head_dim=head_dim, pattern_dim=pattern_dim)
        @test l.linear_q isa Dense
        @test l.linear_k isa Dense
        @test l.linear_v isa Dense
        @test size(l.linear_q(Q)) == (heads*head_dim, target_len, batch_size)
        @test size(l.linear_k(K)) == (heads*head_dim, source_len, batch_size)
        @test size(l.linear_v(V)) == (heads*pattern_dim, source_len, batch_size)

        Y = l(Q, K, V)
        @test size(Y) == (emb_dim, heads*target_len, batch_size)

        g = Zygote.gradient(() -> sum(l(Q, K, V)), Flux.params(l))
        @test length(g.grads) == 6
    end

    @testset "disable out_proj" begin
        l = HopfieldCore(emb_dim, heads; kdim=kdim, vdim=vdim,
            head_dim=head_dim, pattern_dim=pattern_dim, enable_out_proj=false)
        Y = l(Q, K, V)
        @test size(Y) == (heads*pattern_dim, heads*target_len, batch_size)

        g = Zygote.gradient(() -> sum(l(Q, K, V)), Flux.params(l))
        @test length(g.grads) == 4
    end

    @testset "static K, V" begin
        l = HopfieldCore(emb_dim, heads; kdim=kdim, vdim=kdim,
            head_dim=head_dim, pattern_dim=pattern_dim)
        Y = l(Q, nothing, nothing)
        @test size(Y) == (emb_dim, heads*target_len, batch_size)

        g = Zygote.gradient(() -> sum(l(Q, nothing, nothing)), Flux.params(l))
        @test length(g.grads) == 8
    end

    @testset "static Q" begin
        l = HopfieldCore(emb_dim, heads; kdim=kdim, vdim=vdim,
            head_dim=head_dim, pattern_dim=pattern_dim)
        Y = l(nothing, K, V)
        @test size(Y) == (emb_dim, heads*emb_dim, batch_size)

        g = Zygote.gradient(() -> sum(l(nothing, K, V)), Flux.params(l))
        @test length(g.grads) == 8
    end

    @testset "Hopfield" begin
        l = Hopfield(in_channel=>out_channel, hidden_dim, pattern_dim, heads;
            stored_pattern_dim=stored_pattern_dim,
            pattern_projection_dim=pattern_projection_dim)
        @test Hopfields.input_dim(l) == in_channel
        @test Hopfields.hidden_dim(l) == hidden_dim
        @test Hopfields.output_dim(l) == out_channel
        @test Hopfields.pattern_dim(l) == pattern_dim
        @test Hopfields.heads(l) == heads
        @test Hopfields.stored_pattern_dim(l) == stored_pattern_dim
        @test Hopfields.state_pattern_dim(l) == emb_dim
        @test Hopfields.pattern_projection_dim(l) == pattern_projection_dim

        Y = l(Q, K, V)
        @test size(Y) == (out_channel, heads*target_len, batch_size)

        g = Zygote.gradient(() -> sum(l(Q, K, V)), Flux.params(l))
        @test length(g.grads) == 6
    end

    @testset "HopfieldLayer" begin
        l = HopfieldLayer(in_channel=>out_channel, hidden_dim, pattern_dim, heads;
            stored_pattern_dim=stored_pattern_dim)
        @test Hopfields.input_dim(l) == in_channel
        @test Hopfields.hidden_dim(l) == hidden_dim
        @test Hopfields.output_dim(l) == out_channel
        @test Hopfields.pattern_dim(l) == pattern_dim
        @test Hopfields.heads(l) == heads
        @test Hopfields.stored_pattern_dim(l) == stored_pattern_dim
        @test Hopfields.state_pattern_dim(l) == emb_dim
        @test Hopfields.pattern_projection_dim(l) == stored_pattern_dim

        Y = l(Q, nothing, nothing)
        @test size(Y) == (out_channel, heads*target_len, batch_size)

        g = Zygote.gradient(() -> sum(l(Q, nothing, nothing)), Flux.params(l))
        @test length(g.grads) == 8
    end

    @testset "HopfieldPooling" begin
        l = HopfieldPooling(in_channel=>out_channel, hidden_dim, pattern_dim, heads;
            stored_pattern_dim=stored_pattern_dim,
            pattern_projection_dim=pattern_projection_dim)
        @test Hopfields.input_dim(l) == in_channel
        @test Hopfields.hidden_dim(l) == hidden_dim
        @test Hopfields.output_dim(l) == out_channel
        @test Hopfields.pattern_dim(l) == pattern_dim
        @test Hopfields.heads(l) == heads
        @test Hopfields.stored_pattern_dim(l) == stored_pattern_dim
        @test Hopfields.state_pattern_dim(l) == emb_dim
        @test Hopfields.pattern_projection_dim(l) == pattern_projection_dim

        Y = l(nothing, K, V)
        @test size(Y) == (out_channel, heads*emb_dim, batch_size)

        g = Zygote.gradient(() -> sum(l(nothing, K, V)), Flux.params(l))
        @test length(g.grads) == 8
    end
end
