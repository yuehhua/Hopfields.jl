@testset "operations" begin
    A = rand(2, 3, 7)
    B = rand(2, 5, 7)
    @test size(Hopfields.batched_innerprod(A, B)) == (3, 5, 7)

    A = rand(2, 3, 77)
    B = rand(2, 5, 7)
    @test size(Hopfields.batched_innerprod(A, B)) == (3, 5, 77)

    A = rand(2, 3, 7)
    B = rand(2, 5, 77)
    @test size(Hopfields.batched_innerprod(A, B)) == (3, 5, 77)
end
