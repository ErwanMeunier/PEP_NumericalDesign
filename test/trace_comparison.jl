@testset "matched best-so-far advantages" begin
    trace(values, times) = DesignTrace([[Float64(i)] for i in eachindex(values)],
        Float64.(values), collect(1:length(values)), length(values),
        collect(0:(length(values) - 1)), Float64.(times))
    free = trace([10, 8, 9, 4], [1, 3, 4, 8])
    prior = trace([10, 6, 7, 5], [2, 3, 5, 6])
    c = relative_advantage(free, prior; axis = :iterations)
    @test c.budgets == [0, 1, 2, 3]
    @test c.free_values == [10, 8, 8, 4]
    @test c.prior_values == [10, 6, 6, 5]
    @test c.advantage == [0, 0.25, 0.25, -0.25]
    t = relative_advantage(free, prior)
    @test t.budgets == [2, 3, 4, 5, 6]
    @test t.advantage == [0, 0.25, 0.25, 0.25, 0.375]
    censored = relative_advantage(free, prior; budgets = [0, 2, 7, 100])
    @test ismissing(censored.advantage[1])
    @test censored.advantage[2] == 0
    @test all(ismissing, censored.advantage[3:4])
    @test relative_advantage(free, prior; axis = :solves).budgets == [1, 3, 5, 7]
    @test_throws ArgumentError relative_advantage(free, prior; axis = :unknown)
    @test_throws ArgumentError relative_advantage(free, prior; budgets = [2, 1])
    @test_throws ArgumentError relative_advantage(trace([0, 0], [1, 2]), prior)
    @test_throws ArgumentError relative_advantage(trace([1, 2], [NaN, NaN]), prior)
end
