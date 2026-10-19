using Test, Random
using Hyperopt, Plots

@testset "Small defects" begin

    @testset "show" begin
        ho = Hyperoptimizer(iterations=2, params=(:a, :g), candidates=(LinRange(1, 5, 10), [sin, cos]),
            history=Any[[1.0, sin], [5.0, cos]], results=Any[1.0, 2.0])
        str = sprint(show, ho)
        @test endswith(str, "\n")
        @test occursin("  a length: 10", str)
        @test occursin("  g candidates: [sin, cos]", str)
        ho = Hyperoptimizer(iterations=2, params=(:a, :b, :n), candidates=(LinRange(1, 5, 10), [true, false], 1:100),
            history=Any[Any[1.0, true, 12345], Any[5.0, false, 2]], results=Any[1.0, 2.0])
        str = sprint(show, ho)
        @test occursin("  b candidates: [true, false]", str)
        @test occursin(r"\s1\s+true\s+12345", str) # Bool and Integer values are printed as such
        @test !occursin("Bool[", str)
        path, io = mktemp()
        redirect_stdout(io) do
            show(IOBuffer(), ho)
            show(IOBuffer(), Hyperoptimizer(3, a = LinRange(1, 2, 5)))
        end
        close(io)
        @test isempty(read(path, String))
    end

    @testset "warn_on_boundary" begin
        ho = Hyperoptimizer(iterations=3, params=(:g, :x), candidates=([sin, cos, tanh, exp], LinRange(0, 1, 10)),
            history=Any[[tanh, 0.5], [exp, 0.0], [sin, 1.0]], results=Any[0.1, 1.0, 0.8])
        # The categorical parameter g is not reported, and x = 0.5 is not an extreme point of the sampled values
        @test (@test_logs warn_on_boundary(ho)) == Symbol[]
        @test (@test_logs (:warn, r"Parameter x .*: 0.0") warn_on_boundary(ho, :max)) == [:x]
        ho.results[1] = 2.0
        @test (@test_logs (:warn, r"Parameter x .*: 1.0") warn_on_boundary(ho)) == [:x]
    end

    @testset "minimum and maximum" begin
        ho = Hyperoptimizer(3, a = LinRange(1, 2, 5))
        @test isnan(minimum(ho))
        @test isnan(maximum(ho))
        @test isnan(ho.minimum)
        @test isnan(ho.maximum)
        append!(ho.history, [[1.0], [1.5], [2.0]])
        append!(ho.results, [NaN, 2.0, 1.0])
        @test minimum(ho) == ho.minimum == 1.0
        @test maximum(ho) == ho.maximum == 2.0
    end

    @testset "Iteration" begin
        ho = Hyperoptimizer(3, a = LinRange(1, 2, 5), b = [true, false])
        @test eltype(ho) == NamedTuple
        @test Base.IteratorEltype(ho) == Base.EltypeUnknown()
        samples = collect(ho)
        @test length(samples) == 3
        @test isconcretetype(eltype(samples))
        @test eltype(samples) <: NamedTuple{(:i, :a, :b)}

        # Samples without results are not accumulated by repeated iteration
        collect(ho)
        @test length(ho.history) == 3
        @test isempty(ho.results)

        for (i, a, b) in ho
            @test length(ho.history) == length(ho.results) + 1
            @test ho.history[end] == [a, b]
            push!(ho.results, a + b)
        end
        @test length(ho.history) == length(ho.results) == 3
        @test ho.results == sum.(ho.history)

        # A second iteration after all results were recorded continues the optimization
        for (i, a, b) in ho
            push!(ho.results, a + b)
        end
        @test length(ho.history) == length(ho.results) == 6
        @test ho.results == sum.(ho.history)

        # Results of collected samples can be appended after the iteration
        ho = Hyperoptimizer(4, a = LinRange(1, 2, 5), b = [true, false])
        samples = collect(ho)
        append!(ho.results, [s.a + s.b for s in samples])
        @test ho.results == sum.(ho.history)

        # After an error or an interruption of the objective, history and results have equal lengths
        ho = Hyperoptimizer(iterations=5, params=(:a,), candidates=(LinRange(1, 2, 5),), objective=(i, a) -> i < 3 ? a : error("objective failure"))
        @test_throws ErrorException Hyperopt.optimize(ho)
        @test length(ho.history) == length(ho.results) == 2
        ho = Hyperoptimizer(iterations=5, params=(:a,), candidates=(LinRange(1, 2, 5),), objective=(i, a) -> i < 3 ? a : throw(InterruptException()))
        @test_logs (:info, "Aborting hyperoptimization") match_mode=:any Hyperopt.optimize(ho)
        @test length(ho.history) == length(ho.results) == 2

        # Parallel evaluation does not use the history appended by `iterate`
        horp = @thyperopt for i=20, a = LinRange(1, 2, 5), b = [true, false]
            a + b
        end
        @test length(horp.history) == length(horp.results) == 20
        @test horp.results == sum.(horp.history)
    end

    @testset "Typed fields" begin
        ho = @hyperopt for i=3, a = LinRange(1, 2, 5), b = [true, false]
            a + b
        end
        @test isconcretetype(fieldtype(typeof(ho), :params))
        @test isconcretetype(fieldtype(typeof(ho), :candidates))
        @test ho.params == (:a, :b)
    end

    @testset "RandomSampler seeding" begin
        ho = Hyperoptimizer(1, a = 1:10^6)
        draws(s) = [s(ho, k)[1] for k in 1:5]
        Random.seed!(1)
        d1 = draws(RandomSampler())
        Random.seed!(1)
        d2 = draws(RandomSampler())
        d3 = draws(RandomSampler())
        @test d1 == d2
        @test d1 != d3
        # More default samplers than the former number of distinct seeds yield distinct streams
        @test allunique([draws(RandomSampler()) for _ in 1:2000])
    end

    @testset "Plot recipe" begin
        ho = Hyperoptimizer(iterations=5, params=(:a, :g), candidates=(LinRange(1, 1000, 10), [sin, cos]),
            history=Any[[1.0, sin], [10.0, cos], [100.0, sin], [1000.0, cos], [500.0, sin]],
            results=Any[1.0, NaN, 1000.0, Inf])
        series = Hyperopt.RecipesBase.apply_recipe(Dict{Symbol, Any}(), ho)
        @test length(series) == 2
        @test all(s -> s.plotattributes[:yscale] == :log10, series)
        @test series[1].args == ([1.0, 100.0], [1.0, 1000.0])
        @test series[2].args[2] == [1.0, 1000.0]
        plot(ho)

        ho = Hyperoptimizer(iterations=2, params=(:a,), candidates=(LinRange(1, 2, 2),),
            history=Any[[1.0], [2.0]], results=Any[1.0, 2.0])
        series = Hyperopt.RecipesBase.apply_recipe(Dict{Symbol, Any}(), ho)
        @test series[1].plotattributes[:yscale] == :identity
    end
end
