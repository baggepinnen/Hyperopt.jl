using Test, Random
using Hyperopt, Plots

# A state that does not have the layout of the hyperparameters, e.g., the weights of a trained model
mutable struct TrainedModel
    configuration::Vector{Any}
    w::Vector{Float64}
    trained_with::Float64
end

@testset "Hyperband interface" begin
    @info "Testing Hyperband interface"

    @testset "The history records the configurations" begin
        candidates = (algorithm = [:A, :B], a = LinRange(1, 5, 100), c = LinRange(1, 5, 100))
        loss(algorithm, a, c) = (a - 3)^2 + (c - 3)^2 + (algorithm === :A ? 0.0 : 1.0)
        calls = []
        ho = @hyperopt for resources = 27, sampler = Hyperband(η = 3), algorithm = candidates.algorithm, a = candidates.a, c = candidates.c
            model = state === nothing ? TrainedModel([algorithm, a, c], [a, c], 0) : state
            push!(calls, (; resources, configuration = [algorithm, a, c], state))
            model.trained_with += resources
            loss(algorithm, a, c) + 1/resources, model
        end
        budgets = ho.sampler.budgets
        @test length(calls) == length(ho.history) == length(ho.results) == length(budgets) == 69
        @test ho.history == [c.configuration for c in calls]
        @test budgets == [c.resources for c in calls]
        @test all(h -> length(h) == 3 && all(h[i] in candidates[i] for i in 1:3), ho.history)
        @test ho.results ≈ [loss(h...) + 1/b for (h, b) in zip(ho.history, budgets)]
        # A resumed evaluation receives the state returned by the previous evaluation of the same configuration
        resumed = filter(c -> c.state !== nothing, calls)
        @test length(resumed) == 69 - 49
        @test all(c -> c.state.configuration == c.configuration, resumed)

        largest = findall(==(27), budgets)
        k = largest[argmin(ho.results[largest])]
        @test ho.minimizer == ho.history[k]
        @test ho.minimum == ho.results[k]
        best = ho.sampler.minimum
        @test best.loss == ho.minimum
        @test best.budget == 27
        @test best.configuration == ho.minimizer
        @test best.state isa TrainedModel
        @test best.state.configuration == ho.minimizer
        @test best.state.trained_with >= 27

        io = IOBuffer()
        printmin(io, ho)
        @test String(take!(io)) == join(("$p = $v\n" for (p, v) in zip(ho.params, ho.minimizer)))
        @test warn_on_boundary(ho) isa Vector{Symbol}
        @test occursin("minimizer", sprint(show, ho))

        series = Hyperopt.RecipesBase.apply_recipe(Dict{Symbol, Any}(), ho)
        @test length(series) == 3 * 4
        @test [s.plotattributes[:label] for s in series[1:4]] == ["budget = 1.0", "budget = 3.0", "budget = 9.0", "budget = 27.0"]
        @test sum(s -> length(s.args[1]), series[5:8]) == 69
        plot(ho)
    end

    @testset "Results of different budgets" begin
        # The loss increases with the budget, the smallest losses are thus obtained with the smallest budget
        calls = []
        ho = @hyperopt for r = 50, sampler = Hyperband(η = 3), a = LinRange(1, 5, 100)
            push!(calls, state)
            (a - 3)^2 + r/10
        end
        budgets = ho.sampler.budgets
        @test ho.sampler.R == 50
        @test sort(unique(budgets)) == [50/27, 50/9, 50/3, 50]
        largest = findall(==(50), budgets)
        @test length(largest) == 8
        @test ho.minimum == minimum(ho.results[largest]) > minimum(ho.results)
        @test ho.maximum == maximum(ho.results[largest])
        @test ho.minimizer == ho.history[largest[argmin(ho.results[largest])]]
        @test ho.maximizer == ho.history[largest[argmax(ho.results[largest])]]
        @test ho.sampler.minimum == (loss = ho.minimum, budget = 50.0, state = nothing, configuration = ho.minimizer)
        # An objective that returns only the loss is evaluated without a state
        @test all(isnothing, calls)

        # A continued optimization compares the results of both runs
        @hyperopt for r = 50, ho = ho, sampler = Hyperband(η = 3), a = LinRange(1, 5, 100)
            (a - 3)^2 + r/10
        end
        budgets = ho.sampler.budgets
        @test length(budgets) == length(ho.history) == length(ho.results) == 138
        largest = findall(==(50), budgets)
        @test length(largest) == 16
        @test ho.minimum == minimum(ho.results[largest])
        @test ho.sampler.minimum.loss == ho.minimum
        @test ho.sampler.minimum.configuration == ho.minimizer
    end

    @testset "The first loop argument is the maximum budget" begin
        ho = @hyperopt for r = 9, sampler = Hyperband(η = 3), a = LinRange(1, 5, 100)
            (a - 3)^2, [a]
        end
        @test ho.sampler.R == 9
        @test maximum(ho.sampler.budgets) == 9
        @test length(ho.results) == 22
        @test ho.iterations == length(ho) == 17

        ho = @test_logs (:warn, r"The value 18 differs from R = 9") match_mode=:any @hyperopt for r = 18, sampler = Hyperband(R = 9), a = LinRange(1, 5, 100)
            (a - 3)^2, [a]
        end
        @test ho.sampler.R == 9
        @test maximum(ho.sampler.budgets) == 9

        ho = @test_logs @hyperopt for r = 9, sampler = Hyperband(R = 9), a = LinRange(1, 5, 100)
            (a - 3)^2, [a]
        end
        @test length(ho.results) == 22

        ho = Hyperoptimizer(iterations = 1, params = (:a,), candidates = (1:3,), sampler = Hyperband(), objective = (r, a; state = nothing) -> a)
        @test_throws ArgumentError Hyperopt.hyperband(ho)
        @test_throws ArgumentError Hyperopt.brackets(9, 1)
    end

    @testset "Return values of the objective" begin
        candidates = LinRange(1, 5, 100)
        @test_throws ArgumentError @hyperopt for r = 9, sampler = Hyperband(η = 3), a = candidates
            [a], (a - 3)^2 # A vector as the loss
        end
        @test_throws ArgumentError @hyperopt for r = 9, sampler = Hyperband(η = 3), a = candidates
            (a - 3)^2, [a], r
        end
        msg = try
            hyperband((r, pars) -> (pars, pars), [candidates]; R = 9)
        catch e
            sprint(showerror, e)
        end
        @test occursin("must return a real-valued loss, or a tuple `(loss, state)`", msg)

        # Losses that are NaN or infinite, with the progress display enabled
        show_progress = Hyperopt.SHOW_PROGRESS[]
        Hyperopt.SHOW_PROGRESS[] = true
        try
            for value in (NaN, Inf)
                ho = @hyperopt for r = 9, sampler = Hyperband(η = 3), a = candidates
                    value, nothing
                end
                @test length(ho.results) == 22
                @test ho.minimum == Inf
                @test ho.minimizer in ho.history
                @test isequal(ho.sampler.minimum.loss, value)
                ho = hyperband((r, pars) -> value, [candidates]; R = 9)
                @test length(ho.results) == 22
            end
        finally
            Hyperopt.SHOW_PROGRESS[] = show_progress
        end
    end

    @testset "Latin hypercube plan" begin
        candidates = (a = LinRange(1, 5, 100), c = exp10.(LinRange(-1, 1, 100)))
        objective(r, pars) = sum(abs2, pars .- [3, 1])
        # The configurations are distinct points of a plan, in which every candidate occurs in one point
        distinct_points(configurations) = all(j -> allunique(getindex.(configurations, j)), 1:2)
        ho = hyperband(objective, candidates; R = 27, inner = LHSampler())
        @test ho.sampler.nsampled == ho.iterations == 49
        @test length(ho.history) == 69
        configurations = unique(ho.history)
        @test length(configurations) == 49
        @test distinct_points(configurations)

        ho = @hyperopt for r = 27, sampler = Hyperband(η = 3, inner = LHSampler()), a = candidates.a, c = candidates.c
            sum(abs2, [a, c] .- [3, 1]), [a, c]
        end
        configurations = unique(ho.history)
        @test length(configurations) == 49
        @test distinct_points(configurations)
        @hyperopt for r = 27, ho = ho, sampler = Hyperband(η = 3, inner = LHSampler()), a = candidates.a, c = candidates.c
            sum(abs2, [a, c] .- [3, 1]), [a, c]
        end
        @test length(ho.history) == 138
        configurations = unique(ho.history[70:end])
        @test length(configurations) == 49
        @test distinct_points(configurations)

        # The plan must contain a point for each configuration
        msg = try
            @hyperopt for r = 27, sampler = Hyperband(η = 3, inner = LHSampler()), a = LinRange(1, 5, 20), c = LinRange(1, 5, 20)
                (a - 3)^2 + (c - 3)^2
            end
        catch e
            sprint(showerror, e)
        end
        @test occursin("20 points, but Hyperband with R = 27 and η = 3 draws 49 configurations", msg)
        @test_throws ArgumentError hyperband(objective, [LinRange(1, 5, 48), LinRange(1, 5, 48)]; R = 27, inner = LHSampler())

        # The size of the plan does not depend on the length of a categorical candidate vector
        ho = @hyperopt for r = 27, sampler = Hyperband(η = 3, inner = CLHSampler(dims = [Continuous(), Categorical(2)])), a = LinRange(1, 5, 49), b = [true, false]
            (a - 3)^2 + b
        end
        @test length(unique(ho.history)) == 49
        @test length(unique(first.(ho.history))) == 49
    end

    @testset "successive_halving" begin
        ho = Hyperoptimizer(iterations = 1, params = (:a,), candidates = (LinRange(1, 5, 100),), sampler = Hyperband(R = 9),
            objective = (r, a; state = nothing) -> ((a - 3)^2 + 1/r, [a]))
        best = Hyperopt.successive_halving(ho, 9)
        @test ho.sampler.budgets == [fill(1, 9); fill(3, 3); 9]
        @test length(ho.history) == length(ho.results) == 13
        @test best === ho.sampler.minimum
        @test best.budget == 9

        # The function that evaluates a rung captures the objective and the budget, but not the Hyperoptimizer
        mapped = []
        recording_map(f, xs...) = (push!(mapped, f); map(f, xs...))
        Hyperopt.successive_halving(ho, 9, 1, 2; mapfun = recording_map)
        @test length(mapped) == 3
        @test length(ho.results) == 26
        @test !any(f -> any(n -> getfield(f, n) isa Hyperoptimizer, fieldnames(typeof(f))), mapped)

        ho = Hyperoptimizer(iterations = 1, params = (:a,), candidates = (LinRange(1, 5, 100),), sampler = Hyperband(R = 9),
            objective = (r, a; state = nothing) -> (a - 3)^2)
        empty!(mapped)
        Hyperopt.hyperband(ho; mapfun = recording_map)
        @test length(mapped) == 3 + 2 + 1
        @test length(ho.results) == 22
    end

    @testset "BOHB with a state that is not a configuration" begin
        Random.seed!(3)
        candidates = (a = LinRange(1, 5, 100), c = exp10.(LinRange(0, 3, 100)))
        ho = @hyperopt for r = 50, sampler = Hyperband(η = 3, inner = BOHB(dims = [Continuous(), Continuous()])), a = candidates.a, c = candidates.c
            model = state === nothing ? TrainedModel([a, c], [a, c], 0) : state
            model.trained_with += r
            (a - 3)^2 + log10(c / 100)^2, model
        end
        bohb = ho.sampler.inner
        @test bohb.KDE_good !== nothing
        @test sum(length, values(bohb.D)) == length(ho.history) == 69
        @test all(h -> h[1] in candidates.a && h[2] in candidates.c, ho.history)
        @test ho.sampler.minimum.state isa TrainedModel
        @test ho.sampler.minimum.state.configuration == ho.minimizer
    end

    @testset "A sampler is used by one Hyperoptimizer" begin
        sampler = Hyperband(R = 9)
        ho = @hyperopt for r = 9, sampler = sampler, a = LinRange(1, 5, 100)
            (a - 3)^2
        end
        @test length(sampler.budgets) == 22
        @test_throws ArgumentError @hyperopt for r = 9, sampler = sampler, a = LinRange(1, 5, 100)
            (a - 3)^2
        end
        @test length(ho.results) == length(sampler.budgets) == 22
    end

    @testset "Function interface" begin
        received = []
        ho = hyperband([LinRange(1, 5, 100), [:x, :y]]; R = 9) do r, pars
            push!(received, pars)
            (pars[1] - 3)^2 + (pars[2] === :x ? 0 : 1) + 1/r
        end
        # Without a returned state, every evaluation receives the configuration
        @test received == ho.history
        @test length(ho.results) == 22
        @test ho.minimizer in ho.history
        @test ho.sampler.minimum.budget == 9
    end
end
