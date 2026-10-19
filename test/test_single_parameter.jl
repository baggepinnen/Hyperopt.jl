using Test, Hyperopt, Optim

# A global variable with the name of the hyperparameter, which must not be used in the loop body
a = 1000

single_parameter_objective(offset) = @hyperopt for i = 10, a = 1:10
    (a - offset)^2
end

@testset "Single hyperparameter" begin
    @info "Testing single hyperparameter"
    candidates = 1:10

    @testset "RandomSampler" begin
        ho = @hyperopt for i = 20, a = candidates
            a^2
        end
        @test length(ho.history) == length(ho.results) == 20
        @test all(h -> h[1] in candidates, ho.history)
        @test ho.results == [h[1]^2 for h in ho.history]
        io = IOBuffer()
        printmin(io, ho)
        @test String(take!(io)) == "a = $(ho.minimizer[1])\n"
    end

    @testset "LHSampler" begin
        ho = @hyperopt for i = 10, sampler = LHSampler(), a = LinRange(1, 5, 10)
            a^2
        end
        @test sort(first.(ho.history)) == LinRange(1, 5, 10)
        @test ho.results == [h[1]^2 for h in ho.history]
    end

    @testset "Inside a function" begin
        ho = single_parameter_objective(3)
        @test ho.results == [(h[1] - 3)^2 for h in ho.history]
    end

    @testset "thyperopt" begin
        ho = @thyperopt for i = 20, a = candidates
            a^2
        end
        @test length(ho.history) == length(ho.results) == 20
        @test all(h -> h[1] in candidates, ho.history)
        @test ho.results == [h[1]^2 for h in ho.history]
    end

    @testset "phyperopt" begin
        ho = @phyperopt for i = 20, a = candidates
            a^2
        end
        @test length(ho.history) == length(ho.results) == 20
        @test all(h -> h[1] in candidates, ho.history)
        @test ho.results == [h[1]^2 for h in ho.history]
    end

    @testset "Hyperband" begin
        # With R = 9 and η = 3, Hyperband evaluates 17 new configurations and resumes 5 of them
        candidates = LinRange(1, 5, 100)
        calls = []
        ho = @hyperopt for r = 1, sampler = Hyperband(R=9, η=3, inner=RandomSampler()), a = candidates
            push!(calls, (; r, a, state))
            if state !== nothing
                a, = state
            end
            (a - 3)^2 + 1/r, [a]
        end
        @test length(calls) == length(ho.history) == length(ho.results) == 22
        new = filter(c -> c.state === nothing, calls)
        resumed = filter(c -> c.state !== nothing, calls)
        @test length(new) == 17
        @test all(c -> c.a in candidates, calls)
        # A resumed evaluation receives the configuration it originates from and the state returned for it
        @test all(c -> c.state == [c.a], resumed)
        @test first.(ho.history) == [c.a for c in calls]
        @test ho.results ≈ [(c.a - 3)^2 + 1/c.r for c in calls]

        @hyperopt for r = 1, ho = ho, sampler = Hyperband(R=9, η=3, inner=RandomSampler()), a = candidates
            if state !== nothing
                a, = state
            end
            (a - 3)^2 + 1/r, [a]
        end
        @test length(ho.history) == length(ho.results) == 44

        ho = @hyperopt for r = 1, sampler = Hyperband(R=9, η=3, inner=LHSampler()), a = LinRange(1, 5, 9)
            if state !== nothing
                a, = state
            end
            (a - 3)^2 + 1/r, [a]
        end
        @test length(ho.results) == 22
        @test all(h -> h[1] in LinRange(1, 5, 9), ho.history)
    end

    @testset "Hyperband with BOHB" begin
        # With R = 50 and η = 3, Hyperband evaluates 49 new configurations and resumes 20 of them
        calls = []
        ho = @hyperopt for r = 1, sampler = Hyperband(R=50, η=3, inner=BOHB(dims=[Continuous()])), a = LinRange(1, 5, 100)
            push!(calls, (; r, a, state))
            if state !== nothing
                a, = state
            end
            (a - 3)^2 + 1/r, [a]
        end
        @test length(calls) == length(ho.results) == 69
        new = filter(c -> c.state === nothing, calls)
        resumed = filter(c -> c.state !== nothing, calls)
        # Configurations proposed by BOHB are delivered as new configurations
        @test length(new) == 49
        @test all(c -> c.a isa Real, new)
        @test all(c -> c.state == [c.a], resumed)
        @test ho.sampler.inner.KDE_good !== nothing
    end

    @testset "Function interface" begin
        for candidates in ([LinRange(1, 5, 100)], (a = LinRange(1, 5, 100),)), threads in (false, true)
            received = []
            lck = ReentrantLock()
            fun = function (r, pars)
                lock(() -> push!(received, pars), lck)
                (pars[1] - 3)^2 + 1/r, pars
            end
            ho = hyperband(fun, candidates; R=9, η=3, threads)
            @test length(received) == length(ho.results) == 22
            @test all(p -> p isa AbstractVector && length(p) == 1, received)
            @test ho.minimizer[1] in candidates[1]
        end
        ho = hyperoptim(x -> (x[1] - 3)^2, [LinRange(1, 5, 100)], Rmin=1, verbose=false, threads=false)
        @test ho.minimum < 1e-6
        @test ho.minimizer[1] ≈ 3 atol = 1e-2
    end

    @testset "Expressions are evaluated once" begin
        nevals = Ref(0)
        counted(x) = (nevals[] += 1; x)
        ho = @hyperopt for i = counted(5), a = counted(candidates)
            a
        end
        @test nevals[] == 2
        nevals[] = 0
        @hyperopt for i = counted(5), ho = counted(ho), a = counted(candidates)
            a
        end
        @test nevals[] == 3
        @test length(ho.results) == 10
        nevals[] = 0
        @hyperopt for r = counted(1), sampler = Hyperband(R=9, η=3), a = counted(LinRange(1, 5, 100))
            if state !== nothing
                a, = state
            end
            a, [a]
        end
        @test nevals[] == 2
    end

    @testset "The macro argument is not modified" begin
        ex = :(for i = 3, sampler = RandomSampler(), ho = nothing, a = 1:10
            a
        end)
        ex0 = deepcopy(ex)
        Hyperopt.preprocess_expression(ex)
        @test ex == ex0
    end
end
