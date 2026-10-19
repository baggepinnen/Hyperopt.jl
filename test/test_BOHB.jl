using Test, Random, Statistics
Random.seed!(0)
using Hyperopt, Plots
using Optim

struct UnsupportedDimension <: Hyperopt.DimensionType end

@testset "BOHB" begin
    f(a;c=10) = sum(@. 100 + (a-3)^2 + (c-100)^2)
    @test_nowarn Hyperband(50)
    let bohb = @hyperopt for i=18, sampler=Hyperband(R=50, η=3, inner=BOHB(dims=[Continuous(), Continuous()])), a = LinRange(1,5,800), c = exp10.(LinRange(-1,3,1800))
            # println(i, "\t", a, "\t", b, "\t", c)
            if !(state === nothing)
                a,c = state
            end
            res = Optim.optimize(x->f(x[1],c=x[2]), [a,c], NelderMead(), Optim.Options(f_calls_limit=floor(Int, i)))
            Optim.minimum(res), Optim.minimizer(res)
        end
        @test length(bohb.history) == 69
        @test length(bohb.results) == 69
        @test minimum(bohb) < 300
        @hyperopt for i=1, ho=bohb, sampler=Hyperband(R=50, η=3, inner=BOHB(dims=[Continuous(), Continuous()])), a = LinRange(1,5,800), c = exp10.(LinRange(-1,3,1800))
            # println(i, "\t", a, "\t", b, "\t", c)
            if !(state === nothing)
                a,c = state
            end
            res = Optim.optimize(x->f(x[1],c=x[2]), [a,c], NelderMead(), Optim.Options(f_calls_limit=floor(Int, i)))
            Optim.minimum(res), Optim.minimizer(res)
        end
        @test length(bohb.history) == 138
        @test length(bohb.results) == 138
    end

    # extra robust option

    ho = @hyperopt for i=18, sampler=Hyperband(R=50, η=3, inner=BOHB(dims=[UnorderedCategorical(5), Continuous(), Continuous()])),
        algorithm = [SimulatedAnnealing(), ParticleSwarm(), NelderMead(), BFGS(), NewtonTrustRegion()],
        a = LinRange(1,5,1800),
        c = exp10.(LinRange(-1,3,1800))
        if !(state === nothing)
            algorithm, a, c = state # should come in same order as they appear in the argument list above.
        end
        println(i, " algorithm: ", typeof(algorithm))
        res = Optim.optimize(x->f(x[1],c=x[2]), [a,c], algorithm, Optim.Options(time_limit=2i+2, show_trace=false, show_every=5))
        Optim.minimum(res), (algorithm, Optim.minimizer(res)...)
    end
    @test all(h -> h[1] in ho.candidates[1], ho.history)

    @testset "Argument checks" begin
        @test_throws ArgumentError BOHB(dims=[Continuous(), UnsupportedDimension()])
        @test_throws ArgumentError BOHB(dims=[Continuous()], min_bandwidth=0)
        @test_throws ArgumentError BOHB(dims=[Continuous()], N_min=0)
        fun(r, pars) = (sum(abs2, pars .- 1), pars)
        @test_throws ArgumentError hyperband(fun, [1:10, 1:10]; R=9, inner=BOHB(dims=[Continuous()]))
        @test_throws ArgumentError hyperband(fun, [1:10, 1:10]; R=9, inner=BOHB(dims=[Continuous(), Categorical(3)]))
        @test_throws ArgumentError hyperband(fun, [1:10, [:a, :b]]; R=9, inner=BOHB(dims=[Continuous(), Continuous()]))
    end

    @testset "Split into good and poor observations" begin
        losses = [5.0, 1.0, 9.0, 3.0, 7.0, 2.0, 8.0, 4.0, 10.0, 6.0]
        good, bad = Hyperopt.good_bad_split(losses, 3, 0.15)
        @test good == [2, 6, 4]
        @test sort(bad) == [1, 3, 5, 7, 8, 9, 10]
        # With few observations, g(x) is fitted to the N_min worst observations
        good, bad = Hyperopt.good_bad_split(losses[1:5], 3, 0.15)
        @test good == [2, 4, 1]
        @test bad == [1, 5, 3]

        # The densities are fitted to the coordinates of the best and the worst observations
        bohb = BOHB(dims=[Continuous()])
        ho = Hyperoptimizer(iterations=1, params=[:a], candidates=[LinRange(0, 1, 11)], sampler=Hyperband(R=9, inner=bohb))
        xs = 0:0.1:0.9
        Hyperopt.update_observations(ho, 1.0, [[x] for x in xs], xs .^ 2)
        @test bohb.max_valid_budget == 1.0
        @test vec(bohb.KDE_good.data) ≈ [0.0, 0.1]
        @test sort(vec(bohb.KDE_bad.data)) ≈ 0.2:0.1:0.9
        bohb.ρ = 0
        proposals = [only(Hyperopt.kde_proposal(bohb, ho)) for _ in 1:100]
        @test all(in(ho.candidates[1]), proposals)
        @test mean(proposals) < 0.3
    end

    @testset "Proposals are candidate values" begin
        candidates = (
            a = LinRange(1, 5, 800),
            c = exp10.(LinRange(-1, 3, 300)),
            n = [1, 2, 4, 8, 16],
            b = [0.1, 0.5, 2.0],
            t = [true, false],
            g = [sin, cos, identity],
        )
        dims = [Continuous(), Continuous(), Continuous(), Categorical(3), Categorical(2), UnorderedCategorical(3)]
        bohb = BOHB(; dims, ρ=0, random_sampler=RandomSampler(MersenneTwister(0)))
        ho = Hyperoptimizer(iterations=1, params=collect(keys(candidates)), candidates=candidates, sampler=Hyperband(R=9, inner=bohb))
        @test Hyperopt.kde_proposal(bohb, ho) === nothing # No densities have been fitted
        rng = MersenneTwister(1)
        # Observations outside the candidate ranges are mapped to the boundary of the ranges
        observations = [[6rand(rng), exp10(5rand(rng) - 2), rand(rng, 0:20), rand(rng, candidates.b), rand(rng, candidates.t), rand(rng, candidates.g)] for _ in 1:30]
        Hyperopt.update_observations(ho, 1.0, observations, randn(rng, 30))
        @test bohb.KDE_good !== nothing
        proposals = [Hyperopt.kde_proposal(bohb, ho) for _ in 1:300]
        for (j, cands) in enumerate(candidates)
            @test all(p -> any(c -> c === p[j], cands), proposals)
        end
        @test length(unique(getindex.(proposals, 1))) > 10
        @test eltype(getindex.(proposals, 5)) == Bool
        # The observations are not changed by mapping them to coordinates and back
        maps = Hyperopt.coordinate_maps(bohb, ho)
        for x in ([candidates.a[17], candidates.c[123], 4, 2.0, false, cos], [first.(values(candidates))...], [last.(values(candidates))...])
            @test [Hyperopt.from_coordinate(m, Hyperopt.to_coordinate(m, v)) for (m, v) in zip(maps, x)] == x
        end
    end

    @testset "Proposals in the macro and the function interface" begin
        Random.seed!(2)
        ho = @hyperopt for i=18, sampler=Hyperband(R=50, η=3, inner=BOHB(dims=[Continuous(), Continuous()])),
                a = LinRange(1,5,800), c = exp10.(LinRange(1,3,1800))
            if state !== nothing
                a, c = state
            end
            (a-3)^2 + (c-100)^2/1e4, (a, c)
        end
        @test all(h -> h[1] in ho.candidates[1] && h[2] in ho.candidates[2], ho.history)
        # Budgets that differ only by rounding errors are merged
        @test length(ho.sampler.inner.D) == 4

        fun(r, pars) = (sum(abs2, (pars .- [3, 100]) ./ [1, 100]), pars)
        for candidates in ([LinRange(1,5,300), exp10.(LinRange(-1,3,300))], (a=LinRange(1,5,300), c=exp10.(LinRange(-1,3,300))))
            hohb = hyperband(fun, candidates; R=50, inner=BOHB(dims=[Continuous(), Continuous()]))
            @test length(hohb.history) == 69
            @test all(h -> h[1] in candidates[1] && h[2] in candidates[2], hohb.history)
            @test hohb.minimum < 1
        end
    end

    @testset "Reproducibility" begin
        fun(r, pars) = (sum(abs2, (pars .- [3, 100]) ./ [1, 100]), pars)
        candidates = (a=LinRange(1,5,300), c=exp10.(LinRange(-1,3,300)))
        run(seed) = hyperband(fun, candidates; R=27, inner=BOHB(dims=[Continuous(), Continuous()], random_sampler=RandomSampler(MersenneTwister(seed)))).history
        Random.seed!(1)
        h1 = run(3)
        Random.seed!(2)
        @test run(3) == h1
        @test run(4) != h1
    end

    @testset "Comparison with random sampling" begin
        candidates = (a = LinRange(-5, 5, 101), b = exp10.(LinRange(-4, 0, 101)), c = LinRange(0, 10, 101), d = [:x, :y, :z, :w])
        loss(p) = ((p[1] - 1) / 10)^2 + ((log10(p[2]) + 3) / 4)^2 + ((p[3] - 7) / 10)^2 + (p[4] === :z ? 0.0 : 0.1)
        fun(r, p) = (loss(p), p)
        dims = [Continuous(), Continuous(), Continuous(), UnorderedCategorical(4)]
        best_bohb, best_random, proposal_bohb, proposal_random = Float64[], Float64[], Float64[], Float64[]
        for seed in 1:40
            hob = hyperband(fun, candidates; R=81, inner=BOHB(; dims, random_sampler=RandomSampler(MersenneTwister(seed))))
            hor = hyperband(fun, candidates; R=81, inner=RandomSampler(MersenneTwister(seed)))
            @test length(hob.history) == length(hor.history)
            push!(best_bohb, minimum(loss, hob.history))
            push!(best_random, minimum(loss, hor.history))
            # Proposals of the fitted densities, compared with random samples
            bohb = hob.sampler.inner
            bohb.ρ = 0
            push!(proposal_bohb, mean(loss(Hyperopt.kde_proposal(bohb, hob)) for _ in 1:100))
            push!(proposal_random, mean(loss(bohb.random_sampler(hob, 1)) for _ in 1:100))
        end
        @info "Mean of the best loss over $(length(best_bohb)) runs" BOHB=mean(best_bohb) RandomSampler=mean(best_random) BOHB_better=count(best_bohb .< best_random) RandomSampler_better=count(best_bohb .> best_random)
        @test mean(best_bohb) < mean(best_random)
        @test count(best_bohb .< best_random) > count(best_bohb .> best_random)
        @test mean(proposal_bohb) < 0.5mean(proposal_random)
    end
end
