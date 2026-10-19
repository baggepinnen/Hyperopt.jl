using Test, Random
using Hyperopt

# Each row of a plan is a permutation of 1:n for a continuous dimension and contains every level of a categorical
# dimension n ÷ levels or n ÷ levels + 1 times
function is_latin_hypercube(plan, dims)
    n = size(plan, 2)
    all(zip(eachrow(plan), dims)) do (row, dim)
        if dim isa Continuous
            sort(row) == 1:n
        else
            counts = [count(==(l), row) for l in 1:dim.levels]
            sum(counts) == n && all(c -> c in (n ÷ dim.levels, n ÷ dim.levels + 1), counts)
        end
    end
end

energy(plan, continuous) = sum(Hyperopt.pair_potential(Hyperopt.squared_distance(plan, continuous, a, b), size(plan, 2)) for a in axes(plan, 2) for b in a+1:size(plan, 2))

@testset "Latin hypercube samplers" begin
    @info "Testing Latin hypercube samplers"

    @testset "Latin hypercube property" begin
        for n in (1, 2, 7, 50), d in (1, 2, 4)
            ho = Hyperoptimizer(n, LHSampler(); (Symbol("x$j") => 2j .+ (1:n) for j in 1:d)...)
            @test size(ho.sampler.plan) == (d, n)
            @test is_latin_hypercube(ho.sampler.plan, fill(Continuous(), d))
            samples = collect(ho)
            for j in 1:d
                @test sort([s[j+1] for s in samples]) == 2j .+ (1:n)
            end
        end
        dims = [Continuous(), Categorical(3), UnorderedCategorical(4), Continuous(), Categorical(1)]
        for n in (1, 5, 12, 31)
            ho = Hyperoptimizer(n, CLHSampler(; dims); a = 1:n, b = [:x, :y, :z], c = [sin, cos, tan, exp], d = randn(n), e = [true])
            @test is_latin_hypercube(ho.sampler.plan, dims)
        end
        # Only categorical dimensions, with more points than distinct configurations
        ho = Hyperoptimizer(10, CLHSampler(dims=[Categorical(2), Categorical(2)]); a = [1, 2], b = [3, 4])
        @test is_latin_hypercube(ho.sampler.plan, [Categorical(2), Categorical(2)])
        @test length(unique(eachcol(ho.sampler.plan))) == 4
        @test Hyperopt.default_steps(20, 2) == 4000
        @test Hyperopt.default_steps(10^4, 100) == 10^6
    end

    @testset "Optimization of the plan" begin
        dims = [Categorical(3), Continuous(), Continuous()]
        continuous = [false, true, true]
        plan(ho, steps) = Hyperopt.init!(CLHSampler(; dims = ho.sampler.dims, steps, rng = Xoshiro(1)), ho).plan
        ho = Hyperoptimizer(30, CLHSampler(; dims); f = [sin, cos, tan], a = LinRange(0, 1, 30), b = LinRange(0, 1, 30))
        random, optimized = plan(ho, 0), plan(ho, nothing)
        @test is_latin_hypercube(random, dims)
        @test is_latin_hypercube(optimized, dims)
        @test energy(optimized, continuous) < 0.8energy(random, continuous)

        # The points that share a categorical candidate are distributed over the continuous dimension
        ho = Hyperoptimizer(30, CLHSampler(dims = [Categorical(3), Continuous()]); f = [sin, cos, tan], a = LinRange(0, 1, 30))
        optimized = plan(ho, nothing)
        for level in 1:3
            @test minimum(diff(sort(optimized[2, optimized[1, :] .== level]))) >= 2
        end
    end

    @testset "Reproducibility" begin
        run(sampler, c) = (@hyperopt for i = 20, sampler = sampler, a = 1:20, b = 1:20, c = c
            a + b
        end).history
        lh(seed) = run(LHSampler(rng = Xoshiro(seed)), 1:20)
        @test lh(1) == lh(1)
        @test lh(1) != lh(2)
        clh(seed) = run(CLHSampler(dims = [Continuous(), Continuous(), Categorical(2)], rng = Xoshiro(seed)), [:x, :y])
        @test clh(1) == clh(1)
        @test clh(1) != clh(2)
        # The default RNG is seeded from the global RNG
        runs() = run(LHSampler(), 1:20), run(CLHSampler(dims = [Continuous(), Continuous(), Categorical(2)]), [:x, :y])
        Random.seed!(3)
        h1 = runs()
        Random.seed!(3)
        @test runs() == h1
        @test runs() != h1
    end

    @testset "Spawned tasks" begin
        # The samplers have no thread-local state and can be used in tasks on any thread
        tasks = map(1:4Threads.nthreads()) do k
            Threads.@spawn begin
                hol = @hyperopt for i = 20, sampler = LHSampler(), a = LinRange(1, 5, 20), b = LinRange(1, 5, 20)
                    a + b
                end
                hocl = @hyperopt for i = 20, sampler = CLHSampler(dims = [Continuous(), UnorderedCategorical(2)]), a = LinRange(1, 5, 20), b = [true, false]
                    a + b
                end
                length(hol.results), length(hocl.results), Threads.threadid()
            end
        end
        results = fetch.(tasks)
        @test all(r -> r[1:2] == (20, 20), results)
        @info "Threads on which the Latin hypercube samplers were used" sort(unique(last.(results)))
    end

    @testset "Resumption" begin
        for seed in 1:10
            hol = @hyperopt for i = 20, sampler = LHSampler(rng = Xoshiro(seed)), a = LinRange(1, 5, 20), b = LinRange(1, 5, 20)
                a + b
            end
            @hyperopt for i = 20, ho = hol, sampler = LHSampler(), a = LinRange(1, 5, 20), b = LinRange(1, 5, 20)
                a + b
            end
            @test length(hol.history) == length(hol.results) == 40
            @test allunique(hol.history)
            for run in (1:20, 21:40)
                @test sort(first.(hol.history[run])) == LinRange(1, 5, 20)
                @test sort(last.(hol.history[run])) == LinRange(1, 5, 20)
            end
        end

        # Candidates with equal values, and a resumption without a given sampler
        hol = @hyperopt for i = 20, sampler = LHSampler(), a = LinRange(1, 5, 20), b = repeat([true, false], 10)
            a + b
        end
        sampler = hol.sampler
        @test_logs @hyperopt for i = 20, ho = hol, a = LinRange(1, 5, 20), b = repeat([true, false], 10)
            a + b
        end
        @test hol.sampler === sampler
        @test allunique(hol.history)
        @test hol.results == sum.(hol.history)

        hocl = @hyperopt for i = 12, sampler = CLHSampler(dims = [Continuous(), Categorical(2)]), a = LinRange(1, 5, 12), b = [true, false]
            a + b
        end
        @hyperopt for i = 12, ho = hocl, sampler = CLHSampler(dims = [Continuous(), Categorical(2)]), a = LinRange(1, 5, 12), b = [true, false]
            a + b
        end
        @test allunique(hocl.history)

        # With a single hyperparameter, every plan consists of all candidates
        ho1 = @hyperopt for i = 5, sampler = LHSampler(), a = 1:5
            a
        end
        @hyperopt for i = 5, ho = ho1, a = 1:5
            a
        end
        @test sort(first.(ho1.history)) == repeat(1:5, inner = 2)

        # Parallel evaluation
        hot = @thyperopt for i = 20, sampler = LHSampler(), a = LinRange(1, 5, 20), b = LinRange(1, 5, 20)
            a + b
        end
        @thyperopt for i = 20, ho = hot, sampler = LHSampler(), a = LinRange(1, 5, 20), b = LinRange(1, 5, 20)
            a + b
        end
        @test length(hot.history) == 40
        @test allunique(hot.history)
    end

    @testset "The sampler of a resumed Hyperoptimizer" begin
        hor = @hyperopt for i = 5, sampler = RandomSampler(), a = LinRange(1, 5, 5)
            a
        end
        sampler = hor.sampler
        @test_logs (:warn, r"RandomSampler, is continued; the given sampler, LHSampler, is ignored") @hyperopt for i = 5, ho = hor, sampler = LHSampler(), a = LinRange(1, 5, 5)
            a
        end
        @test hor.sampler === sampler
        @test length(hor.results) == 10
        @test_logs @hyperopt for i = 5, ho = hor, sampler = RandomSampler(MersenneTwister(0)), a = LinRange(1, 5, 5)
            a
        end
        @test length(hor.results) == 15
    end

    @testset "Manual iteration" begin
        ho = Hyperoptimizer(10, LHSampler(), a = LinRange(1, 2, 10), b = 1:10)
        # Repeated iteration without recorded results yields the same plan
        @test collect(ho) == collect(ho)
        for (i, a, b) in ho
            push!(ho.results, a + b)
        end
        @test length(ho.history) == length(ho.results) == 10
        @test sort(first.(ho.history)) == LinRange(1, 2, 10)
        # An iteration after results were recorded evaluates a new plan
        for (i, a, b) in ho
            push!(ho.results, a + b)
        end
        @test length(ho.history) == length(ho.results) == 20
        @test sort(last.(ho.history[11:20])) == 1:10
        @test allunique(ho.history)

        ho = Hyperoptimizer(6, CLHSampler(dims = [UnorderedCategorical(3), Continuous()]), f = [sin, cos, tan], a = 1:6)
        samples = collect(ho)
        @test sort([s.a for s in samples]) == 1:6
        @test all(f -> count(s -> s.f === f, samples) == 2, [sin, cos, tan])

        # A Hyperoptimizer constructed with keyword arguments generates the plan when it is first used
        ho = Hyperoptimizer(iterations = 4, params = (:a, :b), candidates = (1:4, 5:8), sampler = LHSampler())
        @test is_latin_hypercube(reduce(hcat, [[s.a, s.b - 4] for s in collect(ho)]), [Continuous(), Continuous()])
    end

    @testset "Categorical dimensions" begin
        ho = @hyperopt for i = 12, sampler = CLHSampler(dims = [UnorderedCategorical(3), Continuous()]), f = [sin, cos, tan], a = LinRange(0, 1, 12)
            f(a)
        end
        @test all(f -> count(h -> h[1] === f, ho.history) == 4, [sin, cos, tan])
        @test sort(last.(ho.history)) == LinRange(0, 1, 12)
        # Categorical and UnorderedCategorical are equivalent
        plan(dims) = Hyperopt.init!(CLHSampler(; dims, rng = Xoshiro(1)), ho).plan
        @test plan([UnorderedCategorical(3), Continuous()]) == plan([Categorical(3), Continuous()])
        @test UnorderedCategorical(3).levels == Categorical(3).levels == 3
        @test Hyperopt.DimensionType >: Union{Continuous, Categorical, UnorderedCategorical}
    end

    @testset "Argument checks" begin
        @test_throws ArgumentError Categorical(0)
        @test_throws ArgumentError UnorderedCategorical(-1)
        @test_throws ArgumentError LHSampler(steps = -1)
        @test_throws ArgumentError CLHSampler(dims = [Continuous(), 1])
        @test_throws UndefKeywordError CLHSampler()
        # The number of levels differs from the number of candidates
        @test_throws ArgumentError Hyperoptimizer(4, CLHSampler(dims = [Categorical(3), Continuous()]), f = [sin, cos], a = 1:4)
        # The number of dimension types differs from the number of hyperparameters
        @test_throws ArgumentError Hyperoptimizer(4, CLHSampler(dims = [Continuous()]), f = [sin, cos], a = 1:4)
        @test_throws ArgumentError Hyperoptimizer(4, CLHSampler(dims = [UnorderedCategorical(2), Continuous()]), f = [sin, cos], a = 1:5)
        @test_throws ArgumentError Hyperoptimizer(4, LHSampler(), a = 1:4, b = 1:3)
    end
end
