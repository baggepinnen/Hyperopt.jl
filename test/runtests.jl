using Test, Random, Distributed, Statistics
Random.seed!(0)
using Hyperopt, Plots
f(a,b=true;c=10) = sum(@. 100 + (a-3)^2 + (b ? 10 : 20) + (c-100)^2) # This function must be defined outside testsets to avoid scoping issues

@testset "Hyperopt" begin


    @testset "Random sampler" begin
        @info "Testing Random sampler"

        # res = map(1:30) do i
        #     @info("Iteration ", i)
        hor = @hyperopt for r=100, sampler=RandomSampler(), a = LinRange(1,5,50), b = [true, false], c = exp10.(LinRange(-1,3,50))
            # println(i, "\t", a, "\t", b, "\t", c)
            f(a,b,c=c)
        end
        @test minimum(hor) < 300
        @test maximum(hor) > 300
        @test length(hor.history) == 100
        @test length(hor.results) == 100
        @test all(hor.history) do h
            all(hi in hor.candidates[i] for (i,hi) in enumerate(h))
        end

        plot(hor)

        printmax(hor)
        printmin(hor)
        @test length(propertynames(hor)) > length(fieldnames(typeof(hor)))

        @hyperopt for i=2, ho=hor, sampler=RandomSampler(), a = LinRange(1,5,50), b = [true, false], c = exp10.(LinRange(-1,3,50))
            f(a,b,c=c)
        end
        @test length(hor.history) == 102
        @test length(hor.results) == 102

    end

    @testset "Latin hypercube" begin
        @info "Testing Latin hypercube"
        hol = @hyperopt for i=100, sampler=LHSampler(), a = LinRange(1,5,100), b = repeat([true, false],50), c = exp10.(LinRange(-1,3,100))
            # println(i, "\t", a, "\t", b, "\t", c)
            f(a,b,c=c)
        end
        @test minimum(hol) < 300
        plot(hol)

        @test length(hol.history) == 100
        @test length(hol.results) == 100
        @hyperopt for i=100, ho = hol, sampler=LHSampler(), a = LinRange(1,5,100), b = repeat([true, false],50), c = exp10.(LinRange(-1,3,100))
            # println(i, "\t", a, "\t", b, "\t", c)
            f(a,b,c=c)
        end
        @test length(hol.history) == 200
        @test length(hol.results) == 200


        # hocl = @hyperopt for i=100, sampler=CLHSampler(dims=[Hyperopt.ContinuousDim(),Hyperopt.CategoricalDim(2),Hyperopt.ContinuousDim()]), a = LinRange(1,5,100), b = [true, false], c = exp10.(LinRange(-1,3,100))
        hocl = @hyperopt for i=100, sampler=CLHSampler(dims=[Hyperopt.Continuous(),Hyperopt.Categorical(2),Hyperopt.Continuous()]), a = LinRange(1,5,100), b = [true, false], c = exp10.(LinRange(-1,3,100))
            # println(i, "\t", a, "\t", b, "\t", c)
            f(a,b,c=c)
        end
        @test minimum(hocl) < 300
        # @hyperopt for i=100, ho = hocl, sampler=CLHSampler(dims=[Hyperopt.ContinuousDim(),Hyperopt.CategoricalDim(2),Hyperopt.ContinuousDim()]), a = LinRange(1,5,100), b = [true, false], c = exp10.(LinRange(-1,3,100))
        @hyperopt for i=100, ho = hocl, sampler=CLHSampler(dims=[Hyperopt.Continuous(),Hyperopt.Categorical(2),Hyperopt.Continuous()]), a = LinRange(1,5,100), b = [true, false], c = exp10.(LinRange(-1,3,100))
            # println(i, "\t", a, "\t", b, "\t", c)
            f(a,b,c=c)
        end
        @test length(hocl.history) == 200
        @test length(hocl.results) == 200

    end

    @testset "Error handling" begin
        @info "Testing Error handling"


        @test_throws ArgumentError @hyperopt for i=100, sampler=LHSampler(), a = LinRange(1,5,10), b = repeat([true, false],50), c = exp10.(LinRange(-1,3,100))
            # println(i, "\t", a, "\t", b, "\t", c)
            f(a,b,c=c)
        end

        # @test_throws ArgumentError @hyperopt for i=100, sampler=CLHSampler(dims=[Hyperopt.ContinuousDim(),Hyperopt.CategoricalDim(2),Hyperopt.ContinuousDim()]), a = LinRange(1,5,99), b = [true, false], c = exp10.(LinRange(-1,3,100))
        @test_throws ArgumentError @hyperopt for i=100, sampler=CLHSampler(dims=[Hyperopt.Continuous(),Hyperopt.CategoricalDim(2),Hyperopt.Continuous()]), a = LinRange(1,5,99), b = [true, false], c = exp10.(LinRange(-1,3,100))
            # println(i, "\t", a, "\t", b, "\t", c)
            f(a,b,c=c)
        end
    end

    #     minimum.((hor,hob,hot,hof))
    # end

    # hor,hob,hot,hof = getindex.(res,1),getindex.(res,2),getindex.(res,3),getindex.(res,4)
    # hor,hob,hot,hof = getindex.(hor,2),getindex.(hob,2),getindex.(hot,2),getindex.(hof,2)
    # res = [hor hob hot hof]
    # sum(log.(res), dims=1)
    @testset "Manual" begin
        @info "Testing Manual"

        ho = Hyperoptimizer(10, a = range(1, stop=2, length=50), b = [true, false], c = randn(100))
        for (i,a,b,c) in ho
            println(i, "\t", a, "\t", b, "\t", c)
        end
        @test length(ho) == 10

        ho = Hyperoptimizer(10, a = range(1, stop=2, length=50), b = [true, false], c = randn(100))
        for vals in ho
            println(vals.i, "\t", vals.a, "\t", vals.b, "\t", vals.c)
        end

        @test length(collect(ho) ) == 10
        @test length([a for a ∈ ho]) == 10

    end



    @testset "Categorical" begin
        @info "Testing Categorical"

        ho = @hyperopt for i=100, a = LinRange(1,5,50), b = [true, false], c = exp10.(LinRange(-1,3,50)), d = [tanh, exp]
            # println(i, "\t", a, "\t", b, "\t", c)
            f(a,b,c=c) + d(a)
        end
    end

    @testset "Hyperband" begin
        using Optim
        f(a;c=10) = sum(@. 100 + (a-3)^2 + (c-100)^2)
        # res = map(1:30) do i
        #     @info("Iteration ", i)
        @test_nowarn Hyperband(50)
        let hohb = @hyperopt for r=18, sampler=Hyperband(R=50, η=3, inner=RandomSampler()), a = LinRange(1,5,800), c = exp10.(LinRange(-1,3,1800))
                # println(i, "\t", a, "\t", b, "\t", c)
                if !(state === nothing)
                    a,c = state
                end
                res = Optim.optimize(x->f(x[1],c=x[2]), [a,c], NelderMead(), Optim.Options(f_calls_limit=floor(Int, r)))
                Optim.minimum(res), Optim.minimizer(res)
            end
            @test length(hohb.history) == 69
            @test length(hohb.results) == 69
            @test minimum(hohb) < 300
            @hyperopt for i=1, ho=hohb, sampler=Hyperband(R=50, η=3, inner=RandomSampler()), a = LinRange(1,5,800), c = exp10.(LinRange(-1,3,1800))
                # println(i, "\t", a, "\t", b, "\t", c)
                if !(state === nothing)
                    a,c = state
                end
                res = Optim.optimize(x->f(x[1],c=x[2]), [a,c], NelderMead(), Optim.Options(f_calls_limit=floor(Int, i)))
                Optim.minimum(res), Optim.minimizer(res)
            end
            @test length(hohb.history) == 138
            @test length(hohb.results) == 138
        end
        # Special logic for the LHsampler as inner
        let hohb = @hyperopt for i=18, sampler=Hyperband(R=30, η=10, inner=LHSampler()), a = LinRange(1,5,300), c = exp10.(LinRange(-1,3,300))
                # println(i, "\t", a, "\t", b, "\t", c)
                if !(state === nothing)
                    a,c = state
                end
                res = Optim.optimize(x->f(x[1],c=x[2]), [a,c], NelderMead(), Optim.Options(f_calls_limit=floor(Int, 100i)))
                Optim.minimum(res), Optim.minimizer(res)
            end
            @test minimum(hohb) < 300
            @test length(hohb.history) == 13
            @test length(hohb.results) == 13
            @hyperopt for i=18, ho=hohb, sampler=Hyperband(R=30, η=10, inner=LHSampler()), a = LinRange(1,5,300), c = exp10.(LinRange(-1,3,300))
                # println(i, "\t", a, "\t", b, "\t", c)
                if !(state === nothing)
                    a,c = state
                end
                res = Optim.optimize(x->f(x[1],c=x[2]), [a,c], NelderMead(), Optim.Options(f_calls_limit=floor(Int, 100i)))
                Optim.minimum(res), Optim.minimizer(res)
            end
            @test length(hohb.history) == 26
            @test length(hohb.results) == 26
        end

        # extra robust option

        hohb = @hyperopt for i=18, sampler=Hyperband(R=50, η=3, inner=RandomSampler()),
            algorithm = [SimulatedAnnealing(), ParticleSwarm(), NelderMead(), BFGS(), NewtonTrustRegion()],
            a = LinRange(1,5,1800),
            c = exp10.(LinRange(-1,3,1800))
            if !(state === nothing)
                a,c,algorithm = state
            end
            println(i, " algorithm: ", typeof(algorithm))
            res = Optim.optimize(x->f(x[1],c=x[2]), [a,c], algorithm, Optim.Options(time_limit=2i+2, show_trace=false, show_every=5))
            Optim.minimum(res), (Optim.minimizer(res)..., algorithm)
        end

        # Function/vector/NamedTuple interface
        fun = function (i, pars)
            res = Optim.optimize(x->f(x[1],c=x[2]), pars, SimulatedAnnealing(), Optim.Options(time_limit=i/100, show_trace=false, show_every=5))
            Optim.minimum(res), Optim.minimizer(res)
        end
        candidates = [LinRange(1,5,300), exp10.(LinRange(-1,3,300))]
        hohb = Hyperopt.hyperband(fun, candidates; R=50)
        @test hohb.minimum < 100.1
        @test hohb.minimizer ≈ [3, 100] rtol = 1e-2

        candidates = (a=LinRange(1,5,300), c=exp10.(LinRange(-1,3,300))) # NamedTuple should also work
        hohb = Hyperopt.hyperband(fun, candidates; R=50)
        @test hohb.minimum < 100.1
        @test hohb.minimizer ≈ [3, 100] rtol = 1e-2
        @test hohb.params == [:a, :c]

        # test that hyperband can be placed inside a function
        function run_hyperband()
            @hyperopt for i=1, sampler=Hyperband(R=50, η=3, inner=RandomSampler()), a = LinRange(1,5,800), c = exp10.(LinRange(-1,3,1800))
                if state !== nothing
                    a,c = state
                end
                res = Optim.optimize(x->f(x[1],c=x[2]), [a,c], NelderMead(), Optim.Options(f_calls_limit=floor(Int, i)))
                Optim.minimum(res), Optim.minimizer(res)
            end
        end
        run_hyperband()

    end

    @testset "Parallel" begin
        @info "Testing Parallel"

        rmprocs(workers())


        horp = @thyperopt for i=300, sampler=RandomSampler(), a = LinRange(1,5,50), b = [true, false], c = exp10.(LinRange(-1,3,50))
            # println(i, "\t", a, "\t", b, "\t", c)
            f(a,b,c=c)
        end
        @test minimum(horp) < 300
        @test length(horp.history) == 300
        @test length(horp.results) == 300
        @test all(1:300) do i
            f(horp.history[i][1:2]..., c=horp.history[i][3]) == horp.results[i]
        end


        horp = @phyperopt for i=300, sampler=RandomSampler(), a = LinRange(1,5,50), b = [true, false], c = exp10.(LinRange(-1,3,50))
            # println(i, "\t", a, "\t", b, "\t", c)
            f(a,b,c=c)
        end
        @test minimum(horp) < 300
        @test length(horp.history) == 300
        @test length(horp.results) == 300
        @test all(1:300) do i
            f(horp.history[i][1:2]..., c=horp.history[i][3]) == horp.results[i]
        end

        Distributed.nworkers() ≤ 1 && addprocs(2)
        @everywhere using Hyperopt
        @everywhere f(a,b=true;c=10) = sum(@. 100 + (a-3)^2 + (b ? 10 : 20) + (c-100)^2)
        horp = @phyperopt for i=300, sampler=RandomSampler(), a = LinRange(1,5,50), b = [true, false], c = exp10.(LinRange(-1,3,50))
            # println(i, "\t", a, "\t", b, "\t", c)
            f(a,b,c=c)
        end
        @test minimum(horp) < 300
        @test length(horp.history) == 300
        @test length(horp.results) == 300

        @phyperopt for i=100, ho=horp, sampler=RandomSampler(), a = LinRange(1,5,50), b = [true, false], c = exp10.(LinRange(-1,3,50))
            # println(i, "\t", a, "\t", b, "\t", c)
            f(a,b,c=c)
        end
        @test length(horp.history) == 400
        @test length(horp.results) == 400

        ## test that history and results are in correct order
        @test all(1:400) do i
            f(horp.history[i][1:2]..., c=horp.history[i][3]) == horp.results[i]
        end


    end
    @testset "BOHB" begin
        @info "Testing BOHB"
        include("test_BOHB.jl")
    end
end
