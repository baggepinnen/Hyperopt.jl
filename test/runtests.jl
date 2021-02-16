using Test, Random, Distributed, Statistics
Random.seed!(0)
using Hyperopt, Plots
f(a,b=true;c=10) = sum(@. 100 + (a-3)^2 + (b ? 10 : 20) + (c-100)^2) # This function must be defined outside testsets to avoid scoping issues

@testset "Hyperopt" begin


    @testset "Random sampler" begin
        @info "Testing Random sampler"

        # res = map(1:30) do i
        #     @info("Iteration ", i)
        hor = @hyperopt for i=100, sampler=RandomSampler(), a = LinRange(1,5,50), b = [true, false], c = exp10.(LinRange(-1,3,50))
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

        @test length(hol.history) == 100
        @test length(hol.results) == 100
        @hyperopt for i=100, ho = hol, sampler=LHSampler(), a = LinRange(1,5,100), b = repeat([true, false],50), c = exp10.(LinRange(-1,3,100))
            # println(i, "\t", a, "\t", b, "\t", c)
            f(a,b,c=c)
        end
        @test length(hol.history) == 200
        @test length(hol.results) == 200


        hocl = @hyperopt for i=100, sampler=CLHSampler(dims=[Continuous(),Categorical(2),Continuous()]), a = LinRange(1,5,100), b = [true, false], c = exp10.(LinRange(-1,3,100))
            # println(i, "\t", a, "\t", b, "\t", c)
            f(a,b,c=c)
        end
        @test minimum(hocl) < 300
        @hyperopt for i=100, ho = hocl, sampler=CLHSampler(dims=[Continuous(),Categorical(2),Continuous()]), a = LinRange(1,5,100), b = [true, false], c = exp10.(LinRange(-1,3,100))
            # println(i, "\t", a, "\t", b, "\t", c)
            f(a,b,c=c)
        end
        @test length(hocl.history) == 200
        @test length(hocl.results) == 200

    end

    @testset "GP sampler" begin
        @info "Testing GP sampler"

        @test_throws ErrorException GPSampler()

        foreach(Hyperopt.HO_RNG) do rng
            Random.seed!(rng, 0)
        end

        results = map(1:20) do _
            hogp = @hyperopt for i=40, sampler=GPSampler(Min), a = LinRange(1,5,100), b = repeat([true, false]',50)[:], c = exp10.(LinRange(-1,3,100))
                # println(i, "\t", a, "\t", b, "\t", c)
                f(a,Bool(b),c=c)
            end
            minimum(hogp)
        end

        @show mean(results)
        @show mean(results .< 300)

        @test mean(results) < 300
        @test mean(results .< 300) >= 0.8


        hogp = @hyperopt for i=50, sampler=GPSampler(Min), a = LinRange(1,5,100), b = repeat([true, false]',50)[:], c = exp10.(LinRange(-1,3,100))
            f(a,Bool(b),c=c)
        end
        minimum(hogp)

        plot(hogp.sampler)
        plot(hogp)

        # One dimension case
        hogp = @hyperopt for i=50, sampler=GPSampler(Min), a = LinRange(1,5,100)
            a
        end
        plot(hogp.sampler)
        plot(hogp)
        @test length(hogp.history) == 50
        @test length(hogp.results) == 50

        @hyperopt for i=10, ho=hogp, sampler=GPSampler(Min), a = LinRange(1,5,100)
            a
        end
        @test length(hogp.history) == 60
        @test length(hogp.results) == 60
    end


    @testset "Error handling" begin
        @info "Testing Error handling"


        @test_throws ArgumentError @hyperopt for i=100, sampler=LHSampler(), a = LinRange(1,5,10), b = repeat([true, false],50), c = exp10.(LinRange(-1,3,100))
            # println(i, "\t", a, "\t", b, "\t", c)
            f(a,b,c=c)
        end

        @test_throws ArgumentError @hyperopt for i=100, sampler=CLHSampler(dims=[Continuous(),Categorical(2),Continuous()]), a = LinRange(1,5,99), b = [true, false], c = exp10.(LinRange(-1,3,100))
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

        @test_throws AssertionError begin
            ho = Hyperoptimizer(10, GPSampler(Min), a = range(1, stop=2, length=50), b = [true, false], c = randn(100))
            for (i,a,b,c) in ho
                println(i, "\t", a, "\t", b, "\t", c)
            end
        end

        ho = Hyperoptimizer(10, GPSampler(Min), a = range(1, stop=2, length=50), b = [true, false], c = randn(100))
        for (i,a,b,c) in ho
            println(i, "\t", a, "\t", b, "\t", c)
            push!(ho.results, randn())
        end


    end



    @testset "Categorical" begin
        @info "Testing Categorical"

        ho = @hyperopt for i=100, a = LinRange(1,5,50), b = [true, false], c = exp10.(LinRange(-1,3,50)), d = [tanh, exp]
            # println(i, "\t", a, "\t", b, "\t", c)
            f(a,b,c=c) + d(a)
        end
    end


    @testset "Utils" begin
        @test Hyperopt.islogspace(exp10.(LinRange(-3, 3, 10)))
        @test !Hyperopt.islogspace(LinRange(-3, 3, 10))
        @test !Hyperopt.islogspace([true, false])
    end


    @testset "Hyperband" begin
        using Optim
        f(a;c=10) = sum(@. 100 + (a-3)^2 + (c-100)^2)
        # res = map(1:30) do i
        #     @info("Iteration ", i)
        @test_nowarn Hyperband(50)
        let hohb = @hyperopt for i=18, sampler=Hyperband(R=50, η=3, inner=RandomSampler()), a = LinRange(1,5,800), c = exp10.(LinRange(-1,3,1800))
                # println(i, "\t", a, "\t", b, "\t", c)
                if !(state === nothing)
                    a,c = state
                end
                res = Optim.optimize(x->f(x[1],c=x[2]), [a,c], NelderMead(), Optim.Options(f_calls_limit=i))
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
                res = Optim.optimize(x->f(x[1],c=x[2]), [a,c], NelderMead(), Optim.Options(f_calls_limit=i))
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
                res = Optim.optimize(x->f(x[1],c=x[2]), [a,c], NelderMead(), Optim.Options(f_calls_limit=100i))
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
                res = Optim.optimize(x->f(x[1],c=x[2]), [a,c], NelderMead(), Optim.Options(f_calls_limit=100i))
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

    end

    @testset "Parallel" begin
        @info "Testing Parallel"

        horp = @phyperopt for i=300, sampler=RandomSampler(), a = LinRange(1,5,50), b = [true, false], c = exp10.(LinRange(-1,3,50))
            # println(i, "\t", a, "\t", b, "\t", c)
            f(a,b,c=c)
        end
        @test minimum(horp) < 300
        @test length(horp.history) == 300
        @test length(horp.results) == 300

        Distributed.nworkers() == 0 && addprocs(2)
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
    end

end
