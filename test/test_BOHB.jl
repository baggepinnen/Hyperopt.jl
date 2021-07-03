using Test, Random, Statistics
Random.seed!(0)
using Hyperopt, Plots
using Optim

@testset "BOHB" begin
    f(a;c=10) = sum(@. 100 + (a-3)^2 + (c-100)^2)
    @test_nowarn Hyperband(50)
    let bohb = @hyperopt for i=18, sampler=Hyperband(R=50, η=3, inner=BOHB(dims=[Continuous(), Continuous()])), a = LinRange(1,5,800), c = exp10.(LinRange(-1,3,1800))
            # println(i, "\t", a, "\t", b, "\t", c)
            if !(state === nothing)
                a,c = state
            end
            res = Optim.optimize(x->f(x[1],c=x[2]), [a,c], NelderMead(), Optim.Options(f_calls_limit=i))
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
            res = Optim.optimize(x->f(x[1],c=x[2]), [a,c], NelderMead(), Optim.Options(f_calls_limit=i))
            Optim.minimum(res), Optim.minimizer(res)
        end
        @test length(bohb.history) == 138
        @test length(bohb.results) == 138
    end

    # extra robust option
    @macroexpand @hyperopt for i=18, sampler=Hyperband(R=50, η=3, inner=BOHB(dims=[UnorderedCategorical(5), Continuous(), Continuous()])),
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
    

    ho = @hyperopt for i=18, sampler=Hyperband(R=50, η=3, inner=BOHB(dims=[UnorderedCategorical(5), Continuous(), Continuous()])),
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