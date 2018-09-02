using Test


using Hyperopt
f(x,a,b=true;c=10) = (sum(randn(1000000));sum(@. x + (a-3)^2 + (b ? 10 : 20) + (c-100)^2))

# res = map(1:50) do i
    # info("Iteration ", i)
    hor = @hyperopt for i=100, sampler=RandomSampler(), a = range(1,stop=5, length=50), b = [true, false], c = exp10.(range(-1,stop=3, length=50))
        # println(i, "\t", a, "\t", b, "\t", c)
        x = 100
        # print(i, " ")
        f(x,a,b,c=c)
    end
    @test minimum(hor)[2] < 300
    hot = @hyperopt for i=50, sampler=TreeSampler(random_init=5,samples_per_leaf=3,n_tries=20), a = range(1,stop=5, length=50), b = [true, false], c = exp10.(range(-1,stop=3, length=50))
        # println(i, "\t", a, "\t", b, "\t", c)
        x = 100
        # print(i, " ")
        f(x,a,b,c=c)
    end
    minimum(hot)


    hof = @hyperopt for i=50, sampler=ForestSampler(), a = range(1,stop=5, length=50), b = [true, false], c = exp10.(range(-1,stop=3, length=50))
        # println(i, "\t", a, "\t", b, "\t", c)
        x = 100
        # print(i, " ")
        f(x,a,b,c=c)
    end
    minimum(hof)

    # minimum.((hor,hot,hof))
# end
#
# hor,hot,hof = getindex.(res,1),getindex.(res,2),getindex.(res,3)
# hor,hot,hof = getindex.(hor,2),getindex.(hot,2),getindex.(hof,2)
# res = [hor hot hof]
# sum(log.(res), 1)

function inner(;random_init=nothing,
    min_samples_leaf=nothing,
    n_tries=nothing,
    n_subfeatures=nothing,
    partial_sampling=nothing,
    n_trees=nothing)
    ho = @hyperopt for i=50, sampler=Hyperopt.ForestSampler(random_init=random_init,
        min_samples_leaf=min_samples_leaf,
        n_tries=n_tries,
        n_subfeatures=n_subfeatures,
        partial_sampling=partial_sampling,
        n_trees=n_trees), a = range(1,stop=5, length=50), b = [true, false], c = exp10.(range(-1,stop=3, length=50))
        # println(i, "\t", a, "\t", b, "\t", c)
        x = 100
        print(i, " ")
        f(x,a,b,c=c)
    end
    minimum(ho)[2]
end
ho2 = @hyperopt for i=100, random_init = 2:10,
    min_samples_leaf = 2:10,
    n_tries = 5:30,
    n_subfeatures = 2:3,
    partial_sampling = range(0.2,stop=1,length=100),
    n_trees = 2:15
    inner(;random_init=random_init,
    min_samples_leaf=min_samples_leaf,
    n_tries=n_tries,
    n_subfeatures=n_subfeatures,
    partial_sampling=partial_sampling,
    n_trees=n_trees)

end
ho = Hyperoptimizer(10, a = range(1, stop=2, length=50), b = [true, false], c = randn(100))
for (i,a,b,c) in ho
    println(i, "\t", a, "\t", b, "\t", c)
end



ho = @hyperopt for i=100, a = range(1,stop=5, length=50), b = [true, false], c = exp10.(range(-1,stop=3, length=50)), d = [tanh, exp]
    # println(i, "\t", a, "\t", b, "\t", c)
    x = 100
    # print(i, " ")
    f(x,a,b,c=c) + d(a)
end
