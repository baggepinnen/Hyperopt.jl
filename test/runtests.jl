@static if VERSION < v"0.7.0-DEV.2005"
    using Base.Test
else
    using Test
end


using Hyperopt
f(x,a,b=true;c=10) = (sum(randn(1000000));sum(@. x + (a-3)^2 + (b ? 10 : 20) + (c-100)^2))

# res = map(1:50) do i
    info("Iteration ", i)
    hor = @hyperopt for i=50, sampler=RandomSampler(), a = linspace(1,5), b = [true, false], c = logspace(-1,3)
        # println(i, "\t", a, "\t", b, "\t", c)
        x = 100
        # print(i, " ")
        f(x,a,b,c=c)
    end
    minimum(hor)
    hot = @hyperopt for i=50, sampler=TreeSampler(random_init=5,samples_per_leaf=3,n_tries=20), a = linspace(1,5), b = [true, false], c = logspace(-1,3)
        # println(i, "\t", a, "\t", b, "\t", c)
        x = 100
        # print(i, " ")
        f(x,a,b,c=c)
    end
    minimum(hot)


    hof = @hyperopt for i=50, sampler=ForestSampler(), a = linspace(1,5), b = [true, false], c = logspace(-1,3)
        # println(i, "\t", a, "\t", b, "\t", c)
        x = 100
        # print(i, " ")
        f(x,a,b,c=c)
    end
    minimum(hof)

    minimum.((hor,hot,hof))
# end
#
# hor,hot,hof = getindex.(res,1),getindex.(res,2),getindex.(res,3)
# hor,hot,hof = getindex.(hor,2),getindex.(hot,2),getindex.(hof,2)
# res = [hor hot hof]
# sum(log.(res), 1)

function inner(;random_init=nothing,
    samples_per_leaf=nothing,
    n_tries=nothing,
    n_features=nothing,
    samples_per_tree=nothing,
    n_trees=nothing)
    ho = @hyperopt for i=50, sampler=Hyperopt.ForestSampler(random_init=random_init,
        samples_per_leaf=samples_per_leaf,
        n_tries=n_tries,
        n_features=n_features,
        samples_per_tree=samples_per_tree,
        n_trees=n_trees), a = linspace(1,5), b = [true, false], c = logspace(-1,3)
        # println(i, "\t", a, "\t", b, "\t", c)
        x = 100
        print(i, " ")
        f(x,a,b,c=c)
    end
    minimum(ho)[2]
end
ho2 = @hyperopt for i=100, random_init = 2:10,
    samples_per_leaf = 2:10,
    n_tries = 5:30,
    n_features = 2:3,
    samples_per_tree = linspace(0.2,1,100),
    n_trees = 2:15
    inner(;random_init=random_init,
    samples_per_leaf=samples_per_leaf,
    n_tries=n_tries,
    n_features=n_features,
    samples_per_tree=samples_per_tree,
    n_trees=n_trees)

end
ho = Hyperoptimizer(10, a = linspace(1,2), b = [true, false], c = randn(100))
for (i,a,b,c) in ho
    println(i, "\t", a, "\t", b, "\t", c)
end
