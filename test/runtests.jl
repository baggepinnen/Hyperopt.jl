@static if VERSION < v"0.7.0-DEV.2005"
    using Base.Test
else
    using Test
end


using Hyperopt
f(x,a,b=true;c=10) = sum(@. x + (a-3)^2 + (b ? 10 : 20) + (c-100)^2)
ho = @hyperopt for i=50, s=TreeSampler(random_init=5,n_samples=3,n_tries=20), a = linspace(1,5), b = [true, false], c = logspace(-1,3)
    # println(i, "\t", a, "\t", b, "\t", c)
    x = 100
    print(i, " ")
    @show f(x,a,b,c=c)
end

ho = @hyperopt for i=50, s=RandomSampler(), a = linspace(1,5), b = [true, false], c = logspace(-1,3)
    # println(i, "\t", a, "\t", b, "\t", c)
    x = 100
    print(i, " ")
    @show f(x,a,b,c=c)
end

minimum(ho)
ho = Hyperoptimizer(10, a = linspace(1,2), b = [true, false], c = randn(100))
for (i,a,b,c) in ho
    println(i, "\t", a, "\t", b, "\t", c)
end
