using Hyperopt
@static if VERSION < v"0.7.0-DEV.2005"
    using Base.Test
else
    using Test
end

f(x,a,b=true;c=10) = sum(x + a + (b ? 10 : 20) + c)
ho = @hyperopt for i=10, a = linspace(1,2), b = [true, false], c = logspace(-1,3)
    println(i, "\t", a, "\t", b, "\t", c)
    x = randn(100)
    f(x,a,b,c=c)
end

minimum(ho)
ho = Hyperoptimizer(10, a = linspace(1,2), b = [true, false], c = randn(100))
for (i,a,b,c) in ho
    println(i, "\t", a, "\t", b, "\t", c)
end
