# Hyperopt

[![Build Status](https://travis-ci.org/baggepinnen/Hyperopt.jl.svg?branch=master)](https://travis-ci.org/baggepinnen/Hyperopt.jl)

[![Coverage Status](https://coveralls.io/repos/baggepinnen/Hyperopt.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/baggepinnen/Hyperopt.jl?branch=master)

[![codecov.io](http://codecov.io/github/baggepinnen/Hyperopt.jl/coverage.svg?branch=master)](http://codecov.io/github/baggepinnen/Hyperopt.jl?branch=master)

A package to perform hyperparameter optimization. Currently supports only random search.

# Usage

```julia
using Hyperopt

f(x,a,b=true;c=10) = sum(x + a + (b ? 10 : 20) + c) # Function to minimize

# Main macro. The first argument to the for loop is always interpreted as the number of iterations
julia> ho = @hyperopt for i=10, a = linspace(1,2,1000), b = [true, false], c = logspace(-1,3,1000)
           println(i, "\t", a, "\t", b, "\t", c)
           x = randn(100)
           f(x,a,b,c=c)
       end
1   1.3683683683683683  true    35.855398574598155
2   1.8568568568568569  false   24.343688735431105
3   1.2002002002002001  true    0.4060772025700365
4   1.967967967967968   false   12.767507043192657
5   1.2482482482482482  true    9.862658461312822
6   1.3823823823823824  false   685.2291595284064
7   1.7457457457457457  false   1.1403996019700327
8   1.864864864864865   false   29.817722900196717
9   1.2982982982982982  false   3.262222009711669
10  1.4504504504504505  false   0.10471768194855202
Hyperopt.Hyperoptimizer
  iterations: Int64 10
  params: Tuple{Symbol,Symbol,Symbol}
  candidates: Array{AbstractArray{T,1} where T}((3,))
  history: Array{Any}((10,))
  results: Array{Any}((10,))
  sampler: RandomSampler RandomSampler()


julia> best_params, min_f = minimum(ho)
(Real[1.2002, true, 0.406077], 1158.6489373344473)
```

The macro `@hyperopt` takes a for-loop with an initial argument determining the number of samples to draw (`i` above)
The subsequent arguments to the for-loop specifies names and candidate values for different hyper parameters (`a = linspace(1,2,1000), b = [true, false], c = logspace(-1,3,1000)` above). Currently uniform random sampling from the candidate values is the only supported optimizer. Log-uniform sampling is achieved with uniform sampling of a logarithmically spaced vector, e.g. `c = logspace(-1,3,1000)`. The parameters `i,a,b,c` can be used inside within the expression sent to the macro and they will hold a value sampled from the corresponding candidate vector each iteration.

The resulting object `ho::Hyperoptimizer` holds all the sampled parameters and function values and can be queried for `minimum/maximum` which returns the best parameters and function value found.

The type `Hyperoptimizer` is iterable, it iterates for the specified number of iterations, each iteration providing a sample of the parameter vector, e.g.
```julia
ho = Hyperoptimizer(10, a = linspace(1,2), b = [true, false], c = randn(100))
for (i,a,b,c) in ho
    println(i, "\t", a, "\t", b, "\t", c)
end

1   1.2244897959183674  false   0.8179751164732062
2   1.7142857142857142  true    0.6536272580487854
3   1.4285714285714286  true    -0.2737451706680355
4   1.6734693877551021  false   -0.12313108128547606
5   1.9795918367346939  false   -0.4350837079334295
6   1.0612244897959184  true    -0.2025613848798039
7   1.469387755102041   false   0.7464858339748051
8   1.8571428571428572  true    -0.9269021128132274
9   1.163265306122449   true    2.6554272337516966
10  1.4081632653061225  true    1.112896676939024
```

If uesd in this way, the hyperoptimizer **can not** keep track of the function values like it did when `@hyperopt` was used.
