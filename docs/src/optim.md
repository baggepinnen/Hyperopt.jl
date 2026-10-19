# Optim

```@setup optim
using Hyperopt, Random
Hyperopt.SHOW_PROGRESS[] = false
Random.seed!(0)
```

Hyperband can be applied to multi-start optimization with [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl): the configurations are initial guesses, and the budget limits the computational effort of a local optimization. A configuration that is evaluated again with a larger budget continues the optimization from the minimizer found by its previous evaluation.

## `hyperoptim`

[`hyperoptim`](@ref) implements this approach with a time limit in seconds as the budget. It is defined in a package extension and requires Optim.jl to be loaded. The interface is similar to that of `Optim.optimize`, but the initial guess is replaced by candidate vectors, one per element of the initial guess. The following example minimizes the [Rastrigin function](https://en.wikipedia.org/wiki/Rastrigin_function), which has many local minima and its global minimum at the origin:
```@example optim
using Optim
rastrigin(x) = 10length(x) + sum(xi -> xi^2 - 10cos(2π * xi), x)

candidates = [LinRange(-5, 5, 100), LinRange(-5, 5, 100)]
ho = hyperoptim(rastrigin, candidates, NelderMead(); R = 9, η = 3)
ho.minimum
```
`ho.minimizer` is the initial guess from which the smallest value was found, and `ho.sampler.minimum.state` is the corresponding minimizer of the function:
```@example optim
ho.minimizer, ho.sampler.minimum.state
```
Most evaluations terminate before the time limit, since `Optim.optimize` stops when its convergence criteria are satisfied.

## `@hyperopt` with Optim

The budget can also be another quantity, e.g., the number of iterations. In the following example, the loop body is a local optimization with at most `iterations` iterations, and the hyperparameters are the initial guess `x1`, `x2` and the optimization algorithm. The state is the minimizer, from which a later evaluation continues the optimization:
```@example optim
ho = @hyperopt for iterations = 81,
                   sampler = Hyperband(η = 3),
                   algorithm = [NelderMead(), BFGS(), SimulatedAnnealing()],
                   x1 = LinRange(-5, 5, 100),
                   x2 = LinRange(-5, 5, 100)
    x0 = state === nothing ? [x1, x2] : state
    res = Optim.optimize(rastrigin, x0, algorithm, Optim.Options(iterations = round(Int, iterations)))
    Optim.minimum(res), Optim.minimizer(res)
end
algorithm, x1, x2 = ho.minimizer
nameof(typeof(algorithm)), x1, x2
```
The minimum and the corresponding minimizer of the function are
```@example optim
ho.minimum, ho.sampler.minimum.state
```
In every evaluation, the hyperparameter variables hold the configuration, i.e., the initial guess and the algorithm of the first evaluation, while `state` holds the minimizer of the previous evaluation.
