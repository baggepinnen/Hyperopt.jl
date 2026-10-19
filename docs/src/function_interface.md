# Function interface

```@setup functions
using Hyperopt, Random
Hyperopt.SHOW_PROGRESS[] = false
Random.seed!(0)
```

The macro [`@hyperopt`](@ref) generates a function from the loop body and evaluates it. This page describes two interfaces without the macro: the manual iteration over a [`Hyperoptimizer`](@ref), and the function [`hyperband`](@ref).

## Manual iteration

A `Hyperoptimizer` is constructed with the number of iterations, optionally a sampler, and one keyword argument per hyperparameter, which gives its candidate vector. Iterating over the `Hyperoptimizer` yields one configuration per iteration, as a `NamedTuple` that contains the index of the iteration `i` and the values of the hyperparameters:
```@example functions
ho = Hyperoptimizer(10, a = LinRange(1, 2, 50), b = [true, false], c = randn(100))
for (i, a, b, c) in ho
    println(i, "\t", a, "\t", b, "\t", c)
end
```
Each iteration appends the configuration to `ho.history`. The results are not recorded by the iteration; to record them, the result of each configuration is appended to `ho.results`:
```@example functions
cost(a, b, c) = (a - 1.5)^2 + b + c^2

ho = Hyperoptimizer(10, a = LinRange(1, 2, 50), b = [true, false], c = randn(100))
for (i, a, b, c) in ho
    push!(ho.results, cost(a, b, c))
end
ho.minimum, ho.minimizer
```
`ho.history[k]` and `ho.results[k]` then refer to the same evaluation. When an iteration begins, the configurations at the end of `ho.history` without a corresponding result are removed, so that repeated iterations do not accumulate configurations that were never evaluated. An iteration that begins after the results of all configurations were recorded appends `ho.iterations` new configurations, which continues the optimization:
```@example functions
for (i, a, b, c) in ho
    push!(ho.results, cost(a, b, c))
end
length(ho.history), length(ho.results)
```
The configurations can also be collected first and evaluated later, e.g., by an external program:
```@example functions
ho = Hyperoptimizer(4, LHSampler(), a = LinRange(1, 2, 4), b = [true, false, true, false])
configurations = collect(ho)
```
```@example functions
append!(ho.results, [cost(x.a, x.b, 0.0) for x in configurations])
printmin(ho)
```
For [`LHSampler`](@ref) and [`CLHSampler`](@ref), an iteration that begins after results were recorded evaluates a new plan.

## Hyperband without the macro

[`hyperband`](@ref) runs Hyperband with an objective function `f(budget, x)` and returns the `Hyperoptimizer`. `x` is the vector of hyperparameter values in the first evaluation of a configuration, and the state returned by the previous evaluation of the same configuration otherwise. The objective returns `(loss, state)`, or only the loss, in which case `x` is the vector of hyperparameter values in every evaluation. The candidate vectors are given as a vector or as a `NamedTuple`; a `NamedTuple` names the hyperparameters, e.g., in plots.

In the following example, the state is a `NamedTuple` that contains the hyperparameters, the weights of the model and the number of epochs trained so far:
```@example functions
const scales = [1.0, 10.0, 100.0]
loss(w) = sum(scales .* (w .- 1) .^ 2)
gradient(w) = 2 .* scales .* (w .- 1)

function objective(epochs, x)
    s = x isa NamedTuple ? x : (lr = x[1], β = x[2], w = zeros(3), v = zeros(3), trained = 0)
    w, v = s.w, s.v
    for _ in 1:(round(Int, epochs) - s.trained)
        v = s.β * v - s.lr * gradient(w)
        w = w + v
    end
    loss(w), (; s.lr, s.β, w, v, trained = round(Int, epochs))
end

candidates = (lr = exp10.(LinRange(-4, -1.5, 100)), β = LinRange(0, 0.95, 100))
ho = hyperband(objective, candidates; R = 81, η = 3)
ho.minimum, ho.minimizer
```
As with the macro, `ho.sampler.minimum` contains the state of the best evaluation with the largest budget:
```@example functions
ho.sampler.minimum.state.w
```
The keyword argument `inner` selects the inner sampler, e.g., `inner = LHSampler()` or `inner = BOHB(dims = [Continuous(), Continuous()])`. See [`hyperoptim`](@ref) and [Optim](optim.md) for an application of `hyperband` to multi-start optimization.
