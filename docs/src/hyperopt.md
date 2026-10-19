# The `@hyperopt` macro

```@setup macro
using Hyperopt, Random
Hyperopt.SHOW_PROGRESS[] = false
Random.seed!(0)
```

## Syntax

```julia
ho = @hyperopt for i = n,
                   sampler = RandomSampler(),
                   ho = existing_ho,
                   a = candidates_a,
                   b = candidates_b
    loop body
end
```

- The first argument of the for loop, `i = n`, determines the number of iterations `n`, i.e., the number of configurations that are drawn and evaluated. In the loop body, `i` holds the index of the current iteration. With a [`Hyperband`](@ref) sampler, `n` is instead the maximum budget, and `i` holds the budget of the current evaluation, see [Hyperband](hyperband.md).
- The optional argument `sampler` selects the sampler that draws the configurations. The default is [`RandomSampler()`](@ref RandomSampler), see [Samplers](samplers.md).
- The optional argument `ho` continues the optimization of an existing [`Hyperoptimizer`](@ref), see [Continuing an optimization](#Continuing-an-optimization).
- The remaining arguments name the hyperparameters and give their candidate vectors. In the loop body, the variables `a` and `b` hold the values of the hyperparameters of the current configuration. A logarithmically spaced candidate vector, e.g., `exp10.(LinRange(-3, 1, 100))`, results in log-uniform sampling.

The value of the loop body is the result of the evaluation, typically a real-valued cost that is to be minimized. The loop body is the body of a function, and the variables of the enclosing scope are accessible as in a closure. The variable `state` is defined in the loop body; it is `nothing` unless the sampler is a [`Hyperband`](@ref) sampler. See the docstring of [`@hyperopt`](@ref) for details.

## Example

The function `f` is minimized with respect to the hyperparameters `a`, `b` and `c`. The candidate vector of `c` is logarithmically spaced.
```@example macro
f(x, a, b = true; c = 10) = sum(@. x + (a - 3)^2 + (b ? 10 : 20) + (c - 100)^2)

ho = @hyperopt for i = 50,
                   sampler = RandomSampler(),
                   a = LinRange(1, 5, 1000),
                   b = [true, false],
                   c = exp10.(LinRange(-1, 3, 1000))
    x = 100
    f(x, a, b; c)
end
```

## Results

The [`Hyperoptimizer`](@ref) `ho` records the evaluated configurations in `ho.history` and the results in `ho.results`; `ho.results[k]` is the result of the configuration `ho.history[k]`:
```@example macro
ho.history[1], ho.results[1]
```
The properties `minimum`, `minimizer`, `maximum` and `maximizer` provide the optimal result and configuration:
```@example macro
ho.minimum, ho.minimizer
```
[`printmin`](@ref) and [`printmax`](@ref) print the optimal configuration together with the names of the hyperparameters:
```@example macro
printmin(ho)
```

A `NaN` result, e.g., of an evaluation that failed, is treated as `Inf` by `minimum` and `minimizer` and as `-Inf` by `maximum` and `maximizer`. It is therefore not selected unless all results are `NaN`:
```@example macro
ho_nan = @hyperopt for i = 20, a = LinRange(-1, 1, 21)
    a < 0 ? NaN : a^2
end
ho_nan.minimum, ho_nan.maximum
```
If no result has been recorded, `ho.minimum` and `ho.maximum` are `NaN`.

[`warn_on_boundary`](@ref) emits a warning for each hyperparameter whose optimal value is the smallest or the largest sampled value, which indicates that the candidate range is to be extended. In the following example, the cost is minimized by `a = 3`, which is outside of the candidate range:
```@example macro
ho_boundary = @hyperopt for i = 30, a = LinRange(0, 2, 100)
    (a - 3)^2
end
boundary_parameters = warn_on_boundary(ho_boundary)
nothing # hide
```
The names of these hyperparameters are returned:
```@example macro
boundary_parameters
```

## Continuing an optimization

The argument `ho = ho` continues the optimization of an existing `Hyperoptimizer`. The configurations and results of the new run are appended to `ho.history` and `ho.results`, and the sampler of `ho` is continued, including the state of its random number generator:
```@example macro
ho = @hyperopt for i = 20, ho = ho,
                   a = LinRange(1, 5, 1000),
                   b = [true, false],
                   c = exp10.(LinRange(-1, 3, 1000))
    f(100, a, b; c)
end
length(ho.history), length(ho.results)
```
The returned `Hyperoptimizer` shares `history` and `results` with the existing one. A sampler given in the continued run is not used; a warning is emitted if it is of another type than the sampler of the existing `Hyperoptimizer`.

## Categorical hyperparameters

The candidates of a hyperparameter can be values of any type, e.g., functions or symbols. [`RandomSampler`](@ref) draws them like numerical values:
```@example macro
ho_cat = @hyperopt for i = 30, fun = [tanh, sin, cos], x = LinRange(0, 1, 50)
    fun(x) + (x - 0.5)^2
end
printmin(ho_cat)
```
[`CLHSampler`](@ref) and [`BOHB`](@ref) require the categorical hyperparameters to be declared with [`Categorical`](@ref) or [`UnorderedCategorical`](@ref), see [Samplers](samplers.md) and [BOHB](bohb.md).

## Interruption and errors

If an evaluation throws an `InterruptException`, e.g., after Ctrl-C, the optimization is stopped and the `Hyperoptimizer` with the results obtained so far is returned. Other exceptions are rethrown. If the optimization of an existing `Hyperoptimizer` was continued, its history then contains only the configurations whose evaluation completed, so that `ho.history` and `ho.results` have equal lengths. An evaluation that is expected to fail for some configurations can return `NaN` instead, e.g., by
```julia
ho = @hyperopt for i = 100, a = LinRange(1, 5, 100)
    try
        train_model(a)
    catch
        NaN
    end
end
```

## Progress meters

`@hyperopt`, [`hyperband`](@ref) and [`hyperoptim`](@ref) display a progress meter. The progress meters are disabled by
```julia
Hyperopt.SHOW_PROGRESS[] = false
```

## Parallel evaluation

The macros [`@phyperopt`](@ref) and [`@thyperopt`](@ref) accept the same syntax as `@hyperopt` and evaluate the loop body in parallel, on the worker processes of the `Distributed` standard library and on threads, respectively. See their docstrings for details.
