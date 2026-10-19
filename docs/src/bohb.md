# BOHB

```@setup bohb
using Hyperopt, Random
Hyperopt.SHOW_PROGRESS[] = false
Random.seed!(0)
```

[`BOHB`](@ref) implements the model-based sampling of [BOHB: Robust and Efficient Hyperparameter Optimization at Scale](https://arxiv.org/abs/1807.01774). It is only valid as the inner sampler of [`Hyperband`](@ref), which records its observations, and replaces the random sampling of the configurations: a fraction `ρ` of the configurations is drawn at random, and the remaining configurations are proposed by maximizing the ratio ``l(x)/g(x)`` of two kernel density estimates, fitted to the best and to the worst configurations evaluated with the largest budget for which sufficiently many observations are available. Before sufficiently many observations are available, all configurations are drawn at random.

## Dimension types

The keyword argument `dims` contains one dimension type per hyperparameter, in the order of the candidate vectors:
- [`Continuous()`](@ref Continuous) for ordered, real-valued hyperparameters, including integer-valued ones. The densities are estimated in the coordinate given by the index into the sorted candidate vector, scaled to ``[0, 1]``. A logarithmically spaced candidate vector, such as `exp10.(LinRange(-4, -1, 100))`, is thus modeled in a logarithmic coordinate.
- [`Categorical(k)`](@ref Categorical) or, equivalently, [`UnorderedCategorical(k)`](@ref UnorderedCategorical) for categorical hyperparameters with `k` candidates, e.g., `Categorical(2)` for `[false, true]`.

Every configuration proposed by BOHB consists of elements of the candidate vectors.

## Example

The example of the page [Hyperband](hyperband.md) is extended by the categorical hyperparameter `nesterov`, which selects Nesterov's accelerated gradient instead of gradient descent with momentum:
```@example bohb
const scales = [1.0, 10.0, 100.0]
loss(w) = sum(scales .* (w .- 1) .^ 2)
gradient(w) = 2 .* scales .* (w .- 1)

function train(w, v, lr, β, nesterov, epochs)
    for _ in 1:epochs
        g = nesterov ? gradient(w + β * v) : gradient(w)
        v = β * v - lr * g
        w = w + v
    end
    w, v
end

function evaluate(lr, β, nesterov, epochs, state)
    w, v, trained = state === nothing ? (zeros(3), zeros(3), 0) : state
    w, v = train(w, v, lr, β, nesterov, round(Int, epochs) - trained)
    loss(w), (w, v, round(Int, epochs))
end

ho = @hyperopt for epochs = 81,
                   sampler = Hyperband(η = 3, inner = BOHB(dims = [Continuous(), Continuous(), Categorical(2)])),
                   lr = exp10.(LinRange(-4, -1.5, 100)),
                   β = LinRange(0, 0.95, 100),
                   nesterov = [false, true]
    evaluate(lr, β, nesterov, epochs, state)
end
printmin(ho)
```
The configurations proposed by BOHB are evaluated as new configurations, i.e., `state` is `nothing` in their first evaluation. The observations of BOHB are the evaluated configurations and their losses; the states returned by the loop body are not used by BOHB. The results are recorded as for any other inner sampler of `Hyperband`, see [Hyperband](hyperband.md):
```@example bohb
ho.minimum, ho.sampler.minimum.budget
```

## Reproducibility and parameters

All random decisions of BOHB use the random number generator of its random sampler, a run is thus reproducible with, e.g., `BOHB(dims = [Continuous(), Continuous()], random_sampler = RandomSampler(Xoshiro(0)))`. The observations are retained when an optimization is continued with `ho = ...`. See the docstring of [`BOHB`](@ref) for the remaining parameters, which correspond to the symbols of the paper.
