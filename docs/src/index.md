# Hyperopt.jl

Hyperopt.jl is a package for hyperparameter optimization in Julia. It provides the following methods:

- Random search, [`RandomSampler`](@ref).
- [Latin hypercube sampling](https://en.wikipedia.org/wiki/Latin_hypercube_sampling) with continuous hyperparameters, [`LHSampler`](@ref), and with continuous and categorical hyperparameters, [`CLHSampler`](@ref).
- [Hyperband](https://arxiv.org/abs/1603.06560), [`Hyperband`](@ref), which evaluates many configurations with a small budget and evaluates the most promising configurations again with larger budgets.
- [BOHB](https://arxiv.org/abs/1807.01774), [`BOHB`](@ref), which replaces the random sampling of Hyperband by sampling from kernel density estimates fitted to the evaluated configurations.

The package is designed to add hyperparameter optimization to existing code. The macro [`@hyperopt`](@ref) is wrapped around the code that evaluates a configuration of the hyperparameters, and the user specifies the names of the hyperparameters and their candidate values.

## Installation

```julia
using Pkg
Pkg.add("Hyperopt")
```

## Quick start

Consider code that trains a model with the manually selected hyperparameters `a` and `b`:
```julia
a = manually_selected_value
b = other_value
cost = train_model(a, b)
```
To optimize `a` and `b`, the code is wrapped in `@hyperopt`, and the manually selected values are replaced by vectors of candidate values:
```julia
ho = @hyperopt for i = number_of_samples,
                   a = candidate_values,
                   b = other_candidate_values
    cost = train_model(a, b)
end
```
The following example evaluates 50 configurations drawn at random:
```@example index
using Hyperopt
Hyperopt.SHOW_PROGRESS[] = false # hide
using Random # hide
Random.seed!(0) # hide
train_model(a, b) = (a - 3)^2 + (b ? 0.0 : 1.0)

ho = @hyperopt for i = 50,
                   a = LinRange(1, 5, 100),
                   b = [true, false]
    cost = train_model(a, b)
end
```
The returned [`Hyperoptimizer`](@ref) records the configurations and the results, and provides the best configuration:
```@example index
ho.minimizer, ho.minimum
```
```@example index
printmin(ho)
```

## Which sampler to use?

- [`RandomSampler`](@ref) is the default sampler and a robust baseline. It supports candidate vectors of any length and any element type.
- [`LHSampler`](@ref) and [`CLHSampler`](@ref) distribute the configurations evenly over the candidate values. They are preferable to random search if the number of iterations is small, but they require candidate vectors whose lengths are related to the number of iterations, see [Samplers](samplers.md).
- [`Hyperband`](@ref) is preferable if the cost of an evaluation is determined by a budget, e.g., the number of training epochs or a time limit, and the result obtained with a small budget indicates the result obtained with a large budget.
- [`BOHB`](@ref) is used as the inner sampler of `Hyperband`. It proposes configurations based on the results of the configurations evaluated before.

## Contents

```@contents
Pages = ["hyperopt.md", "samplers.md", "hyperband.md", "bohb.md", "function_interface.md", "optim.md", "plotting.md", "api.md"]
Depth = 1
```

## Citing

If you use Hyperopt.jl in academic work, please cite it using the entry in [`CITATION.bib`](https://github.com/baggepinnen/Hyperopt.jl/blob/master/CITATION.bib):
```bibtex
@misc{bagge2018hyperopt,
  title        = {Hyperopt.jl: Hyperparameter optimization in {J}ulia},
  author       = {Bagge Carlson, Fredrik},
  year         = {2018},
  month        = aug,
  howpublished = {\url{https://github.com/baggepinnen/Hyperopt.jl}},
  url          = {https://github.com/baggepinnen/Hyperopt.jl},
  note         = {Lund University Publications, \url{https://lup.lub.lu.se/record/6ec19989-9b30-448c-be5e-bae4c4257c7b}}
}
```
