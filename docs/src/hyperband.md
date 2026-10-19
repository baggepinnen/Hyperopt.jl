# Hyperband

```@setup hyperband
using Hyperopt, Random
Hyperopt.SHOW_PROGRESS[] = false
Random.seed!(0)
```

[`Hyperband`](@ref) implements [Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization](https://arxiv.org/abs/1603.06560). Hyperband evaluates configurations of the hyperparameters with a small budget, i.e., an amount of resources, and evaluates the configurations with the smallest losses again with a larger budget. The interpretation of the budget is determined by the user, e.g., the number of training epochs, a time limit or the number of iterations of an optimizer.

## Algorithm

Hyperband has two parameters: the maximum budget ``R`` and the ratio ``\eta`` of the budgets of successive rounds. With ``s_{max} = \lfloor \log_\eta R \rfloor``, Hyperband runs the brackets ``s = s_{max}, s_{max}-1, \dots, 0``. Bracket ``s`` performs successive halving:
1. ``n_s = \lceil \frac{s_{max}+1}{s+1} \eta^s \rceil`` configurations are drawn from the inner sampler and evaluated with the budget ``R \eta^{-s}``.
2. In round ``i = 1, \dots, s``, the ``\lfloor n_s \eta^{-i} \rfloor`` configurations with the smallest losses of the previous round are evaluated again with the budget ``R \eta^{i-s}``.

The bracket ``s = s_{max}`` thus evaluates many configurations with the smallest budget, and the bracket ``s = 0`` evaluates few configurations with the maximum budget only.

## Usage with `@hyperopt`

With a `Hyperband` sampler, the first argument of the for loop names the variable that holds the budget of the current evaluation, and its value is the maximum budget ``R``. The parameter ``\eta`` and the inner sampler, which draws the configurations, are arguments of the sampler: `Hyperband(η = 3, inner = RandomSampler())`.

A configuration that is evaluated again with a larger budget can resume from the result of its previous evaluation, e.g., continue the training of a model instead of training a new model. The loop body therefore returns a tuple `(loss, state)`, where `state` is any object from which a later evaluation can resume. In the next evaluation of the same configuration, the variable `state` holds this object; in the first evaluation of a configuration, `state` is `nothing`. If the loop body returns only the loss, `state` is `nothing` in every evaluation.

In the following example, a model with the weights `w` is trained by gradient descent with momentum. The hyperparameters are the learning rate `lr` and the momentum coefficient `β`, and the budget is the number of training epochs. The state contains the weights, the velocity and the number of epochs trained so far. Configurations with a large learning rate diverge, i.e., their loss increases with the number of epochs.
```@example hyperband
const scales = [1.0, 10.0, 100.0]
loss(w) = sum(scales .* (w .- 1) .^ 2)
gradient(w) = 2 .* scales .* (w .- 1)

# Gradient descent with momentum, starting from the weights w and the velocity v
function train(w, v, lr, β, epochs)
    for _ in 1:epochs
        v = β * v - lr * gradient(w)
        w = w + v
    end
    w, v
end

ho = @hyperopt for epochs = 81,
                   sampler = Hyperband(η = 3, inner = RandomSampler()),
                   lr = exp10.(LinRange(-4, -1.5, 100)),
                   β = LinRange(0, 0.95, 100)
    if state === nothing # The first evaluation of the configuration (lr, β)
        w, v, trained = zeros(3), zeros(3), 0
    else                 # Continue the training of the previous evaluation
        w, v, trained = state
    end
    w, v = train(w, v, lr, β, round(Int, epochs) - trained)
    loss(w), (w, v, round(Int, epochs))
end
nothing # hide
```

## Recorded evaluations

The `Hyperoptimizer` records the configuration, the loss and the budget of each evaluation in `ho.history`, `ho.results` and `ho.sampler.budgets`, respectively. A configuration that is evaluated with several budgets occurs several times in `ho.history`. The number of evaluations with each budget is
```@example hyperband
budgets = ho.sampler.budgets
[budget => count(==(budget), budgets) for budget in sort(unique(budgets))]
```
Since losses obtained with different budgets are not comparable, `ho.minimum`, `ho.minimizer`, `ho.maximum`, `ho.maximizer` and [`printmin`](@ref) consider only the evaluations with the largest budget:
```@example hyperband
ho.minimum, ho.minimizer
```
`ho.sampler.minimum` is the named tuple `(; loss, budget, state, configuration)` of the evaluation that attains `ho.minimum`. Its state contains the trained weights:
```@example hyperband
best = ho.sampler.minimum
best.budget, best.configuration, best.state[1]
```
`plot(ho)` shows the results of each budget as a separate series, see [Plotting](plotting.md).

## Continuing a Hyperband optimization

The argument `ho = ho` continues the optimization: all brackets are run again with new configurations, and the evaluations are appended to `ho.history`, `ho.results` and `ho.sampler.budgets`. The sampler of the existing `Hyperoptimizer` is continued, including its maximum budget and `ho.sampler.minimum`. For brevity, the loop body of the example above is moved into a function:
```@example hyperband
function evaluate(lr, β, epochs, state)
    w, v, trained = state === nothing ? (zeros(3), zeros(3), 0) : state
    w, v = train(w, v, lr, β, round(Int, epochs) - trained)
    loss(w), (w, v, round(Int, epochs))
end

n = length(ho.results)
ho = @hyperopt for epochs = 81, ho = ho,
                   sampler = Hyperband(η = 3, inner = RandomSampler()),
                   lr = exp10.(LinRange(-4, -1.5, 100)),
                   β = LinRange(0, 0.95, 100)
    evaluate(lr, β, epochs, state)
end
n, length(ho.results), ho.minimum
```

## Inner samplers

The configurations are drawn from the inner sampler, [`RandomSampler`](@ref) by default. With [`LHSampler`](@ref) or [`CLHSampler`](@ref), the configurations of all brackets of a run are distinct points of a Latin hypercube plan, which contains one point per element of the candidate vectors. The candidate vectors must therefore contain at least as many elements as a run draws configurations, which is `ho.iterations` after the run; otherwise, an error that states the required number is thrown. With ``R = 27`` and ``\eta = 3``, a run draws 49 configurations:
```@example hyperband
ho_lhs = @hyperopt for epochs = 27,
                       sampler = Hyperband(η = 3, inner = LHSampler()),
                       lr = exp10.(LinRange(-4, -1.5, 100)),
                       β = LinRange(0, 0.95, 100)
    evaluate(lr, β, epochs, state)
end
ho_lhs.iterations, length(unique(ho_lhs.history))
```
[`BOHB`](@ref) is an inner sampler that proposes configurations based on the losses of the evaluated configurations, see [BOHB](bohb.md).

## The maximum budget

The maximum budget is the value of the first loop argument. It can also be given to the sampler, `Hyperband(R = 81, η = 3)`, in which case the value of the first loop argument is to be equal to `R`. If the two values differ, `R` of the sampler is used and a deprecation warning is emitted. In versions before v0.6, the value of the first loop argument was ignored. When an optimization is continued with `ho = ...`, the maximum budget of the existing sampler is used, and a warning is emitted if the value of the first loop argument differs from it.

The budgets of the evaluations are ``R/η^k`` for ``k = 0, 1, …`` and are not smaller than 1. `R` is therefore a real number of at least 1, e.g., `R = 4.5` for a time limit of 4.5 seconds with a smallest budget of 1 second. Successive halving requires ``R ≥ η``; with ``R < η``, a run evaluates a single configuration with the budget `R`, and a warning is emitted.

See [Function interface](function_interface.md) for [`hyperband`](@ref), which runs Hyperband without the macro.
