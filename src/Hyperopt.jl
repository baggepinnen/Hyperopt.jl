"""
Hyperparameter optimization by random search, Latin hypercube sampling, Hyperband and BOHB.

The main interface is the macro [`@hyperopt`](@ref), which evaluates the body of a for loop for configurations of the
hyperparameters and returns a [`Hyperoptimizer`](@ref) that records the configurations and the results. The
documentation is available at <https://baggepinnen.github.io/Hyperopt.jl/stable/>.
"""
module Hyperopt

export Hyperoptimizer, @hyperopt, @phyperopt, @thyperopt, printmin, printmax, warn_on_boundary
export RandomSampler, LHSampler, CLHSampler, hyperband, Hyperband, hyperoptim, BOHB, Continuous, Categorical, UnorderedCategorical

using Random
using ProgressMeter
using MacroTools
using RecipesBase
using Distributed
using Printf

"""
    DimensionType

Abstract supertype of the types `Continuous`, `Categorical` and `UnorderedCategorical`, which describe the hyperparameters for [`CLHSampler`](@ref) and [`BOHB`](@ref).
"""
abstract type DimensionType end

"""
    Continuous()

An ordered hyperparameter, e.g., a real-valued or an integer-valued hyperparameter. [`CLHSampler`](@ref) requires the candidate vector of a `Continuous` hyperparameter to have the same length as the number of iterations. [`BOHB`](@ref) requires real-valued candidates and estimates the densities in the coordinate given by the index into the sorted candidate vector.
"""
struct Continuous <: DimensionType end

"""
    Categorical(levels)

A categorical hyperparameter with `levels` candidates, i.e., its candidate vector has `levels` elements. [`CLHSampler`](@ref) and [`BOHB`](@ref) treat the candidates as unordered, `Categorical(levels)` is therefore equivalent to `UnorderedCategorical(levels)`.
"""
struct Categorical <: DimensionType
    levels::Int
    function Categorical(levels::Integer)
        levels >= 1 || throw(ArgumentError("The number of levels of a categorical dimension must be positive, got $levels"))
        new(levels)
    end
end

"""
    UnorderedCategorical(levels)

A categorical hyperparameter with `levels` unordered candidates, e.g., `UnorderedCategorical(3)` for `fun = [tanh, sin, cos]`. Equivalent to [`Categorical(levels)`](@ref Categorical).
"""
struct UnorderedCategorical <: DimensionType
    levels::Int
    function UnorderedCategorical(levels::Integer)
        levels >= 1 || throw(ArgumentError("The number of levels of a categorical dimension must be positive, got $levels"))
        new(levels)
    end
end

abstract type Sampler end

"""
    Hyperoptimizer(iterations, sampler = RandomSampler(); name1 = candidates1, name2 = candidates2, ...)
    Hyperoptimizer(; iterations, params, candidates, history = [], results = [], sampler = RandomSampler(), objective = nothing)

Specification and record of a hyperparameter optimization. [`@hyperopt`](@ref) constructs and returns a
`Hyperoptimizer`.

The first constructor creates a `Hyperoptimizer` for manual iteration. The keyword arguments name the hyperparameters
and give their candidate vectors. Each iteration over the `Hyperoptimizer` draws a configuration from the sampler,
appends it to `ho.history` and yields it as a `NamedTuple` `(; i, name1, name2, ...)`, where `i` is the index of the
iteration. The result of each configuration is to be appended to `ho.results` by the caller:
```julia
ho = Hyperoptimizer(10, a = LinRange(1, 2, 50), b = [true, false])
for (i, a, b) in ho
    push!(ho.results, cost(a, b))
end
```
See [`iterate`](@ref Base.iterate(::Hyperoptimizer)) for the semantics of repeated iteration. The second
constructor sets the fields directly.

# Fields
- `iterations::Int`: The number of configurations that one run draws, i.e., one execution of `@hyperopt` or one
  iteration over the `Hyperoptimizer`. With a [`Hyperband`](@ref) sampler, [`hyperband`](@ref) sets it to the number
  of configurations that one run of Hyperband draws from the inner sampler.
- `params`: The names of the hyperparameters, as symbols.
- `candidates`: The candidate vectors, one per hyperparameter, in the order of `params`.
- `history`: The evaluated configurations. `history[k]` contains one value per hyperparameter, in the order of
  `params`.
- `results`: The results of the evaluations. `results[k]` is the result of the evaluation of the configuration
  `history[k]`.
- `sampler`: The sampler, e.g., [`RandomSampler`](@ref), [`LHSampler`](@ref), [`CLHSampler`](@ref) or
  [`Hyperband`](@ref).
- `objective`: The function that `@hyperopt` generates from the loop body, called as
  `objective(i, values...; state)`, or `nothing` for a `Hyperoptimizer` that is iterated manually.

# Properties
- `ho.minimum`: The smallest result.
- `ho.minimizer`: The configuration of the smallest result, an element of `ho.history`.
- `ho.maximum`: The largest result.
- `ho.maximizer`: The configuration of the largest result, an element of `ho.history`.

These properties compare the elements of `ho.results`. A `NaN` result, e.g., of an evaluation that failed, is treated
as `Inf` by `minimum` and `minimizer` and as `-Inf` by `maximum` and `maximizer`; it is therefore selected only if all
compared results are `NaN`, in which case `ho.minimum` is `Inf` and `ho.maximum` is `-Inf`. If `ho.results` is
empty, `ho.minimum` and `ho.maximum` are `NaN`, and `ho.minimizer` and `ho.maximizer` are empty vectors.
With a [`Hyperband`](@ref) sampler, only the results obtained with the largest budget in `ho.sampler.budgets` are
compared, since results obtained with different budgets are not comparable.

`minimum(ho)` and `maximum(ho)` are equivalent to `ho.minimum` and `ho.maximum`. [`printmin`](@ref) and
[`printmax`](@ref) print the minimizer and the maximizer, and [`warn_on_boundary`](@ref) reports hyperparameters
whose optimal value is located on the boundary of the sampled values. If Plots.jl is loaded, `plot(ho)` plots the
results against the value of each hyperparameter.
"""
Base.@kwdef mutable struct Hyperoptimizer{S<:Sampler, F, P, C}
    iterations::Int
    params::P
    candidates::C
    history = []
    results = []
    sampler::S = RandomSampler()
    objective::F = nothing
end

function Base.getproperty(ho::Hyperoptimizer, s::Symbol)
    if s === :minimum || s === :minimizer || s === :maximum || s === :maximizer
        idx = compared_indices(ho)
        isempty(idx) && return s === :minimum || s === :maximum ? NaN : []
        sense_min = s === :minimum || s === :minimizer
        results = replace(ho.results[idx], NaN => (sense_min ? Inf : -Inf))
        k = sense_min ? argmin(results) : argmax(results)
        return s === :minimum || s === :maximum ? results[k] : ho.history[idx[k]]
    end
    return getfield(ho,s)
end

# Indices of the results that `ho.minimum`, `ho.minimizer`, `ho.maximum` and `ho.maximizer` compare
compared_indices(ho::Hyperoptimizer) = eachindex(ho.results)

function Base.show(io::IO, ho::Hyperoptimizer)
    println(io, "Hyperoptimizer with")
    candstring = map(zip(ho.params, ho.candidates)) do (param, cand)
        if length(cand) <= 3
            "  $(param) candidates: [$(join(repr.(cand), ", "))]"
        else
            "  $(param) length: $(length(cand))"
        end
    end
    candstring = join(candstring, "\n")
    println(io, candstring)
    println(io, "  minimum / maximum: $((ho.minimum, ho.maximum))")
    println(io, "  minimizer:")
    for (i, v) in enumerate(ho.minimizer)
        @printf(io, "%9s ", string(ho.params[i]))
    end
    println(io)
    for (i, v) in enumerate(ho.minimizer)
        if v isa AbstractFloat
            @printf(io, "%9.4g ", v)
        else
            @printf(io, "%9s ", v)
        end
    end
    println(io)
end

Base.propertynames(ho::Hyperoptimizer) = (:minimum, :minimizer, :maximum, :maximizer, fieldnames(Hyperoptimizer)...)

include("samplers.jl")

function Hyperoptimizer(iterations::Int, sampler::Sampler = RandomSampler(); kwargs...)
    check_sampler(sampler)
    params = keys(kwargs)
    candidates = []
    for kw in kwargs
        push!(candidates, kw[2])
    end
    ho = Hyperoptimizer(iterations=iterations, params=params, candidates=candidates, sampler=sampler)
    init!(sampler, ho)
    ho
end

Base.length(ho::Hyperoptimizer) = ho.iterations
Base.eltype(::Type{<:Hyperoptimizer}) = NamedTuple
# The field types of the yielded `NamedTuple` depend on the sampled values, `collect` therefore determines the element type from the elements.
Base.IteratorEltype(::Type{<:Hyperoptimizer}) = Base.EltypeUnknown()

"""
    iterate(ho::Hyperoptimizer, state = 1; update_history = true)

Draw the sample of iteration `state` from the sampler of `ho` and return it as a `NamedTuple` with the iteration index `i` and one entry per parameter, e.g., `(i = 1, a = 0.3, b = true)`.

If `update_history = true`, the sample is appended to `ho.history`; the corresponding result is to be appended to `ho.results` by the caller:
```julia
for (i, a, b) in ho
    push!(ho.results, f(a, b))
end
```
`ho.history[k]` and `ho.results[k]` then refer to the same evaluation. Entries at the end of `ho.history` without a corresponding entry in `ho.results`, e.g., the samples of an earlier iteration whose results were not recorded, are removed when the iteration starts (`state = 1`). Iterating over `ho` again after the results of all samples were recorded appends `ho.iterations` new samples, which corresponds to continuing the optimization. For [`LHSampler`](@ref) and [`CLHSampler`](@ref), such an iteration evaluates a new plan.
"""
function Base.iterate(ho::Hyperoptimizer, state=1; update_history=true)
    state > ho.iterations && return nothing
    if update_history && state == 1
        discard_unevaluated!(ho)
        start_iteration!(ho.sampler, ho)
    end
    samples = ho.sampler(ho, state)
    update_history && push!(ho.history, samples)
    nt = (; Pair.((:i, ho.params...), (state, samples...))...)
    nt, state+1
end

"""
    discard_unevaluated!(ho::Hyperoptimizer)

Remove the entries at the end of `ho.history` for which `ho.results` contains no corresponding result.
"""
function discard_unevaluated!(ho::Hyperoptimizer)
    n = length(ho.results)
    length(ho.history) > n && resize!(ho.history, n)
    ho
end

"""
Hyperopt internal function

Returns
`params, candidates, sampler_, ho_, fun`
where
`params` is a vector of symbols
`candidates` is a vector of expressions creating candidate vectors
`sampler_` is either nothing or an expression creating a sampler
`ho_` is either nothing or an expression for an existing hyperoptimizer
`fun` is a function `(i, pars...; state = nothing) -> val` that executes the macrobody

The expression `ex` is not modified.
"""
function preprocess_expression(ex)
    ex.head == :for || error("Wrong syntax, Use For-loop syntax, ex: @hyperopt for i=100, param=LinRange(1,10,100) ...")
    params     = []
    candidates = []
    sampler_ = nothing
    ho_ = nothing
    for arg in ex.args[1].args # ex.args[1] = the arguments to the For loop
        if @capture(arg, sampler = sam_) # A sampler was provided
            sampler_ = sam
        elseif @capture(arg, ho = h_)
            ho_ = h
        else
            @capture(arg, param_ = list_) || error("Wrong syntax, Use For-loop syntax, ex: @hyperopt for i=100, param=LinRange(1,10,100) ...")
            push!(params, param)
            push!(candidates, list)
        end
    end
    params = (params...,)
    funname = gensym(:hyperopt_objective)
    body = loop_body(ex)
    # The state is a keyword argument so that a new configuration and a resumed state are
    # not distinguished by the number of positional arguments, see `successive_halving`.
    fun = quote
        function $(funname)($(esc.(params)...); $(esc(:state)) = nothing)
            $(esc(body))
        end
    end

    params, candidates, sampler_, ho_, fun
end

"""
Hyperopt internal function

Returns the body of the for loop `ex` without a trailing `LineNumberNode`, so that the body does not evaluate to `nothing`, ref https://github.com/JuliaLang/julia/pull/41857
The expression `ex` is not modified.
"""
function loop_body(ex)
    body = ex.args[2]
    if Meta.isexpr(body, :block) && !isempty(body.args) && body.args[end] isa LineNumberNode
        body = Expr(:block, body.args[1:end-1]...)
    end
    body
end


"""
    Hyperopt.SHOW_PROGRESS[] = false

Disable the progress meters that `@hyperopt`, `hyperband` and `hyperoptim` display. The progress meters are enabled by default.
"""
const SHOW_PROGRESS = Ref(true)

function optimize(ho::Hyperoptimizer)
    try
        @showprogress desc="Hyperoptimizing" enabled=SHOW_PROGRESS[] for nt = ho
            res = ho.objective(nt...)
            push!(ho.results, res)
        end
    catch e
        discard_unevaluated!(ho)
        if e isa InterruptException
            @info "Aborting hyperoptimization"
        else
            rethrow()
        end
    end
    ho
end

"""
    resumed_sampler(existing, given)

Return the sampler of a run that resumes a `Hyperoptimizer` whose sampler is `existing`. The existing sampler is continued, so that the state of its random number generator and, e.g., the observations of `BOHB` are retained. A given sampler of another type is ignored with a warning. `given` is `nothing` if no sampler is given.
"""
function resumed_sampler(existing, given)
    if given !== nothing && sampler_kind(given) != sampler_kind(existing)
        @warn "The sampler of the existing Hyperoptimizer, $(sampler_kind(existing)), is continued; the given sampler, $(sampler_kind(given)), is ignored"
    end
    existing
end
sampler_kind(s) = string(nameof(typeof(s)))
sampler_kind(s::Hyperband) = "Hyperband with the inner sampler $(sampler_kind(s.inner))"

# BOHB proposes configurations from the observations that Hyperband records, it is therefore only valid as an inner sampler
check_sampler(::Sampler) = nothing
check_sampler(::BOHB) = throw(ArgumentError("BOHB is an inner sampler of Hyperband, e.g., `sampler = Hyperband(η = 3, inner = BOHB(dims = [...]))`. Without Hyperband, no observations are recorded, and BOHB would draw only random configurations."))

function check_iterations(n)
    n isa Integer || throw(ArgumentError("The value of the first loop argument of `@hyperopt` is the number of iterations and must be an integer, got $(repr(n))"))
    n
end

# A continued Hyperband run uses the maximum budget of the existing sampler
function check_resumed_R(hb::Hyperband, value)
    value == hb.R || @warn "The value $(repr(value)) of the first loop argument differs from the maximum budget R = $(repr(hb.R)) of the continued Hyperband sampler and is not used"
    hb
end

function create_ho(params, candidates, sampler, ho_, objective_)
    quote
        objective = $(objective_)
        allcands = $(Expr(:tuple, esc.(candidates)...)) # The first element is the number of iterations, or the maximum budget of Hyperband
        sampler = $(esc(sampler))
        existing_ho = $(esc(ho_))
        resumed = existing_ho isa Hyperoptimizer
        if resumed
            sampler = resumed_sampler(existing_ho.sampler, sampler)
        elseif sampler === nothing
            sampler = RandomSampler()
        end
        check_sampler(sampler)
        if sampler isa Hyperband # The inner sampler is initialized by `hyperband`
            resumed ? check_resumed_R(sampler, allcands[1]) : set_R!(sampler, allcands[1])
            iterations = nconfigurations(sampler)
        else
            iterations = check_iterations(allcands[1])
        end
        ho = Hyperoptimizer(; iterations, params = $(esc(params))[2:end], candidates = allcands[2:end], sampler, objective)
        if resumed # get info from existing ho. the objective function might be changed, due to variables are captured into the closure, so the type of ho also changed.
            ho.history = existing_ho.history # it's important to use the same array, not copy.
            ho.results = existing_ho.results
            init!(ho.sampler, ho)
        else
            ho.sampler isa Union{LHSampler,CLHSampler} && init!(ho.sampler, ho)
        end
        ho
    end
end

"""
    ho = @hyperopt for i = n, [sampler = RandomSampler(),] [ho = existing_ho,] name1 = candidates1, name2 = candidates2, ...
        loop body
    end

Evaluate the loop body for configurations of the hyperparameters `name1, name2, ...` drawn from the candidate vectors
`candidates1, candidates2, ...`, and return the [`Hyperoptimizer`](@ref) `ho` that records the configurations in
`ho.history` and the results in `ho.results`. The optimal configuration is obtained with `ho.minimizer`, see
[`Hyperoptimizer`](@ref).

The arguments of the for loop are:
- `i = n`: The first argument. Without [`Hyperband`](@ref), `n` is the number of iterations, i.e., the number of
  configurations that are drawn and evaluated, an integer, and `i` holds the index of the current iteration. With a
  `Hyperband` sampler, `n` is the maximum budget `R`, a real number of at least 1, and `i` holds the budget of the
  current evaluation. Any variable name can be used instead of `i`.
- `sampler = s` (optional): The sampler that draws the configurations: [`RandomSampler`](@ref) (the default),
  [`LHSampler`](@ref), [`CLHSampler`](@ref) or [`Hyperband`](@ref). [`BOHB`](@ref) is only valid as the inner sampler
  of `Hyperband`.
- `ho = existing_ho` (optional): A `Hyperoptimizer` whose optimization is continued. The configurations and results of
  the new run are appended to `existing_ho.history` and `existing_ho.results`, which the returned `Hyperoptimizer`
  shares with `existing_ho`. The sampler of `existing_ho` is continued, including the state of its random number
  generator and, for [`BOHB`](@ref), its observations. A sampler given with `sampler = s` is not used, and a warning is
  emitted if it is of another type than the sampler of `existing_ho`. With a `Hyperband` sampler, the value `n` is not
  used, and a warning is emitted if it differs from the maximum budget `R` of the existing sampler.
- `name = candidates`: One argument per hyperparameter. `candidates` is a vector of the values that the hyperparameter
  can take, e.g., `LinRange(1, 5, 100)`, `exp10.(LinRange(-3, 1, 100))` for logarithmically spaced values,
  `[true, false]` or `[tanh, sin]`. In the loop body, the variable `name` holds the value of the hyperparameter.

The names `sampler` and `ho` are reserved and cannot be used as names of hyperparameters.

The loop body becomes the body of a function whose arguments are the first loop variable and the hyperparameters;
variables of the enclosing scope are accessible as in a closure. The keyword argument `state` of this function is
also accessible in the loop body; it is `nothing` unless the sampler is a `Hyperband` sampler.

The value of the loop body is the result of the evaluation:
- Without `Hyperband`, the result is appended to `ho.results`. The result is typically a real-valued cost, which
  `ho.minimum` and `ho.minimizer` minimize.
- With `Hyperband`, the loop body returns a tuple `(loss, state)`, where `state` is any object from which a later
  evaluation of the same configuration with a larger budget can resume, or only the real-valued loss. The loss is
  appended to `ho.results`, and the state is passed to the next evaluation of the configuration as the keyword
  argument `state`. See [`Hyperband`](@ref).

If an evaluation throws an `InterruptException`, e.g., after Ctrl-C, the optimization is stopped and `ho` is returned
with the results obtained so far. Other exceptions are rethrown. A progress meter is displayed unless
[`Hyperopt.SHOW_PROGRESS[]`](@ref Hyperopt.SHOW_PROGRESS) is `false`. The macros [`@phyperopt`](@ref) and
[`@thyperopt`](@ref) evaluate the loop body in parallel.

# Example
```julia
cost(a, b, c) = (a - 3)^2 + (b ? 10 : 20) + (log10(c) - 2)^2
ho = @hyperopt for i = 50, sampler = RandomSampler(), a = LinRange(1, 5, 1000), b = [true, false], c = exp10.(LinRange(-1, 3, 1000))
    cost(a, b, c)
end
ho.minimum, ho.minimizer
```
"""
macro hyperopt(ex)
    pre = preprocess_expression(ex)
    ho_ = create_ho(pre...)
    :(optimize($ho_))
end

include("parallel.jl")

"""
    minimum(ho::Hyperoptimizer)

Return the smallest result in `ho.results`, equivalent to `ho.minimum`. `NaN` results are treated as `Inf`. If `ho` contains no results, `NaN` is returned.
With a [`Hyperband`](@ref) sampler, only the results obtained with the largest budget are compared.
"""
Base.minimum(ho::Hyperoptimizer) = ho.minimum

"""
    maximum(ho::Hyperoptimizer)

Return the largest result in `ho.results`, equivalent to `ho.maximum`. `NaN` results are treated as `-Inf`. If `ho` contains no results, `NaN` is returned.
With a [`Hyperband`](@ref) sampler, only the results obtained with the largest budget are compared.
"""
Base.maximum(ho::Hyperoptimizer) = ho.maximum

"""
    printmin([io = stdout,] ho)

Print the name and the value of each hyperparameter of the configuration `ho.minimizer`, one hyperparameter per line.
With a [`Hyperband`](@ref) sampler, the configuration is the minimizer among the results obtained with the largest
budget, see [`Hyperoptimizer`](@ref).
"""
printmin(ho::Hyperoptimizer) = printmin(stdout, ho)
function printmin(io::IO, ho::Hyperoptimizer)
    for (param, value) in zip(ho.params, ho.minimizer)
        println(io, param, " = ", value)
    end
end

"""
    printmax([io = stdout,] ho)

Print the name and the value of each hyperparameter of the configuration `ho.maximizer`, one hyperparameter per line.
With a [`Hyperband`](@ref) sampler, the configuration is the maximizer among the results obtained with the largest
budget, see [`Hyperoptimizer`](@ref).
"""
printmax(ho::Hyperoptimizer) = printmax(stdout, ho)
function printmax(io::IO, ho::Hyperoptimizer)
    for (param, value) in zip(ho.params, ho.maximizer)
        println(io, param, " = ", value)
    end
end

"""
    warn_on_boundary(ho, sense = :min)

Emit a warning for each hyperparameter whose value in the optimal configuration is the smallest or the largest value
of this hyperparameter in `ho.history`, and return the names of these hyperparameters as a vector of symbols. The
optimal configuration is `ho.minimizer` if `sense = :min`, and `ho.maximizer` otherwise.

Example: if the hyperparameter `a` has the candidates `1:10` and the optimum is obtained with `a = 1`, the optimum may
be located outside of the candidate range, and the range is to be extended. The warning is also emitted if the
smallest sampled value of `a` is larger than 1 and the optimum is obtained with this value.

Only hyperparameters whose candidates are real numbers with more than three distinct values are considered, since
categorical candidates have no order.
"""
function warn_on_boundary(ho, sense = :min)
    m = sense == :min ? ho.minimizer : ho.maximizer
    on_boundary = Symbol[]
    for i in eachindex(m)
        c = unique(ho.candidates[i])
        (length(c) > 3 && all(x -> x isa Real, c)) || continue
        if m[i] ∈ extrema(h[i] for h in ho.history)
            @warn "Parameter $(ho.params[i]) obtained its optimum on an extremum of the sampled region: $(m[i])"
            push!(on_boundary, Symbol(ho.params[i]))
        end
    end
    on_boundary
end

include("plotting.jl")

"""
    hyperoptim(f, candidates, algorithm = Optim.NelderMead(), opts = Optim.Options(); R = 30, η = 3, inner = RandomSampler(), threads = true, verbose = false, Rmin = 0.1, kwargs...)

Multi-start optimization with Optim and Hyperband. The interface is similar to that of `Optim.optimize`, but the initial guess is replaced by candidate vectors, one per element of the initial guess, from which [`hyperband`](@ref) draws initial guesses. The budget of an evaluation is the time limit in seconds of a call to `Optim.optimize`. An evaluation with a larger budget continues the optimization from the minimizer found by the previous evaluation of the same initial guess.

Returns the `Hyperoptimizer` `ho`. `ho.minimum` is the smallest value of `f` found with the largest budget, `ho.minimizer` is the initial guess from which this value was found, and `ho.sampler.minimum.state` is the corresponding minimizer of `f`.

This function requires Optim.jl to be loaded, i.e., `using Optim`.

# Arguments:
- `f`: A function from `Rⁿ` to `R`, i.e., it takes a vector of parameters.
- `candidates`: A vector or a `NamedTuple` of candidate vectors, one per element of the vector of parameters.
- `algorithm`: One of the Optim algorithms.
- `opts`: `Optim.Options`. The field `time_limit` is replaced by the budget of each call to `Optim.optimize`.
- `R`: The maximum budget, i.e., the largest time limit in seconds.
- `η`: The ratio of the budgets of successive rounds of successive halving, see [`Hyperband`](@ref).
- `inner`: The inner sampler
- `threads`: Use threading to evaluate the cost function
- `verbose`: Print the budget and the minimum after each call to `Optim.optimize`.
- `Rmin`: The smallest time limit in seconds; smaller budgets are replaced by `Rmin`.
- `kwargs`: Are sent to `Optim.optimize`.
"""
function hyperoptim end

end # module
