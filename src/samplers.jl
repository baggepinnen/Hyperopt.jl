"""
    RandomSampler{T<:AbstractRNG} <: Sampler

Sample a value for each parameter uniformly at random from the candidate vectors. Log-uniform sampling available by providing a log-spaced candidate vector.

Optionally, pass an `AbstractRNG` to initialize. By default, a `Xoshiro` generator is seeded with a random number drawn from the global RNG, so that `Random.seed!` makes the samples reproducible.

`RandomSampler` is the default sampler of [`@hyperopt`](@ref) and the default inner sampler of [`Hyperband`](@ref). The values are drawn independently and with replacement, a configuration may therefore be drawn more than once. The candidate vectors may have different lengths and may contain values of any type, e.g., functions.
"""
struct RandomSampler{T<:AbstractRNG} <: Sampler
    # The parallel macros draw all samples on the calling process, the RNG is therefore not shared with other processes
    rng::T
    # Permits the concurrent use of the sampler by several tasks
    lock::ReentrantLock
end

RandomSampler() = RandomSampler(Xoshiro(rand(UInt)))
RandomSampler(rng::AbstractRNG) = RandomSampler(rng, ReentrantLock())

function (s::RandomSampler)(ho, iter)
    with_rng(s) do rng
        [list[rand(rng, 1:length(list))] for list in ho.candidates]
    end
end

# Calls f with the RNG of the random sampler
with_rng(f, s::RandomSampler) = lock(() -> f(s.rng), s.lock)


# Latin Hypercube Sampler ======================================================

"""
    LHSampler(; rng = Xoshiro(rand(UInt)), steps = nothing)

Sample from an optimized Latin hypercube plan. The plan contains one point per iteration, and every element of each candidate vector occurs in exactly one point. All candidate vectors must therefore have the same length as the number of iterations. [`CLHSampler`](@ref) admits categorical hyperparameters with fewer candidates.

The plan is obtained from a random Latin hypercube by `steps` proposed exchanges of the entries of two points in one dimension. An exchange is accepted if it reduces the Audze-Eglais potential energy of the plan, i.e., the sum of the inverse squared distances between all pairs of points, in the coordinates given by the candidate indices divided by the number of iterations. The computational cost of a step is proportional to `n*d` for `n` iterations and `d` hyperparameters. The default number of steps, `clamp(10^8 ÷ (n*d), n*d, 100*n*d)`, thus limits the cost to approximately `10^8` evaluations of coordinate differences if `n*d ≤ 10^4`. With `steps = 0`, the plan is a random Latin hypercube.

The plan is drawn from `rng`. By default, a `Xoshiro` generator is seeded with a random number drawn from the global RNG, so that `Random.seed!` makes the plans reproducible.

A new plan is generated for every run of `@hyperopt`, including a run that resumes an existing `Hyperoptimizer` with `ho = ...`, and for every iteration over a `Hyperoptimizer` that begins after results were recorded. The optimization of a new plan reduces the number of its points that coincide with configurations in `ho.history` with priority over the potential energy: an exchange that reduces this number is accepted, and an exchange that increases it is rejected. Configurations are thus not evaluated again if the number of steps suffices. With a single hyperparameter, however, every plan consists of all candidates.

As the inner sampler of [`Hyperband`](@ref), the plan contains one point per element of the candidate vectors instead of one point per iteration, see [`Hyperband`](@ref).

# Example
```julia
ho = @hyperopt for i = 100, sampler = LHSampler(), a = LinRange(1, 5, 100), b = repeat([true, false], 50)
    (a - 3)^2 + b
end
```
"""
mutable struct LHSampler{R<:AbstractRNG} <: Sampler
    rng::R
    steps::Union{Int, Nothing}
    "Candidate indices of the current plan, one row per hyperparameter and one column per iteration"
    plan::Matrix{Int}
    "Length of `ho.results` when the current plan was generated, or -1 if no plan has been generated"
    start::Int
end

function LHSampler(; rng::AbstractRNG = Xoshiro(rand(UInt)), steps::Union{Integer, Nothing} = nothing)
    check_steps(steps)
    LHSampler(rng, steps, zeros(Int, 0, 0), -1)
end

"""
    CLHSampler(; dims, rng = Xoshiro(rand(UInt)), steps = nothing)

Sample from an optimized Latin hypercube plan with continuous and categorical dimensions. `dims` contains one dimension type per hyperparameter, in the order of the candidate vectors, e.g., `dims = [Continuous(), Categorical(2), Continuous()]`:
- `Continuous()`: the candidate vector must have the same length as the number of iterations, and every candidate occurs in exactly one point of the plan.
- `Categorical(k)` or `UnorderedCategorical(k)`: a categorical hyperparameter with `k` candidates, i.e., the candidate vector must have `k` elements. In a plan with `n` points, every candidate occurs in `n ÷ k` or `n ÷ k + 1` points. Both types are treated as unordered: a categorical dimension contributes 1 to the squared distance between two points with different candidates and 0 otherwise. The optimized plan thus distributes the points that share a categorical candidate over the continuous dimensions.

The keyword arguments `rng` and `steps`, and the generation of new plans are described in [`LHSampler`](@ref).

# Example
```julia
ho = @hyperopt for i = 100, sampler = CLHSampler(dims = [Continuous(), Categorical(2)]), a = LinRange(1, 5, 100), b = [true, false]
    (a - 3)^2 + b
end
```
"""
mutable struct CLHSampler{R<:AbstractRNG} <: Sampler
    dims::Vector{DimensionType}
    rng::R
    steps::Union{Int, Nothing}
    plan::Matrix{Int}
    start::Int
end

function CLHSampler(; dims, rng::AbstractRNG = Xoshiro(rand(UInt)), steps::Union{Integer, Nothing} = nothing)
    for dim in dims
        dim isa Union{Continuous, Categorical, UnorderedCategorical} ||
            throw(ArgumentError("CLHSampler supports the dimension types Continuous, Categorical and UnorderedCategorical, got $(typeof(dim))"))
    end
    check_steps(steps)
    CLHSampler(collect(DimensionType, dims), rng, steps, zeros(Int, 0, 0), -1)
end

const LatinHypercubeSampler = Union{LHSampler, CLHSampler}

check_steps(steps) = (steps === nothing || steps >= 0) || throw(ArgumentError("The number of steps must be nonnegative, got $steps"))

dimensions(::LHSampler, ho) = fill(Continuous(), length(ho.candidates))
dimensions(s::CLHSampler, ho) = s.dims

default_steps(n, d) = clamp(10^8 ÷ (n*d), n*d, 100*n*d)

"""
    init!(sampler, ho)

Prepare `sampler` for a run of `ho` that begins at the current length of `ho.results`. A Latin hypercube sampler generates a new plan with `ho.iterations` points, other samplers are not modified.
"""
init!(s::Sampler, ho) = s

function init!(s::LatinHypercubeSampler, ho)
    dims = dimensions(s, ho)
    check_candidates(dims, ho)
    n, d = ho.iterations, length(dims)
    X = random_plan(s.rng, dims, n)
    if n > 1 && d > 1
        steps = something(s.steps, default_steps(n, d))
        optimize_plan!(s.rng, X, [dim isa Continuous for dim in dims], steps, evaluated_configurations(ho)...)
    end
    s.plan = X
    s.start = length(ho.results)
    s
end

"""
    start_iteration!(sampler, ho)

Called when an iteration over `ho` begins. A Latin hypercube sampler generates a new plan if results were recorded after its current plan was generated, other samplers are not modified.
"""
start_iteration!(s::Sampler, ho) = s

function start_iteration!(s::LatinHypercubeSampler, ho)
    (s.start == length(ho.results) && size(s.plan) == (length(ho.candidates), ho.iterations)) || init!(s, ho)
    s
end

function (s::LatinHypercubeSampler)(ho, iter)
    s.start < 0 && init!(s, ho)
    [list[s.plan[dim, iter]] for (dim, list) in enumerate(ho.candidates)]
end

function check_candidates(dims, ho)
    length(dims) == length(ho.candidates) ||
        throw(ArgumentError("Latin hypercube sampling: `dims` contains $(length(dims)) dimension types, but there are $(length(ho.candidates)) hyperparameters"))
    for (dim, c, name) in zip(dims, ho.candidates, ho.params)
        if dim isa Continuous
            length(c) == ho.iterations ||
                throw(ArgumentError("Latin hypercube sampling requires the candidate vector of every continuous hyperparameter to have the same length as the number of iterations, $(ho.iterations), but the candidate vector of $name has length $(length(c))"))
        else
            length(c) == dim.levels ||
                throw(ArgumentError("Latin hypercube sampling: the hyperparameter $name has the dimension type $(nameof(typeof(dim)))($(dim.levels)), but $(length(c)) candidates"))
        end
    end
end

# Random Latin hypercube with one row of candidate indices per dimension and one column per point
function random_plan(rng, dims, n)
    X = Matrix{Int}(undef, length(dims), n)
    for (j, dim) in enumerate(dims)
        X[j, :] = random_levels(rng, dim, n)
    end
    X
end

random_levels(rng, ::Continuous, n) = randperm(rng, n)

function random_levels(rng, dim::Union{Categorical, UnorderedCategorical}, n)
    order = randperm(rng, dim.levels) # Determines which levels occur once more than the others
    shuffle!(rng, [order[mod1(i, dim.levels)] for i in 1:n])
end

# Configurations in `ho.history`, represented by candidate indices, and the index that represents each candidate.
# Equal candidates of a hyperparameter are represented by the same index. Configurations that contain a value
# that is not a candidate are omitted.
function evaluated_configurations(ho)
    lookup = [Dict(v => i for (i, v) in enumerate(c)) for c in ho.candidates]
    representative = [[l[v] for v in c] for (l, c) in zip(lookup, ho.candidates)]
    evaluated = Set{Vector{Int}}()
    for h in ho.history
        (h isa Union{AbstractVector, Tuple} && length(h) == length(lookup)) || continue
        x = [get(l, v, 0) for (l, v) in zip(lookup, h)]
        0 in x || push!(evaluated, x)
    end
    representative, evaluated
end

# Squared difference of the coordinates of the candidate indices x and y in a dimension of a plan with n points
coordinate_distance2(continuous::Bool, x, y, n) = continuous ? ((x - y) / n)^2 : Float64(x != y)

function squared_distance(X, continuous, a, b)
    n = size(X, 2)
    s = 0.0
    @inbounds for j in axes(X, 1)
        s += coordinate_distance2(continuous[j], X[j, a], X[j, b], n)
    end
    s
end

# Contribution of a pair of points with the squared distance s to the potential energy. The lower bound of the
# squared distance is smaller than any positive squared distance and only applies to coincident points, which
# occur if all dimensions are categorical.
pair_potential(s, n) = 1 / max(s, 1 / (4n^2))

"""
    optimize_plan!(rng, X, continuous, steps, representative, evaluated)

Optimize the plan `X` by `steps` proposed exchanges of two entries in a row, see [`LHSampler`](@ref). `continuous[j]` indicates whether dimension `j` is continuous. The candidate index `i` of dimension `j` is represented by `representative[j][i]` in the set `evaluated` of configurations that are not to be evaluated again.
"""
function optimize_plan!(rng, X, continuous, steps, representative, evaluated)
    d, n = size(X)
    x = Vector{Int}(undef, d)
    function nevaluated(points...)
        isempty(evaluated) && return 0
        count(points) do i
            for j in 1:d
                x[j] = representative[j][X[j, i]]
            end
            x in evaluated
        end
    end
    for _ in 1:steps
        j, a, b = rand(rng, 1:d), rand(rng, 1:n), rand(rng, 1:n)
        X[j, a] == X[j, b] && continue
        Δ = 0.0 # Change of the potential energy by the exchange
        for k in 1:n
            (k == a || k == b) && continue
            sa, sb = squared_distance(X, continuous, a, k), squared_distance(X, continuous, b, k)
            ca = coordinate_distance2(continuous[j], X[j, a], X[j, k], n)
            cb = coordinate_distance2(continuous[j], X[j, b], X[j, k], n)
            Δ += pair_potential(sa - ca + cb, n) + pair_potential(sb - cb + ca, n) - pair_potential(sa, n) - pair_potential(sb, n)
        end
        before = nevaluated(a, b)
        X[j, a], X[j, b] = X[j, b], X[j, a]
        after = nevaluated(a, b)
        if after > before || (after == before && Δ >= 0)
            X[j, a], X[j, b] = X[j, b], X[j, a]
        end
    end
    X
end


# Hyperband ====================================================================

"""
    Hyperband(; R = nothing, η = 3, inner = RandomSampler())
    Hyperband(R)

Sampler that implements [Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization](https://arxiv.org/abs/1603.06560).
Hyperband runs several brackets of successive halving. In each bracket, configurations drawn from the sampler `inner`
are evaluated with a budget, i.e., an amount of resources. The fraction `1/η` of the configurations with the smallest
losses is evaluated again with `η` times the budget, until the maximum budget `R` is reached.

With `@hyperopt`, the first loop argument names the variable that holds the budget of an evaluation, and its value is
the maximum budget `R`, which then does not have to be given to the sampler:
```julia
ho = @hyperopt for resources = 81, sampler = Hyperband(η = 3), a = LinRange(1, 5, 100), b = [true, false]
    model = state === nothing ? initial_model(a, b) : state
    train!(model, resources)
    loss(model), model
end
```
In every evaluation, the hyperparameter variables hold the configuration. The keyword `state` is `nothing` in the first
evaluation of a configuration, and otherwise the state returned by the previous evaluation of the same configuration.
The loop body returns `(loss, state)`, where the state is any object from which a later evaluation can resume, e.g.,
the weights of a model and the state of its optimizer. If the loop body returns only the loss, `state` is `nothing` in
every evaluation. If `R` is also given to the sampler and differs from the value of the first loop argument, the value
of the sampler is used and a deprecation warning is emitted.

The budgets of the evaluations are `R/η^k` for `k = 0, 1, ...` and are not smaller than 1; `R` is therefore a real
number of at least 1, e.g., `R = 4.5` for a time limit of 4.5 seconds with a smallest budget of 1 second. Successive
halving requires `R ≥ η`; with `R < η`, a run evaluates a single configuration with the budget `R`, and a warning is
emitted.

The evaluations are recorded in `ho.history` (the configurations), `ho.results` (the losses) and `ho.sampler.budgets`
(the budgets). `ho.minimum`, `ho.minimizer`, `ho.maximum` and `ho.maximizer` compare only the results obtained with the
largest budget. `ho.sampler.minimum` is `nothing` before the first evaluation, and otherwise the named tuple
`(; loss, budget, state, configuration)` of the evaluation that `ho.minimum` refers to. `plot(ho)` shows the results
of each budget as a separate series. `ho.iterations` is the number of configurations that one run of Hyperband draws
from `inner`. A `Hyperband` sampler records the evaluations of one `Hyperoptimizer` and cannot be used for several.

A run that continues an existing `Hyperoptimizer` with `ho = ...` runs all brackets again and continues the sampler of
the existing `Hyperoptimizer`, including `R`, `η`, the recorded budgets and `minimum`. The value of the first loop
argument of the continued run is not used, and a warning is emitted if it differs from `R`.

With `inner = LHSampler()` or `CLHSampler(...)`, the configurations of all brackets of a run are distinct points of a
Latin hypercube plan, which contains one point per element of the candidate vectors. The candidate vectors must
therefore contain at least `ho.iterations` elements.

See also [`hyperband`](@ref) for an interface without the macro.
"""
Base.@kwdef mutable struct Hyperband <: Sampler
    R = nothing
    η = 3
    inner::Sampler = RandomSampler()
    budgets::Vector{Float64} = Float64[]
    minimum::Union{Nothing, NamedTuple} = nothing
    # Number of configurations drawn from `inner` in the current run, the next configuration is drawn with the index
    # `nsampled + 1`
    nsampled::Int = 0
end
Hyperband(R) = Hyperband(R=R)

# Sets the maximum budget from the value of the first loop argument of `@hyperopt`
function set_R!(hb::Hyperband, value)
    if hb.R === nothing
        hb.R = value
    elseif value != hb.R
        Base.depwarn("The value of the first loop argument of `@hyperopt` with a `Hyperband` sampler is the maximum budget R. The value $(repr(value)) differs from R = $(repr(hb.R)) of the sampler and is ignored; a differing value will be an error in a future release. Give the same value in both places, or omit R from the sampler, e.g., `@hyperopt for resources = $(repr(hb.R)), sampler = Hyperband(η = $(repr(hb.η))), ...`.", :set_R!; force=true)
    end
    hb
end

# The number of configurations and the budgets of the rungs of each bracket, in the order in which Hyperband runs the
# brackets. The budgets are computed from `R`, so that rungs of different brackets with the same budget have equal budgets.
function brackets(R, η)
    η > 1 || throw(ArgumentError("Hyperband: η must be greater than 1, got η = $η"))
    R isa Real && R >= 1 || throw(ArgumentError("Hyperband: the maximum budget R must be a real number of at least 1, since the budgets R/η^k of the evaluations are not smaller than 1; got R = $(repr(R)). Express the budget in units in which the smallest budget is at least 1."))
    if log(η,R) ≈ ceil(log(η, R))       # Catch machine precision error for e.g.
        smax = ceil(Int, log(η, R))     # η=3 and R ∈ (3 .^ [5, 10, 13, 15, 17, 20, 23, 26, 27]), or
    else                                # η=9 and R ∈ (9 .^ [5, 10, 13, 15, 17])
        smax = floor(Int, log(η,R))
    end
    B = (smax + 1)*R # Total budget of a bracket
    map(smax:-1:0) do s
        n = ceil(Int, (B/R)*((η^s)/(s+1)))
        n, [R / η^(s-i) for i in 0:s]
    end
end

nconfigurations(hb::Hyperband) = sum(first, brackets(hb.R, hb.η))

function optimize(ho::Hyperoptimizer{<:Hyperband}, mapfun = map)
    try
        hyperband(ho; mapfun)
    catch e
        is_interrupt(e) || rethrow()
        @info "Aborting hyperband"
    end
    ho
end

"""
    hyperband(ho::Hyperoptimizer{Hyperband}; threads = false, mapfun = threads ? Hyperopt.tmap : map)

Run Hyperband with the sampler `ho.sampler` and the objective `ho.objective`, which is called as
`ho.objective(budget, configuration...; state)` and returns `(loss, state)` or only the loss, see [`Hyperband`](@ref).
The evaluations are appended to `ho.history`, `ho.results` and `ho.sampler.budgets`. `mapfun` evaluates the
configurations of a rung, see `Hyperopt.successive_halving`.

Return `ho.sampler.minimum`.
"""
function hyperband(ho::Hyperoptimizer{Hyperband}; threads=false, mapfun=threads ? tmap : map)
    hb = ho.sampler
    hb.R === nothing && throw(ArgumentError("Hyperband: the maximum budget R is not specified"))
    length(hb.budgets) == length(ho.results) || throw(ArgumentError("The Hyperband sampler has recorded the budgets of $(length(hb.budgets)) evaluations, but the Hyperoptimizer contains $(length(ho.results)) results. A Hyperband sampler records the evaluations of one Hyperoptimizer and cannot be used for several; construct a new sampler for each optimization, or continue an optimization with `ho = ...`."))
    bs = brackets(hb.R, hb.η)
    hb.R < hb.η && @warn "Hyperband with R = $(hb.R) < η = $(hb.η) evaluates a single configuration with the budget R, since successive halving requires R ≥ η"
    ho.iterations = sum(first, bs)
    init_inner!(hb, ho)
    hb.nsampled = 0
    p = Progress(length(bs); dt=1, desc="Hyperband", enabled=SHOW_PROGRESS[])
    for (k, (n, budgets)) in enumerate(bs)
        successive_halving(ho, n, budgets; mapfun)
        ProgressMeter.next!(p; showvalues = progress_values(hb, k, length(bs)))
    end
    return hb.minimum
end

function progress_values(hb::Hyperband, k, nbrackets)
    values = Tuple{String, Any}[("bracket", "$k of $nbrackets")]
    m = hb.minimum
    m === nothing || append!(values, [("minimum", m.loss), ("with budget", m.budget), ("minimizer", m.configuration)])
    values
end

# Prepares the Latin hypercube plan of the inner sampler for a run, whose `ho.iterations` configurations are drawn with
# distinct indices into the plan. The plan contains one point per element of the candidate vectors.
function init_inner!(hb::Hyperband, ho)
    s = hb.inner
    s isa Union{LHSampler,CLHSampler} || return
    nconfigs = ho.iterations
    N = lhs_plan_size(s, ho)
    nconfigs <= N || throw(ArgumentError("The Latin hypercube plan of the inner sampler contains one point per element of the candidate vectors (of the Continuous dimensions for CLHSampler), $N points, but Hyperband with R = $(hb.R) and η = $(hb.η) draws $nconfigs configurations. Provide candidate vectors with at least $nconfigs elements, or use another inner sampler."))
    ho.iterations = N # `init!` creates a plan with `ho.iterations` points
    try
        init!(s, ho)
    finally
        ho.iterations = nconfigs
    end
end

lhs_plan_size(::LHSampler, ho) = maximum(length, ho.candidates)
function lhs_plan_size(s::CLHSampler, ho)
    lengths = [length(c) for (d, c) in zip(s.dims, ho.candidates) if d isa Continuous]
    isempty(lengths) ? ho.iterations : maximum(lengths)
end

"""
    successive_halving(ho, n, r = 1, s = round(Int, log(ho.sampler.η, n)); threads = false, mapfun = threads ? Hyperopt.tmap : map)
    successive_halving(ho, n, budgets::AbstractVector; threads = false, mapfun = threads ? Hyperopt.tmap : map)

Run successive halving with the `Hyperband` sampler `ho.sampler`. `n` configurations are drawn from the inner sampler and
evaluated with the budget `budgets[1]`. The fraction `1/η` of the configurations with the smallest losses is evaluated
again with the budget `budgets[2]`, and so on. The first method uses the budgets `r * η^i` for `i = 0:s`.

`mapfun(f, configurations, states)` evaluates the configurations of a rung and returns the values of `f`, e.g., `map`,
`Hyperopt.tmap` or `Distributed.pmap`. The function `f` captures only `ho.objective` and the budget.

Return `ho.sampler.minimum`.
"""
function successive_halving(ho, n, r::Real=1, s=round(Int, log(ho.sampler.η, n)); kwargs...)
    η = ho.sampler.η
    successive_halving(ho, n, [r*η^i for i in 0:s]; kwargs...)
end

function successive_halving(ho, n, budgets::AbstractVector; threads=false, mapfun=threads ? tmap : map)
    hb = ho.sampler
    objective = ho.objective
    η = hb.η
    C = [hb.inner(ho, hb.nsampled + i) for i in 1:n] # Configurations
    hb.nsampled += n
    T = fill(nothing, n) # States, `nothing` for configurations that have not been evaluated
    for (i, budget) in enumerate(budgets)
        isempty(C) && break
        LT = map(loss_and_state, mapfun((c, t) -> objective(budget, c...; state=t), C, T))
        L, T = first.(LT), last.(LT)
        append!(ho.history, C)
        append!(ho.results, L)
        append!(hb.budgets, fill(budget, length(L)))
        if hb.inner isa BOHB
            update_observations(ho, budget, C, L)
        end
        update_minimum!(hb, budget, L, T, C)
        kept = sortperm(L)[1:floor(Int, n/η^i)]
        C, T = C[kept], T[kept]
    end
    return hb.minimum
end

loss_and_state(loss::Real) = (loss, nothing)
loss_and_state(val::Tuple{Real, Any}) = val
loss_and_state(val) = throw(ArgumentError("The objective of Hyperband must return a real-valued loss, or a tuple `(loss, state)` of a real-valued loss and a state from which a later evaluation can resume. The objective returned a value of type $(typeof(val))."))

# Replaces `hb.minimum` by an evaluation with a larger budget, or with the same budget and a smaller loss, so that it
# refers to the same evaluation as `ho.minimum`
function update_minimum!(hb::Hyperband, budget, L, T, C)
    for k in eachindex(L)
        m = hb.minimum
        if m === nothing || budget > m.budget || (budget == m.budget && compared_loss(L[k]) < compared_loss(m.loss))
            hb.minimum = (loss=L[k], budget=Float64(budget), state=T[k], configuration=C[k])
        end
    end
end

compared_loss(loss) = isnan(loss) ? Inf : loss

# Only the results obtained with the largest budget are compared. Results without a recorded budget, e.g., a history
# that was not recorded by `hyperband`, are all compared.
function compared_indices(ho::Hyperoptimizer{Hyperband})
    budgets = ho.sampler.budgets
    (isempty(budgets) || length(budgets) != length(ho.results)) && return eachindex(ho.results)
    findall(==(maximum(budgets)), budgets)
end

"""
    hyperband(f, candidates; R, η = 3, inner = RandomSampler(), threads = false, mapfun = threads ? Hyperopt.tmap : map)

Run Hyperband without the macro `@hyperopt` and return the `Hyperoptimizer`. `candidates` is a vector or a `NamedTuple`
of candidate vectors, one per hyperparameter. The objective is called as `f(budget, pars)`, where `pars` is the vector
of hyperparameter values in the first evaluation of a configuration, and the state returned by the previous evaluation
of the same configuration otherwise. `f` returns `(loss, state)` or only the loss; in the latter case, `pars` is the
vector of hyperparameter values in every evaluation. `mapfun` evaluates the configurations of a rung, e.g.,
`Distributed.pmap`, see `Hyperopt.successive_halving`. See [`Hyperband`](@ref) for the remaining arguments and the
recorded results.
"""
function hyperband(f, candidates; R, η=3, inner = RandomSampler(), threads=false, mapfun=threads ? tmap : map)
    sampler = Hyperband(; R, η, inner)
    objective(i, pars...; state = nothing) = f(i, state === nothing ? [pars...] : state)
    ho = Hyperoptimizer(;
        iterations = nconfigurations(sampler),
        params = [Symbol("$i") for i in eachindex(candidates)],
        candidates,
        history = [],
        results = Real[],
        sampler,
        objective,
    )
    hyperband(ho; mapfun)
    ho
end

# BOHB ====================================================================
# Acknowledgement: Code structure refers to official implementation of BOHB in ['HpBandSter'](https://github.com/automl/HpBandSter)
# 
# Copyright of HpBandSter: 
# 
#     Copyright (c) 2017-2018, ML4AAD
#     All rights reserved.

# Coordinates in which the densities are estimated. A `Continuous` dimension is represented by the index
# into its sorted distinct candidate values, scaled to [0, 1]; a categorical dimension is represented by
# the index into its candidate vector.
struct ContinuousCoordinate{V<:AbstractVector}
    values::V
end

struct CategoricalCoordinate{V<:AbstractVector}
    values::V
end

nlevels(::ContinuousCoordinate) = 0
nlevels(c::CategoricalCoordinate) = length(c.values)

function coordinate_map(::Continuous, candidates, name)
    all(x -> x isa Real, candidates) ||
        throw(ArgumentError("BOHB: the hyperparameter $name has the dimension type Continuous, which requires real-valued candidates"))
    ContinuousCoordinate(sort!(unique(candidates)))
end

function coordinate_map(dim::Union{Categorical, UnorderedCategorical}, candidates, name)
    dim.levels == length(candidates) ||
        throw(ArgumentError("BOHB: the hyperparameter $name has the dimension type $(nameof(typeof(dim)))($(dim.levels)), but $(length(candidates)) candidates"))
    CategoricalCoordinate(candidates)
end

function coordinate_maps(s, ho)
    length(s.dims) == length(ho.candidates) ||
        throw(ArgumentError("BOHB: `dims` contains $(length(s.dims)) dimension types, but there are $(length(ho.candidates)) hyperparameters"))
    [coordinate_map(dim, collect(c), name) for (dim, c, name) in zip(s.dims, ho.candidates, ho.params)]
end

# Linear interpolation between the coordinates of the neighboring candidates. Values outside the range of the
# candidates are mapped to the boundary of [0, 1].
function to_coordinate(c::ContinuousCoordinate, x::Real)
    v, n = c.values, length(c.values)
    (n == 1 || x <= v[1]) && return 0.0
    x >= v[end] && return 1.0
    k = searchsortedlast(v, x)
    (k - 1 + (x - v[k]) / (v[k+1] - v[k])) / (n - 1)
end

function to_coordinate(c::CategoricalCoordinate, x)
    k = findfirst(isequal(x), c.values)
    k === nothing && throw(ArgumentError("BOHB: the value $(repr(x)) of a categorical hyperparameter is not one of its candidates $(c.values)"))
    Float64(k)
end

nearest_index(c::ContinuousCoordinate, u) = round(Int, u * (length(c.values) - 1)) + 1

# Coordinate of the candidate nearest to the coordinate u
function snap(c::ContinuousCoordinate, u)
    n = length(c.values)
    n == 1 ? 0.0 : (nearest_index(c, u) - 1) / (n - 1)
end

from_coordinate(c::ContinuousCoordinate, u) = c.values[nearest_index(c, u)]
from_coordinate(c::CategoricalCoordinate, k) = c.values[round(Int, k)]

# Kernel density estimate with a product kernel, consisting of Gaussian kernels for continuous dimensions and
# Aitchison-Aitken kernels for categorical dimensions, see Appendix D of the paper. Each column of `data` contains
# an observation in the coordinates of the densities. `levels[j]` is the number of categories of dimension `j`,
# or zero if dimension `j` is continuous.
struct ProductKDE
    data::Matrix{Float64}
    bandwidth::Vector{Float64}
    levels::Vector{Int}
end

# The bandwidths are determined by Scott's rule of thumb and bounded from below by `min_bandwidth`. The bandwidths
# of the categorical dimensions are bounded from above by `(levels[j]-1)/levels[j]`, for which the kernel is
# uniform over the categories.
function fit_kde(data::Matrix{Float64}, levels::Vector{Int}, min_bandwidth::Real)
    d, n = size(data)
    bandwidth = map(1:d) do j
        x = view(data, j, :)
        x̄ = sum(x) / n
        σ = sqrt(sum(xi -> (xi - x̄)^2, x) / n)
        h = max(1.06 * σ * n^(-1 / (4 + d)), min_bandwidth)
        levels[j] == 0 ? h : min(h, (levels[j] - 1) / levels[j])
    end
    ProductKDE(data, bandwidth, levels)
end

function logpdf(kde::ProductKDE, x::AbstractVector)
    d, n = size(kde.data)
    logk = zeros(n)
    for i in 1:n, j in 1:d
        h, c, xij = kde.bandwidth[j], kde.levels[j], kde.data[j, i]
        logk[i] += if c == 0
            -((x[j] - xij) / h)^2 / 2 - log(h) - log(2π) / 2
        else
            x[j] == xij ? log1p(-h) : log(h / (c - 1))
        end
    end
    m = maximum(logk)
    m + log(sum(l -> exp(l - m), logk)) - log(n)
end

# Sample of the normal distribution with mean μ ∈ [0, 1] and standard deviation σ, truncated to [0, 1]
function truncated_normal(rng, μ, σ)
    while true
        u = μ + σ * randn(rng)
        0 <= u <= 1 && return u
    end
end

# Sample of l'(x), the density of good configurations with the bandwidths of the continuous dimensions
# multiplied by bw_factor. Continuous coordinates are rounded to the nearest candidate.
function sample_good(rng, kde::ProductKDE, maps, bw_factor)
    i = rand(rng, 1:size(kde.data, 2))
    map(eachindex(maps)) do j
        μ, h, c = kde.data[j, i], kde.bandwidth[j], kde.levels[j]
        if c == 0
            snap(maps[j], truncated_normal(rng, μ, bw_factor * h))
        elseif rand(rng) < h # One of the other c-1 categories, each with probability h/(c-1)
            k = rand(rng, 1:c-1)
            Float64(k < μ ? k : k + 1)
        else
            μ
        end
    end
end

# Evaluated configuration in the coordinates of the densities, and its loss
struct ObservationsRecord
    x::Vector{Float64}
    loss::Float64
end

"""
    BOHB(; dims, ρ = 1/3, q = 0.15, N_s = 64, N_min = nothing, bw_factor = 3, min_bandwidth = 1e-3, random_sampler = RandomSampler())

Inner sampler of [`Hyperband`](@ref) that implements the model-based sampling of
[BOHB: Robust and Efficient Hyperparameter Optimization at Scale](https://arxiv.org/abs/1807.01774).
A fraction `ρ` of the configurations is sampled at random. The remaining configurations are proposed by
maximizing the ratio l(x)/g(x) of two kernel density estimates, fitted to the best and the worst configurations
evaluated with the largest budget for which sufficiently many observations are available. The observations are the
evaluated configurations, i.e., the values of the hyperparameters, and their losses; the states returned by the
objective are not used. The observations are recorded by Hyperband; `BOHB` is therefore only valid as the inner
sampler of a `Hyperband` sampler, and `@hyperopt` throws an `ArgumentError` for `sampler = BOHB(...)`. A run that
continues an existing `Hyperoptimizer` with `ho = ...` continues the sampler, including its observations.

`dims` contains one dimension type per hyperparameter, in the order of the candidate vectors:
- `Continuous()`: an ordered, real-valued hyperparameter. The densities are estimated in the coordinate given by the
  index into the sorted candidate vector, scaled to [0, 1]. A log-spaced candidate vector thus corresponds to a
  logarithmic coordinate. Ordered discrete hyperparameters, such as integers, may also be declared `Continuous`.
- `Categorical(n)` or `UnorderedCategorical(n)`: a categorical hyperparameter with `n` candidates. Both types are
  treated as unordered and use an Aitchison-Aitken kernel.

Every proposal consists of elements of the candidate vectors.

The remaining keyword arguments refer to the symbols in the paper:
- `ρ`: Fraction of random samples
- `q`: Fraction of the best observations used to fit the density of good configurations l(x)
- `N_s`: Number of samples drawn from l(x), of which the one with the largest ratio l(x)/g(x) is proposed
- `N_min`: Minimum number of observations used to fit each density, the default is the number of hyperparameters plus one
- `bw_factor`: Factor by which the bandwidths of the continuous dimensions are multiplied when sampling from l(x)
- `min_bandwidth`: Lower bound of the bandwidths. This parameter does not occur in the paper but is used in the reference implementation HpBandSter.
- `random_sampler`: `RandomSampler` used for the random configurations. Its RNG is used for all random decisions of
  BOHB, a run is thus reproducible with, e.g., `random_sampler = RandomSampler(MersenneTwister(0))`.

# Example
```julia
ho = @hyperopt for resources = 27, sampler = Hyperband(η = 3, inner = BOHB(dims = [Continuous(), Categorical(2)])), a = LinRange(1, 5, 100), b = [true, false]
    (a - 3)^2 + b + 1/resources
end
```
"""
mutable struct BOHB <: Sampler
    dims::Vector{DimensionType}
    N_min::Union{Int, Nothing}
    ρ::Float64
    q::Float64
    N_s::Int
    bw_factor::Float64
    min_bandwidth::Float64
    random_sampler::RandomSampler
    "Observations for each budget"
    D::Dict{Float64, Vector{ObservationsRecord}}
    "Largest budget for which the densities have been fitted"
    max_valid_budget::Union{Float64, Nothing}
    "Density of good configurations l(x)"
    KDE_good::Union{ProductKDE, Nothing}
    "Density of poor configurations g(x)"
    KDE_bad::Union{ProductKDE, Nothing}
end

function BOHB(; dims, N_min = nothing, ρ = 1/3, q = 0.15, N_s = 64, bw_factor = 3, min_bandwidth = 1e-3, random_sampler = RandomSampler())
    for dim in dims
        dim isa Union{Continuous, Categorical, UnorderedCategorical} ||
            throw(ArgumentError("BOHB supports the dimension types Continuous, Categorical and UnorderedCategorical, got $(typeof(dim))"))
    end
    min_bandwidth > 0 || throw(ArgumentError("BOHB: min_bandwidth must be positive"))
    (N_min === nothing || N_min >= 1) || throw(ArgumentError("BOHB: N_min must be positive"))
    BOHB(dims, N_min, ρ, q, N_s, bw_factor, min_bandwidth, random_sampler, Dict{Float64, Vector{ObservationsRecord}}(), nothing, nothing, nothing)
end

"""
    kde_proposal(s::BOHB, ho)

Return a configuration, a vector of candidate values, that maximizes l(x)/g(x) among `s.N_s` samples of l'(x)
(lines 4-6 of Algorithm 2 in the paper). Return `nothing` with probability `s.ρ`, or if no densities have been fitted.
"""
function kde_proposal(s::BOHB, ho)
    maps = coordinate_maps(s, ho)
    x = with_rng(s.random_sampler) do rng
        (s.KDE_good === nothing || rand(rng) < s.ρ) && return nothing
        samples = [sample_good(rng, s.KDE_good, maps, s.bw_factor) for _ in 1:s.N_s]
        argmax(x -> logpdf(s.KDE_good, x) - logpdf(s.KDE_bad, x), samples)
    end
    x === nothing ? nothing : [from_coordinate(m, xj) for (m, xj) in zip(maps, x)]
end

function (s::BOHB)(ho, iter)
    proposal = kde_proposal(s, ho)
    proposal === nothing && return s.random_sampler(ho, iter)
    proposal
end

# Update budget observations in BOHB
function update_observations(ho::Hyperoptimizer{Hyperband}, rᵢ, observations, losses)
    bohb = ho.sampler.inner
    maps = coordinate_maps(bohb, ho)
    budget = Float64(rᵢ)
    for b in keys(bohb.D) # Budgets that differ only by rounding errors are merged
        b ≈ budget && (budget = b)
    end
    records = get!(Vector{ObservationsRecord}, bohb.D, budget)
    for (config, loss) in zip(observations, losses)
        length(config) == length(maps) ||
            throw(ArgumentError("BOHB: an evaluated configuration $(repr(config)) does not contain one value for each of the $(length(maps)) hyperparameters"))
        x = [to_coordinate(m, v) for (m, v) in zip(maps, config)]
        push!(records, ObservationsRecord(x, isfinite(loss) ? loss : Inf))
    end
    N_min = something(bohb.N_min, length(maps) + 1)
    # Line 2 of Algorithm 2 in the paper
    if length(records) >= N_min + 2 && (bohb.max_valid_budget === nothing || budget >= bohb.max_valid_budget)
        bohb.max_valid_budget = budget
        update_KDEs(bohb, records, N_min, nlevels.(maps))
    end
end

"""
    good, bad = good_bad_split(losses, N_min, q)

Indices of the `N_l = max(N_min, ⌊q N_b⌋)` best and the `max(N_min, N_b - N_l)` worst of the `N_b = length(losses)`
observations, used to fit the densities of good and poor configurations l(x) and g(x), see Eq. (3) of the paper.
The two sets are disjoint if `N_b ≥ N_l + N_min`.
"""
function good_bad_split(losses, N_min, q)
    N_b = length(losses)
    N_l = max(N_min, floor(Int, q * N_b))
    N_g = max(N_min, N_b - N_l)
    idx = sortperm(losses)
    idx[1:N_l], idx[end-N_g+1:end]
end

function update_KDEs(bohb::BOHB, records, N_min, levels)
    good, bad = good_bad_split([r.loss for r in records], N_min, bohb.q)
    bohb.KDE_good = fit_kde(stack(records[i].x for i in good), levels, bohb.min_bandwidth)
    bohb.KDE_bad = fit_kde(stack(records[i].x for i in bad), levels, bohb.min_bandwidth)
end
