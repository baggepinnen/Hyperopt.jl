"""
    RandomSampler{T<:AbstractRNG} <: Sampler

Sample a value for each parameter uniformly at random from the candidate vectors. Log-uniform sampling available by providing a log-spaced candidate vector.

Optionally, pass an `AbstractRNG` to initialize. By default, a `Xoshiro` generator is seeded with a random number drawn from the global RNG, so that `Random.seed!` makes the samples reproducible.
"""
struct RandomSampler{T<:AbstractRNG} <: Sampler
    rng_channel::RemoteChannel{Channel{T}}
    lock::ReentrantLock
end

RandomSampler() = RandomSampler(Xoshiro(rand(UInt)))

function RandomSampler(rng::T) where {T <: AbstractRNG}
    # We use a *single* RNG across threads and processors to ensure
    # that e.g. different processors don't have different copies of the
    # RNG.
    channel = RemoteChannel(()->Channel{T}(1), 1)
    put!(channel, rng)
    return RandomSampler(channel, ReentrantLock())
end

function (s::RandomSampler)(ho, iter)
    # Lock first, since RemoteChannel's may not be threadsafe:
    # <https://github.com/JuliaLang/julia/issues/37706>
    return lock(s.lock) do
        rng = take!(s.rng_channel)
        result = [list[rand(rng, 1:length(list))] for list in ho.candidates]
        put!(s.rng_channel, rng)
        return result
    end
end


# Latin Hypercube Sampler ======================================================

"""
Sample from a latin hypercube
"""
Base.@kwdef mutable struct LHSampler <: Sampler
    samples = zeros(0,0)
    iters = -1
end
function init!(s::LHSampler, ho)
    s.samples != zeros(0,0) && return # Already initialized
    ndims = length(ho.candidates)
    all(length(c) == ho.iterations for c in ho.candidates) ||
        throw(ArgumentError("Latin hypercube sampling requires all candidate vectors to have the same length as the number of iterations, got lengths $(repr.(collect(zip(ho.params, [length(c) for c in ho.candidates]))))) with $(ho.iterations) iterations"))
        s.iters == -1 && (s.iters = (1000*100*2)÷ho.iterations÷ndims)
    X, fit = LHCoptim(ho.iterations,ndims,s.iters)
    s.samples = copy(X')
end

function (s::LHSampler)(ho, iter)
    [list[s.samples[dim,iter]] for (dim,list) in enumerate(ho.candidates)]
end

"""
    CLHSampler(dims=[Continuous(), Categorical(2), ...])
Sample from a categorical/continuous latin hypercube. All continuous variables must have the same length of the candidate vectors.
"""
Base.@kwdef mutable struct CLHSampler <: Sampler
    samples = zeros(0,0)
    dims = []
end
function init!(s::CLHSampler, ho)
    s.samples != zeros(0,0) && return # Already initialized
    all(zip(s.dims, ho.candidates)) do (d,c)
        d isa Categorical || length(c) == ho.iterations
    end || throw(ArgumentError("Latin hypercube sampling requires all candidate vectors for Continuous variables to have the same length as the number of iterations, got lengths $(repr.(collect(zip(ho.params, length.(ho.candidates)))))"))
    ndims = length(ho.candidates)
    initialSample = randomLHC(ho.iterations,s.dims)
    X,_ = LHCoptim!(initialSample, 500, dims=s.dims)
    s.samples = copy(X')
end

function (s::CLHSampler)(ho, iter)
    [list[s.samples[dim,iter]] for (dim,list) in enumerate(ho.candidates)]
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

The evaluations are recorded in `ho.history` (the configurations), `ho.results` (the losses) and `ho.sampler.budgets`
(the budgets). `ho.minimum`, `ho.minimizer`, `ho.maximum` and `ho.maximizer` compare only the results obtained with the
largest budget. `ho.sampler.minimum` is `nothing` before the first evaluation, and otherwise the named tuple
`(; loss, budget, state, configuration)` of the evaluation that `ho.minimum` refers to. `plot(ho)` shows the results
of each budget as a separate series. `ho.iterations` is the number of configurations that one run of Hyperband draws
from `inner`. A `Hyperband` sampler records the evaluations of one `Hyperoptimizer` and cannot be used for several.

With `inner = LHSampler()` or `CLHSampler(...)`, the configurations of all brackets, and of continued runs with
`ho = ...`, are distinct points of one Latin hypercube plan. The plan contains one point per element of the candidate
vectors, which must therefore contain at least as many elements as configurations are drawn.

See also [`hyperband`](@ref) for an interface without the macro.
"""
Base.@kwdef mutable struct Hyperband <: Sampler
    R = nothing
    η = 3
    inner::Sampler = RandomSampler()
    budgets::Vector{Float64} = Float64[]
    minimum::Union{Nothing, NamedTuple} = nothing
    # Number of configurations drawn from `inner`, the next configuration is drawn with the index `nsampled + 1`
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

function optimize(ho::Hyperoptimizer{<:Hyperband})
    try
        hyperband(ho)
    catch e
        if e isa InterruptException
            @info "Aborting hyperband"
        else
            rethrow(e)
        end
    end
    ho
end

"""
    hyperband(ho::Hyperoptimizer{Hyperband}; threads = false, mapfun = threads ? ThreadPools.tmap : map)

Run Hyperband with the sampler `ho.sampler` and the objective `ho.objective`, which is called as
`ho.objective(budget, configuration...; state)` and returns `(loss, state)` or only the loss, see [`Hyperband`](@ref).
The evaluations are appended to `ho.history`, `ho.results` and `ho.sampler.budgets`. `mapfun` evaluates the
configurations of a rung, see `Hyperopt.successive_halving`.

Return `ho.sampler.minimum`.
"""
function hyperband(ho::Hyperoptimizer{Hyperband}; threads=false, mapfun=threads ? ThreadPools.tmap : map)
    hb = ho.sampler
    hb.R === nothing && throw(ArgumentError("Hyperband: the maximum budget R is not specified"))
    length(hb.budgets) == length(ho.results) || throw(ArgumentError("The Hyperband sampler has recorded the budgets of $(length(hb.budgets)) evaluations, but the Hyperoptimizer contains $(length(ho.results)) results. A Hyperband sampler records the evaluations of one Hyperoptimizer and cannot be used for several; construct a new sampler for each optimization, or continue an optimization with `ho = ...`."))
    bs = brackets(hb.R, hb.η)
    ho.iterations = sum(first, bs)
    init_inner!(hb, ho)
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

# Initializes the Latin hypercube plan of the inner sampler, and checks that the plan contains a point that has not been
# drawn before for each configuration of the run
function init_inner!(hb::Hyperband, ho)
    s = hb.inner
    s isa Union{LHSampler,CLHSampler} || return
    nconfigs = ho.iterations
    ho.iterations = lhs_plan_size(s, ho) # `init!` creates a plan with `ho.iterations` points
    try
        init!(s, ho)
    finally
        ho.iterations = nconfigs
    end
    N = size(s.samples, 2)
    hb.nsampled + nconfigs <= N || throw(ArgumentError("The Latin hypercube plan of the inner sampler contains $N points, but Hyperband with R = $(hb.R) and η = $(hb.η) draws $nconfigs configurations$(hb.nsampled > 0 ? " in addition to the $(hb.nsampled) configurations drawn before" : ""). The plan contains one point per element of the candidate vectors (of the Continuous dimensions for CLHSampler); provide candidate vectors with at least $(hb.nsampled + nconfigs) elements, or use another inner sampler."))
end

lhs_plan_size(::LHSampler, ho) = maximum(length, ho.candidates)
function lhs_plan_size(s::CLHSampler, ho)
    lengths = [length(c) for (d, c) in zip(s.dims, ho.candidates) if !(d isa Categorical)]
    isempty(lengths) ? ho.iterations : maximum(lengths)
end

"""
    successive_halving(ho, n, r = 1, s = round(Int, log(ho.sampler.η, n)); threads = false, mapfun = threads ? ThreadPools.tmap : map)
    successive_halving(ho, n, budgets::AbstractVector; threads = false, mapfun = threads ? ThreadPools.tmap : map)

Run successive halving with the `Hyperband` sampler `ho.sampler`. `n` configurations are drawn from the inner sampler and
evaluated with the budget `budgets[1]`. The fraction `1/η` of the configurations with the smallest losses is evaluated
again with the budget `budgets[2]`, and so on. The first method uses the budgets `r * η^i` for `i = 0:s`.

`mapfun(f, configurations, states)` evaluates the configurations of a rung and returns the values of `f`, e.g., `map`,
`ThreadPools.tmap` or `Distributed.pmap`. The function `f` captures only `ho.objective` and the budget.

Return `ho.sampler.minimum`.
"""
function successive_halving(ho, n, r::Real=1, s=round(Int, log(ho.sampler.η, n)); kwargs...)
    η = ho.sampler.η
    successive_halving(ho, n, [r*η^i for i in 0:s]; kwargs...)
end

function successive_halving(ho, n, budgets::AbstractVector; threads=false, mapfun=threads ? ThreadPools.tmap : map)
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
    hyperband(f, candidates; R, η = 3, inner = RandomSampler(), threads = false)

Run Hyperband without the macro `@hyperopt` and return the `Hyperoptimizer`. `candidates` is a vector or a `NamedTuple`
of candidate vectors, one per hyperparameter. The objective is called as `f(budget, pars)`, where `pars` is the vector
of hyperparameter values in the first evaluation of a configuration, and the state returned by the previous evaluation
of the same configuration otherwise. `f` returns `(loss, state)` or only the loss; in the latter case, `pars` is the
vector of hyperparameter values in every evaluation. See [`Hyperband`](@ref) for the remaining arguments and the
recorded results.
"""
function hyperband(f, candidates; R, η=3, inner = RandomSampler(), threads=false)
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
    hyperband(ho; threads)
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
objective are not used.

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

# Calls f with the RNG of the random sampler
function with_rng(f, s::RandomSampler)
    lock(s.lock) do
        rng = take!(s.rng_channel)
        try
            f(rng)
        finally
            put!(s.rng_channel, rng)
        end
    end
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
