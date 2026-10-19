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
    LHSampler(; rng = Xoshiro(rand(UInt)), steps = nothing)

Sample from an optimized Latin hypercube plan. The plan contains one point per iteration, and every element of each candidate vector occurs in exactly one point. All candidate vectors must therefore have the same length as the number of iterations. [`CLHSampler`](@ref) admits categorical hyperparameters with fewer candidates.

The plan is obtained from a random Latin hypercube by `steps` proposed exchanges of the entries of two points in one dimension. An exchange is accepted if it reduces the Audze-Eglais potential energy of the plan, i.e., the sum of the inverse squared distances between all pairs of points, in the coordinates given by the candidate indices divided by the number of iterations. The computational cost of a step is proportional to `n*d` for `n` iterations and `d` hyperparameters. The default number of steps, `clamp(10^8 ÷ (n*d), n*d, 100*n*d)`, thus limits the cost to approximately `10^8` evaluations of coordinate differences if `n*d ≤ 10^4`. With `steps = 0`, the plan is a random Latin hypercube.

The plan is drawn from `rng`. By default, a `Xoshiro` generator is seeded with a random number drawn from the global RNG, so that `Random.seed!` makes the plans reproducible.

A new plan is generated for every run of `@hyperopt`, including a run that resumes an existing `Hyperoptimizer` with `ho = ...`, and for every iteration over a `Hyperoptimizer` that begins after results were recorded. The optimization of a new plan reduces the number of its points that coincide with configurations in `ho.history` with priority over the potential energy: an exchange that reduces this number is accepted, and an exchange that increases it is rejected. Configurations are thus not evaluated again if the number of steps suffices. With a single hyperparameter, however, every plan consists of all candidates.
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
- `Categorical(k)` or `UnorderedCategorical(k)`: a categorical hyperparameter with `k` candidates. In a plan with `n` points, every candidate occurs in `n ÷ k` or `n ÷ k + 1` points. Both types are treated as unordered: a categorical dimension contributes 1 to the squared distance between two points with different candidates and 0 otherwise. The optimized plan thus distributes the points that share a categorical candidate over the continuous dimensions.

The keyword arguments `rng` and `steps`, and the generation of new plans are described in [`LHSampler`](@ref).
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

"""
Base.@kwdef mutable struct Hyperband <: Sampler
    R
    η = 3
    minimum = (Inf,)
    inner::Sampler = RandomSampler()
end
Hyperband(R) = Hyperband(R=R)

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

function hyperband(ho::Hyperoptimizer{Hyperband}; threads=false)
    hb = ho.sampler
    R, η = hb.R, hb.η
    hb.minimum = (Inf,)
    if log(η,R) ≈ ceil(log(η, R))       # Catch machine precision error for e.g. 
        smax = ceil(Int, log(η, R))     # η=3 and R ∈ (3 .^ [5, 10, 13, 15, 17, 20, 23, 26, 27]), or
    else                                # η=9 and R ∈ (9 .^ [5, 10, 13, 15, 17])
        smax = floor(Int, log(η,R))
    end
    
    B = (smax + 1)*R # B is budget
    
    p = Progress(smax+1; dt=1, desc="Hyperband", enabled=SHOW_PROGRESS[])
    for s in smax:-1:0
        n = ceil(Int, (B/R)*((η^s)/(s+1)))
        r = R / (η^s)
        minᵢ = successive_halving(ho, n, r, s; threads)
        if minᵢ[1] < hb.minimum[1]
            hb.minimum = minᵢ
        end
        ProgressMeter.next!(p; showvalues = [("bracket (of $(smax+1))",smax-s+1), ("minimum", hb.minimum[1]), ("with resources", hb.minimum[2]), ("minizer", hb.minimum[3])])
    end
    return hb.minimum
end


function successive_halving(ho, n, r=1, s=round(Int, log(hb.η, n)); threads=false)
    hb = ho.sampler
    costfun = ho.objective
    η = hb.η
    minimum = Inf
    C = [ hb.inner(ho, i) for i=1:n ] # Configurations
    T = fill(nothing, n) # States, `nothing` for configurations that have not been evaluated
    mapfun = threads ? ThreadPools.tmap : map
    for i in 0:s
        nᵢ = n/(η^i) # Flooring takes place below
        rᵢ = r*(η^i)
        LTend = mapfun((c, t)->costfun(rᵢ, c...; state=t), C, T)
        L, T = first.(LTend), last.(LTend)
        append!(ho.history, T)
        append!(ho.results, L)
        if hb.inner isa BOHB
            update_observations(ho, rᵢ, T, L)
        end
        perm = sortperm(L)
        besti = perm[1]
        if L[besti] < minimum[1]
            minimum = (L[besti], rᵢ, T[besti])
        end
        kept = perm[1:floor(Int,nᵢ/η)]
        C, T = C[kept], T[kept]
    end
    return minimum
end


function hyperband(f, candidates; R, η=3, inner = RandomSampler(), threads=false)
    sampler = Hyperband(; R, η, inner)
    objective(i, pars...; state = nothing) = f(i, state === nothing ? [pars...] : state)
    ho = Hyperoptimizer(;
        iterations = 1,
        params = [Symbol("$i") for i in eachindex(candidates)],
        candidates,
        history = [],
        results = Real[],
        sampler,
        objective,
    )
    if inner isa Union{LHSampler,CLHSampler}
        ho.iterations = length(candidates[1])
        init!(inner, ho)
    end
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
evaluated with the largest budget for which sufficiently many observations are available.

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
