"""
    RandomSampler{T<:AbstractRNG} <: Sampler

Sample a value for each parameter uniformly at random from the candidate vectors. Log-uniform sampling available by providing a log-spaced candidate vector.

Optionally, pass an `AbstractRNG` to initialize.
"""
struct RandomSampler{T<:AbstractRNG} <: Sampler
    rng_channel::RemoteChannel{Channel{T}}
    lock::ReentrantLock
end

RandomSampler() = RandomSampler(MersenneTwister(rand(1:1000)))

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
    if log(η,R) ≈ ceil(log(η, R)        # Catch machine precision error for e.g. 
        smax = ceil(Int, log(η, R)      # η=3 and R ∈ (3 .^ [5, 10, 13, 15, 17, 20, 23, 26, 27]), or
    else                                # η=9 and R ∈ (9 .^ [5, 10, 13, 15, 17])
        smax = floor(Int, log(η,R))
    end
    
    B = (smax + 1)*R # B is budget
    
    p = Progress(smax+1, 1, "Hyperband")
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
    T = [ hb.inner(ho, i) for i=1:n ]
    mapfun = threads ? ThreadPools.tmap : map
    for i in 0:s
        nᵢ = n/(η^i) # Flooring takes place below 
        rᵢ = r*(η^i) 
        if i == 0
            LTend = mapfun(t->costfun(rᵢ, t...), T)
        else
            LTend = mapfun(t->costfun(rᵢ, t), T)
        end
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
        T = T[perm[1:floor(Int,nᵢ/η)]]
        Base.CoreLogging.@logmsg -1 "successive_halving" progress=i/s
    end
    return minimum
end


function hyperband(f, candidates; R, η=3, inner = RandomSampler(), threads=false)
    sampler = Hyperband(; R, η, inner)
    objective(i, pars...) = f(i, [pars...]) # The objective needs two methods, accepting a vector and a list of args
    objective(i, state) = f(i, state) 
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

# struct to record BOHB Observation
mutable struct ObservationsRecord
    dim::Int
    observation::Union{Vector, Tuple}
    loss::Real
end
function ObservationsRecord(observation, loss)
    ObservationsRecord(length(observation), observation, loss)
end

"""
    BOHB samplers
All variable names refer symbols in the [paper](`https://arxiv.org/pdf/1807.01774v1.pdf`)
- `ρ`: Fraction of random samples
- `q`: Fraction of best observations to build l and g 
- `N_s`: Sample batch number
- `N_min`: Minimum number of points to build a model
- `bw_factor`: Bandwidth factor
- `D`: Evaluated observations
- `max_valid_budget`: Maximum budget i that |D_{i}| is big enough to fit a model
- `N_b`: |D_{max_valid_budget}|
- `KDE_good`: KDE consists of "good observations", see BOHB paper
- `KDE_bad`: KDE consists of "bad observations", see BOHB paper
"""
Base.@kwdef mutable struct BOHB <: Sampler
    dims::Union{Vector{DimensionType}, Nothing}=nothing
    # hyperparameters for BOHB
    N_min::Union{Int, Nothing} = nothing
    ρ::AbstractFloat = 1/3
    q::AbstractFloat = 0.15
    N_s::Int = 64
    bw_factor::Real = 3
    "minimum bandwidth: this parameter doesn't occur in the paper but used in the official implementation"
    min_bandwidth::Real = 1e-3
    "Random sampler used for random sampling in BOHB algorithm"
    random_sampler::RandomSampler = RandomSampler()
    # Context data
    ## Current observations, stored in a Dict, in which key is budget, value is a observation array to fit KDEs
    ## key of D: A real number represents budget
    ## value of D: An vector of ObservationsRecord, all the records of corresponding budget
    D::Dict{Real, Vector{ObservationsRecord}} = Dict{Real, Vector{ObservationsRecord}}()
    "Current maximum budget that |D_{b}| > N_{min}+2, means it is valid for fit KDEs"
    max_valid_budget::Union{Number, Nothing} = nothing
    "|D| of max_valid_budget"
    N_b::Union{Int, Nothing} = nothing
    ## Good and bad kernel density estimator
    KDE_good::Union{MultiKDE.KDEMulti, Nothing} = nothing
    KDE_bad::Union{MultiKDE.KDEMulti, Nothing} = nothing
end

# object call of BOHB sampler
function (s::BOHB)(ho, iter)
    # with probability ρ, return random sampled observations. 
    # If max_valid_budget is nothing, which means currently we don't have enough sample for TPE, random sample as well. 
    if rand() < s.ρ || s.max_valid_budget === nothing
        return s.random_sampler(ho, iter)
    end
    potential_samples = [sample_potential_hyperparam(s.KDE_good, s.min_bandwidth, s.bw_factor) for _ in 1:s.N_s]
    scores = [score(sample, s.KDE_good, s.KDE_bad) for sample in potential_samples]
    _, best_idx = findmax(scores)
    [potential_samples[best_idx]]
end

# Sample score l(x)/g(x), refers to line 6 of Algorithm2 in paper
function score(sample::Vector, KDE_good::MultiKDE.KDEMulti, KDE_bad::MultiKDE.KDEMulti)
    pdf(KDE_good, sample) / pdf(KDE_bad, sample)
end

# Update budget observations in BOHB
function update_observations(ho::Hyperoptimizer{Hyperband}, rᵢ, observations, losses)
    bohb = ho.sampler.inner
    if !haskey(bohb.D, rᵢ)
        bohb.D[rᵢ] = []
    end
    for (c, l) in zip(observations, losses)
        push!(bohb.D[rᵢ], ObservationsRecord(c, l))
    end
    D_length = length(bohb.D[rᵢ])
    if bohb.N_min === nothing
        bohb.N_min = length(ho.candidates)+1
    end
    if D_length > bohb.N_min+2 && (bohb.max_valid_budget===nothing || rᵢ >= bohb.max_valid_budget)
        bohb.max_valid_budget, bohb.N_b = rᵢ, D_length
        update_KDEs(ho)
    end
end

function update_KDEs(ho::Hyperoptimizer{Hyperband})
    bohb = ho.sampler.inner
    records = bohb.D[bohb.max_valid_budget]
    # fit KDEs according to Eqs. (2) and (3) in paper
    N_bl = max(bohb.N_min, floor(Int, bohb.q*bohb.N_b))
    N_bg = max(bohb.N_min, bohb.N_b-N_bl)
    sort_idx = sortperm(records, by=d->d.loss)
    idx_N_bl = sort_idx[begin:N_bl]
    idx_N_bg = reverse(sort_idx)[N_bg:end]
    bohb.KDE_good = KDEMulti(bohb.dims, records[idx_N_bl], bohb.min_bandwidth, ho.candidates) 
    bohb.KDE_bad = KDEMulti(bohb.dims, records[idx_N_bg], bohb.min_bandwidth, ho.candidates)
end

# sample from KDEMulti
function sample_potential_hyperparam(kde::MultiKDE.KDEMulti, min_bandwidth, bw_factor)
    idx = rand(1:size(kde.mat_observations)[2])
    param = [kde.observations[i][idx] for i in 1:length(kde.observations)]
    sample = Vector()
    for (_i, _param, dim_type, _kde) in zip(1:length(kde.dims), param, kde.dims, kde.KDEs)
        bw = max(_kde.bandwidth, min_bandwidth)
        local ele
        if dim_type isa MultiKDE.ContinuousDim
            bw = bw*bw_factor
            ele = rand(truncated(Normal(_param, bw), -_param/bw, (1-_param)/bw))
        elseif dim_type isa Union{MultiKDE.CategoricalDim, MultiKDE.UnorderedCategoricalDim}
            ele = rand() < (1-bw) ? _param : rand(1:dim_type.levels)
        else
            error(string("Dim type ", string(dim_type), " not supported. "))
        end
        if kde.mapped[_i]
            ele = kde.index_to_unordered[_kde][ele]
        end
        push!(sample, ele)
    end
    sample
end

# Constructor extensions and adapters for MultiKDE.jl
const DIMENSION_TYPE = Dict(Categorical=>MultiKDE.CategoricalDim, Continuous=>MultiKDE.ContinuousDim, UnorderedCategorical=>MultiKDE.UnorderedCategoricalDim)

function MultiKDE.KDEMulti(dim_types::Vector{DimensionType}, records::Vector{ObservationsRecord}, min_bandwidth::Real, candidates::Tuple)
    # Get KDEMulti with min_bandwidth
    dim = records[1].dim
    observations = Vector{Vector}()
    for record in records
        @assert record.dim == dim "All observations need to be same dimension. "
        _observations = record.observation
        if _observations isa Tuple
            _observations = [_obs for _obs in _observations]
        end
        push!(observations, _observations)
    end
    multi_kde = KDEMulti(dim_types, observations, candidates)
    for i in 1:length(multi_kde.KDEs)
        multi_kde.KDEs[i].bandwidth = max(multi_kde.KDEs[i].bandwidth, min_bandwidth)
    end
    multi_kde
end

function MultiKDE.KDEMulti(dims::Vector{DimensionType}, observations::Vector, candidates::Tuple)
    dims = Vector{MultiKDE.DimensionType}([DIMENSION_TYPE[typeof(dim)] === MultiKDE.ContinuousDim ? DIMENSION_TYPE[typeof(dim)]() : 
                                            DIMENSION_TYPE[typeof(dim)](dim.levels) for dim in dims])
    MultiKDE.KDEMulti(dims, observations, candidates)
end

function Base.getproperty(dim_type::Union{MultiKDE.CategoricalDim, MultiKDE.UnorderedCategoricalDim}, v::Symbol)
    if v === :levels
        return getfield(dim_type, :level)
    end
    getfield(dim_type, v)
end
