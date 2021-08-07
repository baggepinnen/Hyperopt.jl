    """
Sample a value For each parameter uniformly at random from the candidate vectors. Log-uniform sampling available by providing a log-spaced candidate vector.
"""
struct RandomSampler <: Sampler end

function (s::RandomSampler)(ho, iter)
    [list[rand(HO_RNG[threadid()], 1:length(list))] for list in ho.candidates]
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
        throw(ArgumentError("Latin hypercube sampling requires all candidate vectors to have the same length as the number of iterations, got lengths $(repr.(collect(zip(ho.params, length.(ho.candidates))))) with $(ho.iterations) iterations"))
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
    inner = RandomSampler()
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
    R,η = hb.R, hb.η
    hb.minimum = (Inf,)
    smax = floor(Int, log(η,R))
    B = (smax + 1)*R # B is budget
    Juno.progress() do id
        for s in smax:-1:0
            n = ceil(Int, (B/R)*((η^s)/(s+1)))
            r = R / (η^s)
            minᵢ = successive_halving(ho, n, r, s; threads)
            if minᵢ[1] < hb.minimum[1]
                hb.minimum = minᵢ
            end
            Base.CoreLogging.@logmsg -1 "Hyperband" progress=(smax-s)+1/(smax+1)  _id=id
        end
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
    Juno.progress() do id
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

            # Find top K candidates
            perm = sortperm(L)
            besti = perm[1]
            if L[besti] < minimum[1]
                minimum = (L[besti], rᵢ, T[besti])
            end
            T = T[perm[1:floor(Int,nᵢ/η)]]
            Base.CoreLogging.@logmsg -1 "successive_halving" progress=i/s  _id=id
        end
    end
    return minimum
end


function hyperband(f, candidates; R, η=3, inner = RandomSampler(), threads=false)
    sampler = Hyperband(; R, η, inner)
    objective(i, pars...) = f(i, [pars...]) # The objective needs two methods, accepting a vector and a list of args
    objective(i, pars) = f(i, pars) 
    ho = Hyperoptimizer(;
        iterations = 1,
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