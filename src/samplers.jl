"""
Sample a value For each parameter uniformly at random from the candidate vectors. Log-uniform sampling available by providing a log-spaced candidate vector.
"""
struct RandomSampler <: Sampler end
function (s::RandomSampler)(ho)
    [list[rand(1:length(list))] for list in ho.candidates]
end


# Blue noise ===================================================================

"""
Try to spread out parameters as blue noise (only high frequency). Should sample the space better than random sampling. Use this if you intend to run less than, say, 2000.
"""
@with_kw mutable struct BlueNoiseSampler <: Sampler
    samples = zeros(0,0)
    orders = Int[]
end
function init!(s::BlueNoiseSampler, ho)
    s.samples != zeros(0,0) && return # Already initialized
    dims = length(ho.candidates)
    all(length(c) == ho.iterations for c in ho.candidates) ||
        throw(ArgumentError("BlueNoiseSampler requires all candidate vectors to have the same length as the number of iterations, got lengths $(repr.(collect(zip(ho.params, length.(ho.candidates))))) with $(ho.iterations) iterations"))
    s.samples = bluenoise(dims = dims, nsamples = ho.iterations)
    s.orders = [sortperm(s.samples[dim,:]) for dim = 1:dims]
end

function (s::BlueNoiseSampler)(ho)
    init!(s, ho)
    iter = length(ho.history)+1
    [list[s.orders[dim][iter]] for (dim,list) in enumerate(ho.candidates)]
end

function bluenoise(;
    dims,
    nsamples,
    iters = 100,
    points = rand(dims, nsamples),
    α = 0.005
    )

    n          = nsamples = size(points,2)
    dims       = size(points,1)
    rdmax      = [1/(2n), sqrt(1/(2sqrt(3)*n)), (1/(4sqrt(2)*n))^(1/3), (1/(8n))^(1/4)]
    w          = 1 ./ rdmax
    w        ./= sum(w)
    fadelength = iters ÷ 2
    fadein(i)  = min(i,fadelength)*w[1]/fadelength
    points2    = copy(points)
    grad       = zeros(dims)
    d          = zeros(dims)
    for i = 1:iters
        i == iters ÷ 2 && (α *= 0.2)
        perms = [sortperm(points[dim,:]) for dim = 1:dims]
        for p1 = 1:nsamples
            grad .*= 0
            for dim = 1:dims
                optimal = (-0.5 + perms[dim][p1])/n
                grad[dim] += fadein(i)*sign(optimal - points[dim,p1])
            end
            for p2 = 1:nsamples
                d .= points[:,p1] .- points[:,p2]
                nd = norm(d)
                # nd > 0.2 && continue
                grad .+= w[dims]*d./n/(0.001 + nd)^3
            end
            points2[:,p1] .+= α*grad
        end
        clamp!(points2,0.,1.)
        copyto!(points, points2)
    end
    points
end


# Latin Hypercube Sampler ======================================================

"""
Sample from a latin hypercube
"""
@with_kw mutable struct LHSampler <: Sampler
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

function (s::LHSampler)(ho)
    init!(s, ho)
    iter = length(ho.history)+1
    [list[s.samples[dim,iter]] for (dim,list) in enumerate(ho.candidates)]
end

"""
    CLHSampler(dims=[Continuous(), Categorical(2), ...])
Sample from a categorical/continuous latin hypercube. All continuous variables must have the same length of the candidate vectors.
"""
@with_kw mutable struct CLHSampler <: Sampler
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

function (s::CLHSampler)(ho)
    init!(s, ho)
    iter = length(ho.history)+1
    [list[s.samples[dim,iter]] for (dim,list) in enumerate(ho.candidates)]
end





# GaussianProcesses sampler ====================================================

"""
Sample using Bayesian optimization. `GPSampler(Min)/GPSampler(Max)` fits a Gaussian process to the data and tries to use this model to figure out where the best point to sample next is (using expected improvement). Underneath, the package [BayesianOptimization.jl](https://github.com/jbrea/BayesianOptimization.jl/) is used. We try to provide reasonable defaults for the underlying model and optimizer and we do not provide any customization options for this sampler. If you want advanced control, use BayesianOptimization.jl directly.
"""
@with_kw mutable struct GPSampler <: Sampler
    sense
    model = nothing
    # opt = nothing
    modeloptimizer = nothing
    logdims = nothing
    candidates = nothing
end
GPSampler() = error("The GPSampler needs to know if you want to maximize or minimize. Choose between `GPSampler(Max)/GPSampler(Min)`")
GPSampler(sense) = GPSampler(sense=sense)

function islogspace(x)
    all(x->x > 0, x) || return false
    length(x) > 4    || return false
    std(diff(log.(x))) < sqrt(eps(eltype(x)))
end

function to_logspace(x,logdims)
    map(x,logdims) do x,l
        l ? log10.(x) : x
    end
end

function from_logspace(x,logdims)
    map(x,logdims) do x,l
        l ? exp10.(x) : x
    end
end

function init!(s::GPSampler, ho)
    s.model === nothing || return # Already initialized
    ndims = length(ho.candidates)
    logdims = islogspace.(ho.candidates)
    candidates = to_logspace(ho.candidates, logdims)
    lower_bounds = [minimum.(candidates)...]
    upper_bounds = [maximum.(candidates)...]
    widths = upper_bounds - lower_bounds
    kernel_widths = widths/10
    log_function_noise = 0.
    hypertuning_interval = max(9, ho.iterations ÷ 9)
    model = ElasticGPE(ndims, mean = MeanConst(0.),
                       kernel = SEArd(log.(kernel_widths), log_function_noise), logNoise = 0.)
    modeloptimizer = MLGPOptimizer(every = hypertuning_interval, noisebounds = [-5, 0], # log bounds on the function noise?
                                    maxeval = 40) # max iters for optimization of the GP hyperparams
    # opt = BOpt(f, model, ExpectedImprovement(),
    #            modeloptimizer, lower_bounds, upper_bounds,
    #            maxiterations = ho.iterations, sense = Min, repetitions = 1,
    #            acquisitionoptions = (maxeval = 4000, restarts = 50),
    #            verbosity = Progress)
    # result = boptimize!(opt)

    # s.opt = opt
    s.model = model
    s.modeloptimizer = modeloptimizer
    s.logdims = logdims
    s.candidates = candidates
end

function (s::GPSampler)(ho)
    init!(s, ho)
    iter = length(ho.history)+1
    if iter == 1
        return [rand(list) for (dim,list) in enumerate(ho.candidates)]
    else
        input = reshape(to_logspace(float.(ho.history[end]), s.logdims), :, 1)
        try
            BayesianOptimization.update!(s.model, input, [Int(s.sense)*ho.results[end]])
            BayesianOptimization.optimizemodel!(s.modeloptimizer, s.model) # This determines whether to run or not internally?
        catch ex
            @warn("BayesianOptimization failed at iter $iter: error: ", ex)
        end
    end

    acqfunc = BayesianOptimization.acquisitionfunction(ExpectedImprovement(Int(s.sense)*maximum(s.model.y)), s.model)
    ho2 = Hyperoptimizer(iterations=min(1000, prod(length, s.candidates)), params=ho.params, candidates=s.candidates)
    for params in ho2
        params2 = [params[2:end]...]
        res = -Inf
        try
            res = acqfunc(params2)
        catch ex
            @warn("BayesianOptimization acqfunc failed at iter $iter: error: ", ex)
        end
        push!(ho2.results, res)
        push!(ho2.history, params2)
    end
    return from_logspace(ho2.minimizer, s.logdims)

end


# Hyperband ====================================================================

"""

"""
@with_kw mutable struct Hyperband <: Sampler
    R
    η::Int = 3
    minimum = (Inf,)
    inner = RandomSampler()
end
Hyperband(R) = Hyperband(R=R)

function macrobody(ex, params, candidates, sampler::Hyperband)
    @show params, candidates
    quote
        iters = $(esc(candidates[1]))
        if $sampler.inner isa LHSampler
            smax = floor(Int, log($sampler.η,$sampler.R))
            B = (smax + 1)*$sampler.R
            iters = floor(Int,$sampler.R*$sampler.η^smax)
            ss = string($sampler.inner)
            @info "Starting Hyperband with inner sampler $(ss). Setting the number of iterations to R*η^log(η,R)=$(iters), make sure all candidate vectors have this length as well!"
        end
        ho = Hyperoptimizer(iterations = iters, params = $(esc(params[2:end])), candidates = $(Expr(:tuple, esc.(candidates[2:end])...)), sampler=$sampler)

        costfun = $(Expr(:tuple, esc.(params)...)) -> $(esc(ex.args[2]))
        hyperband($sampler, ho, costfun)
        ho
    end
end

function hyperband(hb::Hyperband, ho, costfun)
    R,η = hb.R, hb.η
    hb.minimum = (Inf,)
    smax = floor(Int, log(η,R))
    B = (smax + 1)*R
    # ho.iterations >= R*η^smax || error("The number of iterations must be larger than R*η^log(η,R) = $(R*η^smax)")
    Juno.progress() do id
        for s in smax:-1:0
            n = ceil(Int, (B/R)*((η^s)/(s+1)))
            r = R / (η^s)
            minᵢ = successive_halving(hb, ho, costfun, n, r, s)
            if minᵢ[1] < hb.minimum[1]
                hb.minimum = minᵢ
            end
            Base.CoreLogging.@logmsg -1 "Hyperband" progress=(smax-s)+1/(smax+1)  _id=id
        end
    end
    return hb.minimum
end


function successive_halving(hb, ho, costfun, n, r=1, s=round(Int, log(hb.η, n)))
    η = hb.η
    minimum = Inf
    T = [ hb.inner(ho) for i=1:n ]
    # append!(ho.history, T)
    Juno.progress() do id
        for i in 0:s
            nᵢ = floor(Int,n/(η^i))
            rᵢ = floor(Int,r*(η^i))
            LTend = [ costfun(rᵢ, t...) for t in T ]
            L, T = first.(LTend), last.(LTend)
            # if i == 0
            #     append!(ho.results, L)
            # else
            #     for t in eachindex(T) # Update results for those that were continued
            #         ho.results[findlast(x->x==T[t], ho.history)] = L[t]
            #     end
            # end
            append!(ho.history, T)
            append!(ho.results, L)

            perm = sortperm(L)
            besti = perm[1]
            if L[besti] < minimum[1]
                minimum = (L[besti], rᵢ, T[besti])
            end
            T = T[perm[1:floor(Int,nᵢ/η)]]
            # T, minimum
            # T, minimum = top_k(hb,T,L,nᵢ,minimum)
            Base.CoreLogging.@logmsg -1 "successive_halving" progress=i/s  _id=id
        end
    end
    return minimum
end

function top_k(hb,T,L,nᵢ,minimum)
    η = hb.η
end
