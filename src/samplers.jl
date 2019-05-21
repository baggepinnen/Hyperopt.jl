"""
Sample a value For each parameter uniformly at random from the candidate vectors. Log-uniform sampling available by providing a log-spaced candidate vector.
"""
struct RandomSampler <: Sampler end
function (s::RandomSampler)(ho)
    [list[rand(1:length(list))] for list in ho.candidates]
end

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
    log_function_noise = 5.
    hypertuning_interval = max(2, ho.iterations ÷ 10)
    model = ElasticGPE(ndims, mean = MeanConst(0.),
                       kernel = SEArd(log.(kernel_widths), log_function_noise), logNoise = 0.)
    modeloptimizer = MLGPOptimizer(every = hypertuning_interval, noisebounds = [-5, 3], # log bounds on the function noise?
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

    acqfunc = BayesianOptimization.acquisitionfunction(ExpectedImprovement(Int(s.sense)*maximum(ho.results)), s.model)
    ho2 = Hyperoptimizer(iterations=min(1000, prod(length, s.candidates)), params=ho.params, candidates=s.candidates)
    for params in ho2
        params2 = [params[2:end]...]
        res = -Inf
        try
            res = acqfunc(params2)
        catch
        end
        push!(ho2.results, res)
        push!(ho2.history, params2)
    end
    return from_logspace(ho2.minimizer, s.logdims)

end
