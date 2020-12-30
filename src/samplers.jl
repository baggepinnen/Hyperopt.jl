    """
Sample a value For each parameter uniformly at random from the candidate vectors. Log-uniform sampling available by providing a log-spaced candidate vector.
"""
struct RandomSampler <: Sampler end

function init!(::RandomSampler, ho) end

function (s::RandomSampler)(ho, iter)
    [list[rand(1:length(list))] for list in ho.candidates]
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

function (s::LHSampler)(ho, iter)
    init!(s, ho)
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

function (s::CLHSampler)(ho, iter)
    init!(s, ho)
    [list[s.samples[dim,iter]] for (dim,list) in enumerate(ho.candidates)]
end





# GaussianProcesses sampler ====================================================

"""
Sample using Bayesian optimization. `GPSampler(Min)/GPSampler(Max)` fits a Gaussian process to the data and tries to use this model to figure out where the best point to sample next is (using expected improvement). Underneath, the package [BayesianOptimization.jl](https://github.com/jbrea/BayesianOptimization.jl/) is used. We try to provide reasonable defaults for the underlying model and optimizer and we do not provide any customization options for this sampler. If you want advanced control, use BayesianOptimization.jl directly.
"""
mutable struct GPSampler <: Sampler
    sense
    model
    # opt
    modeloptimizer
    logdims
    candidates
end

GPSampler() = error("You must specify GPSampler(Min)/GPSampler(Max) for minimization or maximization.")
GPSampler(sense) = GPSampler(sense,nothing,nothing,nothing,nothing)

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
    ndims                = length(ho.candidates)
    logdims              = islogspace.(ho.candidates)
    candidates           = to_logspace(ho.candidates, logdims)
    lower_bounds         = [minimum.(candidates)...]
    upper_bounds         = [maximum.(candidates)...]
    widths               = upper_bounds - lower_bounds
    kernel_widths        = widths/10
    log_function_noise   = 0.
    hypertuning_interval = max(9, ho.iterations ÷ 9)
    model = ElasticGPE(ndims, mean = MeanConst(0.),
                       kernel = SEArd(log.(kernel_widths), log_function_noise), logNoise = 0.)

    modeloptimizer = MAPGPOptimizer(every = hypertuning_interval, noisebounds = [-5, 0], # log bounds on the function noise?
                                    maxeval = 40) # max iters for optimization of the GP hyperparams

    s.model = model
    s.modeloptimizer = modeloptimizer
    s.logdims = logdims
    s.candidates = candidates
end


function (s::GPSampler)(ho, iter)
    init!(s, ho)
    iter = length(ho.history)+1
    if iter <= 3
        return [rand(list) for (dim,list) in enumerate(ho.candidates)]
    else
        input = reshape(to_logspace(float.(ho.history[end]), s.logdims), :, 1)
        try
            BayesianOptimization.update!(s.model, input, [Int(s.sense)*ho.results[end]])
            if iter >= 9
                BayesianOptimization.optimizemodel!(s.modeloptimizer, s.model) # This determines whether to run or not internally?
            end
        catch ex
            @warn("BayesianOptimization failed at iter $iter: error: ", ex)
        end
    end

    # We now optimize the acq function using the random sampler. This could potentially be improved upon
    acqfunc = BayesianOptimization.acquisitionfunction(ExpectedImprovement(maximum(s.model.y)), s.model)
    # plot(s) |> display
    # plot!(x->acqfunc([x]), 1, 5, sp=2)
    # sleep(0.2)
    iters = min(30, prod(length, s.candidates))
    # iters = 3000
    ho2 = Hyperoptimizer(iterations=iters, params=ho.params, candidates=s.candidates)
    for params in ho2
        params2 = collect(params)[2:end]
        res = -Inf
        try
            res = acqfunc(params2)
        catch ex
            @warn("BayesianOptimization acqfunc failed at iter $iter: error: ", ex)
        end
        push!(ho2.results, res)
        # push!(ho2.history, params2)
    end

    # @show ho2.maximizer
    # xpl = reduce(vcat,ho2.history)
    # @show length(xpl)
    # scatter!(xpl, zeros(3000), sp=2) |> display

    return from_logspace(ho2.maximizer, s.logdims)

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

function macrobody_hyperband(ex, params, candidates, sampler)
    quote
        iters = $(esc(candidates[1]))
        if $(esc(sampler)).inner isa LHSampler
            smax = floor(Int, log($(esc(sampler)).η,$(esc(sampler)).R))
            B = (smax + 1)*$(esc(sampler)).R
            iters = floor(Int,$(esc(sampler)).R*$(esc(sampler)).η^smax)
            ss = string($(esc(sampler)).inner)
            @info "Starting Hyperband with inner sampler $(ss). Setting the number of iterations to R*η^log(η,R)=$(iters), make sure all candidate vectors have this length as well!"
        end
        ho = Hyperoptimizer(iterations = iters, params = $(esc(params[2:end])), candidates = $(Expr(:tuple, esc.(candidates[2:end])...)), sampler=$(esc(sampler)))

        costfun = $(Expr(:tuple, esc.(params)...)) -> begin
            $(esc(:(state = nothing)))
            $(esc(ex.args[2]))
        end
        (::$typeof(costfun))($(esc(params[1])), $(esc(:state))) = $(esc(ex.args[2]))
        hyperband($(esc(sampler)), ho, costfun)
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
    T = [ hb.inner(ho, i) for i=1:n ]
    # append!(ho.history, T)
    Juno.progress() do id
        for i in 0:s
            nᵢ = floor(Int,n/(η^i))
            rᵢ = floor(Int,r*(η^i))
            if i == 0
                LTend = [ costfun(rᵢ, t...) for t in T ]
            else
                LTend = [ costfun(rᵢ, t) for t in T ]
            end
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
