module Hyperopt

export Hyperoptimizer, @hyperopt, @phyperopt, @thyperopt, printmin, printmax, warn_on_boundary
export RandomSampler, LHSampler, CLHSampler, Continuous, Categorical, Hyperband, hyperband, hyperoptim

using Base.Threads: threadid, nthreads
using LinearAlgebra, Statistics, Random
using Juno
using MacroTools
using MacroTools: postwalk, prewalk
using RecipesBase
using Distributed
using LatinHypercubeSampling
using ThreadPools
using Requires

const HO_RNG = [MersenneTwister(rand(1:1000)) for _ in 1:nthreads()]

abstract type Sampler end
Base.@kwdef mutable struct Hyperoptimizer{S<:Sampler, F}
    iterations::Int
    params
    candidates
    history = []
    results = []
    sampler::S = RandomSampler()
    objective::F = nothing
end

function Base.getproperty(ho::Hyperoptimizer, s::Symbol)
    s === :minimum && (return minimum(replace(ho.results, NaN => Inf)))
    s === :minimizer && (return ho.history[argmin(replace(ho.results, NaN => Inf))])
    s === :maximum && (return maximum(replace(ho.results, NaN => Inf)))
    s === :maximizer && (return ho.history[argmax(replace(ho.results, NaN => Inf))])
    return getfield(ho,s)
end

Base.propertynames(ho::Hyperoptimizer) = (:minimum, :minimizer, :maximum, :maximizer, fieldnames(Hyperoptimizer)...)

include("samplers.jl")

function Hyperoptimizer(iterations::Int, sampler::Sampler = RandomSampler(); kwargs...)
    params = keys(kwargs)
    candidates = []
    for kw in kwargs
        push!(candidates, kw[2])
    end
    Hyperoptimizer(iterations=iterations, params=params, candidates=candidates, sampler=sampler)
end

Base.length(ho::Hyperoptimizer) = ho.iterations


function Base.iterate(ho::Hyperoptimizer, state=1)
    state > ho.iterations && return nothing
    samples = ho.sampler(ho, state)
    push!(ho.history, samples)
    nt = (; Pair.((:i, ho.params...), (state, samples...))...)
    nt, state+1
end

"""
Hyperopt internal function

Returns
`params, candidates, sampler_, ho_, fun`
where
`params` is a vector of symbols
`candidates` is a vector of expresions creating candidate vectors
`sampler_` is an expression creating a sampler
`ho_` is either nothing or an expression for an existing hyperoptimizer
`fun` is a function from `(i, pars...) -> val` that executes the macrobody
"""
function preprocess_expression(ex)
    ex.head == :for || error("Wrong syntax, Use For-loop syntax, ex: @hyperopt for i=100, param=LinRange(1,10,100) ...")
    params     = []
    candidates = []
    sampler_ = :(RandomSampler())
    ho_ = nothing
    loop = ex.args[1].args # ex.args[1] = the arguments to the For loop
    i = 1
    while i <= length(loop)
        if @capture(loop[i], sampler = sam_) # A sampler was provided
            sampler_ = sam
            deleteat!(loop,i) # Remove the sampler from the args
            continue
        end
        if @capture(loop[i], ho = h_)
            ho_ = h
            deleteat!(loop,i)
            continue
        end
        @capture(loop[i], param_ = list_) || error("Wrong syntax, Use For-loop syntax, ex: @hyperopt for i=100, param=LinRange(1,10,100) ...")
        push!(params, param)
        push!(candidates, list)
        i += 1
    end
    params = (params...,)
    funname = gensym(:hyperopt_objective)
    state_ = :(state = nothing)
    fun = quote # produces a function(i, pars...)
        function $(funname)($(Expr(:tuple, esc.(params)...))...)
            $(esc(state_))
            $(esc(ex.args[2]))
        end
        function $(funname)($(esc(:i)), $(esc(:state)))
            $(esc(ex.args[2]))
        end
    end

    params, candidates, sampler_, ho_, fun
end


function optimize(ho::Hyperoptimizer)
    Juno.progress() do id
        for nt = ho
            res = ho.objective(nt...)
            push!(ho.results, res)
            Base.CoreLogging.@logmsg Base.CoreLogging.BelowMinLevel "Hyperopt" progress=nt.i/ho.iterations  _id=id
        end
    end
    ho
end

function create_ho(params, candidates, sampler, ho_, objective_)
    quote
        objective, iters = $(objective_), $(esc(candidates[1]))
        ho = Hyperoptimizer(; iterations = iters, params = $(esc(params[2:end])), candidates = $(Expr(:tuple, esc.(candidates[2:end])...)), sampler=$(esc(sampler)), objective)
        if $(esc(ho_)) isa Hyperoptimizer # get info from existing ho. the objective function might be changed, due to variables are captured into the closure, so the type of ho also changed.
            ho.sampler = $(esc(ho_)).sampler
            ho.history = $(esc(ho_)).history # it's important to use the same array, not copy.
            ho.results = $(esc(ho_)).results
        else
            s = ho.sampler
            if s isa Hyperband
                ho.iterations = length($(esc(candidates[2])))
                s = s.inner
            end
            s isa Union{LHSampler,CLHSampler} && init!(s, ho)
        end
        ho
    end
end

macro hyperopt(ex)
    pre = preprocess_expression(ex)
    ho_ = create_ho(pre...)
    :(optimize($ho_))
end

function pmacrobody(ex, params, ho_, pmap=pmap)
    quote
        function workaround_function()
            ho = $(ho_)
            # Getting the history right is tricky when using workers. The approach I've found to work is to
            # save the actual array (not copy) in hist, temporarily use a new array that will later be discarded
            # reassign the original array and then append the new history. If a new array is used, the change will not be visible in the original hyperoptimizer
            hist = ho.history
            ho.history = []
            res = $(pmap)(1:ho.iterations) do i
                $(Expr(:tuple, esc.(params)...)),_ = iterate(ho,i)
                res = $(esc(ex.args[2])) # ex.args[2] = Body of the For loop

                res, $(Expr(:tuple, esc.(params[2:end])...))
            end
            ho.history = hist
            append!(ho.results, getindex.(res,1))
            append!(ho.history, getindex.(res,2)) # history automatically appended by the iteration
            ho
        end
        workaround_function()
    end
end

"""
Same as `@hyperopt` but uses `Distributed.pmap` for parallel evaluation of the cost function.
"""
macro phyperopt(ex)
    pre = preprocess_expression(ex)
    ho_ = create_ho(pre...)
    pmacrobody(ex, pre[1], ho_)
end

"""
Same as `@hyperopt` but uses `ThreadPools.tmap` for multithreaded evaluation of the cost function.
"""
macro thyperopt(ex)
    pre = preprocess_expression(ex)
    ho_ = create_ho(pre...)
    pmacrobody(ex, pre[1], ho_, ThreadPools.tmap)
end

function Base.minimum(ho::Hyperoptimizer)
    m,i = findmin(replace(ho.results, NaN => Inf))
    m
end
function Base.maximum(ho::Hyperoptimizer)
    m,i = findmax(replace(ho.results, NaN => -Inf))
    m
end

function printmin(ho::Hyperoptimizer)
    for (param, value) in zip(ho.params, ho.minimizer)
        println(param, " = ", value)
    end
end

function printmax(ho::Hyperoptimizer)
    for (param, value) in zip(ho.params, ho.maximizer)
        println(param, " = ", value)
    end
end

"""
    warn_on_boundary(ho, sense = :min)

Prints a warning message for each parameter where the optimum was obtained on an extreme point of the sampled space.

Example: If parameter `a` can take values in 1:10 and the optimum was obtained at
`a = 1`, it's an indication that the parameter was constraind by the search space.
The warning is effective even if the lowest value of `a` that was sampled was higher than 1,
but the optimum occured on the lowest sampled value.
"""
function warn_on_boundary(ho, sense = :min)
    m = sense == :min ? ho.minimizer : ho.maximizer
    n_params = length(m)
    extremas = map(1:n_params) do i
        c = ho.candidates[i]
        if c isa AbstractArray{<:Real} 
            extrema(getindex.(ho.history, i))
        else
            (m[i],)
        end
    end
    for i in eachindex(m)
        c = unique(ho.candidates[i])
        if m[i] ∈ extremas[i] && length(c) > 3
            println("Parameter $(ho.params[i]) obtained its optimum on an extremum of the sampled region: $(m[i])")
        end
    end
end

include("plotting.jl")

function __init__()
    Requires.@require Optim="429524aa-4258-5aef-a3af-852621145aeb" include("optim.jl")
end

end # module
