module Hyperopt

export Hyperoptimizer, @hyperopt, @phyperopt, printmin, printmax
export RandomSampler, BlueNoiseSampler, LHSampler, CLHSampler, Continuous, Categorical, GPSampler, Max, Min, Hyperband

using LinearAlgebra, Statistics
using Juno
using Lazy
using MacroTools
using MacroTools: postwalk, prewalk
using Parameters
using RecipesBase
using Distributed
using LatinHypercubeSampling
using BayesianOptimization, GaussianProcesses

abstract type Sampler end
@with_kw struct Hyperoptimizer
    iterations::Int
    params
    candidates
    history = []
    results = []
    sampler::Sampler = RandomSampler()
end

function Base.getproperty(ho::Hyperoptimizer, s::Symbol)
    s == :minimum && (return minimum(replace(ho.results, NaN => Inf)))
    s == :minimizer && (return minimum(ho)[1])
    return getfield(ho,s)
end

include("samplers.jl")

function Hyperoptimizer(iterations::Int, sampler::Sampler = RandomSampler(); kwargs...)
    params = keys(kwargs)
    candidates = []
    for kw in kwargs
        push!(candidates, kw[2])
    end
    Hyperoptimizer(iterations=iterations, params=params, candidates=candidates, sampler=sampler)
end

Lazy.@forward Hyperoptimizer.history Base.length, Base.getindex


function Base.iterate(ho::Hyperoptimizer, state=1)
    state > ho.iterations && return nothing
    samples = ho.sampler(ho)
    push!(ho.history, samples)
    [state;samples], state+1
end

function preprocess_expression(ex)
    ex.head == :for || error("Wrong syntax, Use For-loop syntax, ex: @hyperopt for i=100, param=LinRange(1,10,100) ...")
    params     = []
    candidates = []
    sampler_ = :(RandomSampler())
    loop = ex.args[1].args # ex.args[1] = the arguments to the For loop
    i = 1
    while i <= length(loop)
        if @capture(loop[i], sampler = sam_) # A sampler was provided
            sampler_ = sam
            deleteat!(loop,i) # Remove the sampler from the args
            continue
        end
        @capture(loop[i], param_ = list_) || error("Wrong syntax, Use For-loop syntax, ex: @hyperopt for i=100, param=LinRange(1,10,100) ...")
        push!(params, param)
        push!(candidates, list)
        i += 1
    end
    params = ntuple(i->params[i], length(params))

    params, candidates, sampler_
end

function macrobody(ex, params, candidates, sampler)
    quote
        ho = Hyperoptimizer(iterations = $(esc(candidates[1])), params = $(esc(params[2:end])), candidates = $(Expr(:tuple, esc.(candidates[2:end])...)), sampler=$(esc(sampler)))
        Juno.progress() do id
            for $(Expr(:tuple, esc.(params)...)) = ho
                res = $(esc(ex.args[2])) # ex.args[2] = Body of the For loop
                push!(ho.results, res)
                Base.CoreLogging.@logmsg -1 "Hyperopt" progress=$(esc(params[1]))/ho.iterations  _id=id
            end
        end
        ho
    end
end

macro hyperopt(ex)
    params, candidates, sampler_ = preprocess_expression(ex)
    macrobody(ex, params, candidates, eval(sampler_))
end

function pmacrobody(ex, params, candidates, sampler_)
    quote
        function workaround_function()
            ho = Hyperoptimizer(iterations = $(esc(candidates[1])), params = $(esc(params[2:end])), candidates = $(Expr(:tuple, esc.(candidates[2:end])...)), sampler=$(esc(sampler_)))
            ho.sampler isa GPSampler && error("We currently do not support running the GPSampler in parallel. If this is an issue, open an issue ;)")
            res = pmap(1:ho.iterations) do i
                $(Expr(:tuple, esc.(params)...)),_ = iterate(ho,i)
                res = $(esc(ex.args[2])) # ex.args[2] = Body of the For loop

                res, $(Expr(:tuple, esc.(params[2:end])...))
            end
            append!(ho.results, getindex.(res,1))
            empty!(ho.history) # The call to iterate(ho) populates history, but only on host process
            append!(ho.history, getindex.(res,2))
            ho
        end
        workaround_function()
    end
end

macro phyperopt(ex)
    params, candidates, sampler_ = preprocess_expression(ex)
    pmacrobody(ex, params, candidates, sampler_)
end

function Base.minimum(ho::Hyperoptimizer)
    m,i = findmin(replace(ho.results, NaN => Inf))
    ho.history[i], m
end
function Base.maximum(ho::Hyperoptimizer)
    m,i = findmax(replace(ho.results, NaN => Inf))
    ho.history[i], m
end

function printmin(ho::Hyperoptimizer)
    m,i = minimum(ho)
    for (param, value) in zip(ho.params, m)
        println(param, " = ", value)
    end
end

function printmax(ho::Hyperoptimizer)
    m,i = maximum(ho)
    for (param, value) in zip(ho.params, m)
        println(param, " = ", value)
    end
end

include("plotting.jl")
end # module
