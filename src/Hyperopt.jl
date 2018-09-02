module Hyperopt

export Hyperoptimizer, @hyperopt, printmin, printmax

using Lazy
using MacroTools
using MacroTools: postwalk, prewalk
using Parameters
using RecipesBase
using DecisionTree

abstract type Sampler end
@with_kw struct Hyperoptimizer
    iterations::Int
    params
    candidates
    history = []
    results = []
    sampler::Sampler = RandomSampler()
end
include("samplers.jl")

function Hyperoptimizer(iterations::Int; kwargs...)
    params = ntuple(i->kwargs[i][1], length(kwargs))
    candidates = []
    for kw in kwargs
        push!(candidates, kw[2])
    end
    Hyperoptimizer(iterations=iterations, params=params, candidates=candidates)
end

Lazy.@forward Hyperoptimizer.history Base.length, Base.getindex


function Base.iterate(ho::Hyperoptimizer, state=1)
    state > ho.iterations && return nothing
    samples = ho.sampler(ho)
    push!(ho.history, samples)
    [state;samples], state+1
end


macro hyperopt(ex)
    ex.head == :for || error("Wrong syntax, use for-loop syntax")
    params     = []
    candidates = []
    sampler_ = :(RandomSampler())
    dump(ex.args[1])
    error()
    ex.args[1] = prewalk(ex.args[1]) do x # ex.args[1] = the arguments to the for loop
        if @capture(x, sampler = sam_) # A sampler was provided
            sampler_ = sam
            return nothing # Remove the sampler from the args
        end
        @capture(x, param_ = list_) || return x
        push!(params, param)
        push!(candidates, list)
        x
    end
    params = ntuple(i->params[i], length(params))
    quote
        ho = Hyperoptimizer(iterations = $(esc(candidates[1])), params = $(esc(params[2:end])), candidates = $(Expr(:tuple, esc.(candidates[2:end])...)), sampler=$(esc(sampler_)))
        for $(Expr(:tuple, esc.(params)...)) = ho
            res = $(esc(ex.args[2])) # ex.args[2] = Body of the for loop
            push!(ho.results, res)
        end
        ho
    end
end

function Base.minimum(ho::Hyperoptimizer)
    m,i = findmin(ho.results)
    ho.history[i], m
end
function Base.maximum(ho::Hyperoptimizer)
    m,i = findmax(ho.results)
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


@recipe function plot(ho::Hyperoptimizer)
    N = length(ho.params)
    layout --> N
    for i = 1:N
        params = getindex.(ho.history, i)
        perm = sortperm(params)
        ylabel --> "Function value"
        @series begin
            xlabel --> ho.params[i]
            params[perm], ho.results[perm]
        end
    end
end

end # module
