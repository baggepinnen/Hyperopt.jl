module Hyperopt

export Hyperoptimizer, @hyperopt, printmin, printmax

using LinearAlgebra, Statistics
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
    ex.head == :for || error("Wrong syntax, Use For-loop syntax")
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
        @capture(loop[i], param_ = list_) || error("Wrong syntax In @hyperopt")
        push!(params, param)
        push!(candidates, list)
        i += 1
    end
    params = ntuple(i->params[i], length(params))
    quote
        ho = Hyperoptimizer(iterations = $(esc(candidates[1])), params = $(esc(params[2:end])), candidates = $(Expr(:tuple, esc.(candidates[2:end])...)), sampler=$(esc(sampler_)))
        for $(Expr(:tuple, esc.(params)...)) = ho
            res = $(esc(ex.args[2])) # ex.args[2] = Body of the For loop
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
        seriestype --> :scatter
        @series begin
            xlabel --> ho.params[i]
            subplot --> i
            label --> "Sampled points"
            if /(extrema(params)...) < 2e-2 && minimum(params) > 0
                xscale --> :log10
            end
            if /(extrema(ho.results)...) < 2e-2 && minimum(ho.results) > 0
                yscale --> :log10
            end
            params[perm], ho.results[perm]
        end

        if ho.sampler isa TreeSampler || ho.sampler isa ForestSampler
            apply = ho.sampler isa TreeSampler ? apply_tree : apply_forest
            m = model(ho.sampler,ho)
            @series begin
                xlabel --> ho.params[i]
                subplot --> i
                label --> "Model predictions (around best found value)"
                xvals = repeat(minimum(ho)[1],1,length(params))
                xvals[i,:] = params[perm]
                linestyle --> :dash
                params[perm], apply(m, copy(xvals'))
            end
            @series begin
                xlabel --> ho.params[i]
                subplot --> i
                label --> "Model predictions (at samples)"
                linestyle --> :dash
                params[perm], apply(m, copy(hcat(ho.history...)'))
            end
        end
    end
end

end # module
