module Hyperopt

export Hyperoptimizer, @hyperopt, @phyperopt, printmin, printmax
export RandomSampler, BlueNoiseSampler, LHSampler, CLHSampler, Continuous, Categorical

using LinearAlgebra, Statistics
using Lazy
using MacroTools
using MacroTools: postwalk, prewalk
using Parameters
using RecipesBase
using Distributed
using LatinHypercubeSampling

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
    s == :minimum && (return minimum(ho.results))
    s == :minimizer && (return minimum(ho)[1])
    return getfield(ho,s)
end

include("samplers.jl")

function Hyperoptimizer(iterations::Int; kwargs...)
    params = keys(kwargs)
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

function preprocess_expression(ex)
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

    params, candidates, sampler_
end

macro hyperopt(ex)
    params, candidates, sampler_ = preprocess_expression(ex)
    quote
        ho = Hyperoptimizer(iterations = $(esc(candidates[1])), params = $(esc(params[2:end])), candidates = $(Expr(:tuple, esc.(candidates[2:end])...)), sampler=$(esc(sampler_)))
        for $(Expr(:tuple, esc.(params)...)) = ho
            res = $(esc(ex.args[2])) # ex.args[2] = Body of the For loop
            push!(ho.results, res)
        end
        ho
    end
end

macro phyperopt(ex)
    params, candidates, sampler_ = preprocess_expression(ex)
    quote
        function workaround_function()
            ho = Hyperoptimizer(iterations = $(esc(candidates[1])), params = $(esc(params[2:end])), candidates = $(Expr(:tuple, esc.(candidates[2:end])...)), sampler=$(esc(sampler_)))

            res = pmap(1:ho.iterations) do i
                $(Expr(:tuple, esc.(params)...)),_ = iterate(ho,i)
                res = $(esc(ex.args[2])) # ex.args[2] = Body of the For loop

                res, $(Expr(:tuple, esc.(params[2:end])...))
            end
            append!(ho.results, getindex.(res,1))
            for r in res
                push!(ho.history, r[2])
            end
            ho
        end
        workaround_function()
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

    end
end

end # module
