module Hyperopt

export Hyperoptimizer, @hyperopt, printmin, printmax

using MacroTools
using MacroTools: postwalk
using Parameters

abstract type Sampler end
struct RandomSampler <: Sampler end

@with_kw struct Hyperoptimizer
    iterations::Int
    params
    candidates
    history = []
    results = []
    sampler::Sampler = RandomSampler()
end

function Hyperoptimizer(iterations::Int; kwargs...)
    params = ntuple(i->kwargs[i][1], length(kwargs))
    candidates = []
    for kw in kwargs
        push!(candidates, kw[2])
    end
    Hyperoptimizer(iterations=iterations, params=params, candidates=candidates)
end


Base.start(ho::Hyperoptimizer) = 1

function Base.next(ho::Hyperoptimizer, state)
    samples = [list[rand(1:length(list))] for list in ho.candidates]
    push!(ho.history, samples)
    [state;samples], state+1
end

Base.done(ho::Hyperoptimizer, state) = state > ho.iterations

macro hyperopt(ex)
    ex.head == :for || error("Wrong syntax, use for-loop syntax")
    params     = []
    candidates = []
    postwalk(ex.args[1]) do x # ex.args[1] = the arguments to the for loop
        @capture(x, param_ = list_) || return x
        push!(params, param)
        push!(candidates, list)
    end
    params = ntuple(i->params[i], length(params))
    ho = Hyperoptimizer(iterations = candidates[1], params = params[2:end], candidates = eval.(candidates[2:end]))
    quote
        for $(Expr(:tuple, esc.(params)...)) = $(ho)
            res = $(esc(ex.args[2])) # ex.args[2] = Body of the for loop
            push!($(ho).results, res)
            res
        end
        $ho
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
    for (param, value) in zip(hp.params, m)
        println(param, " = ", value)
    end
end

function printmax(ho::Hyperoptimizer)
    m,i = maximum(ho)
    for (param, value) in zip(hp.params, m)
        println(param, " = ", value)
    end
end



end # module
