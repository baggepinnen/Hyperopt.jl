module Hyperopt

export Hyperoptimizer, @hyperopt, @phyperopt, printmin, printmax
export RandomSampler, BlueNoiseSampler, LHSampler, CLHSampler, Continuous, Categorical, GPSampler, Max, Min, Hyperband

using Base.Threads: threadid, nthreads
using LinearAlgebra, Statistics, Random
using Juno
using MacroTools
using MacroTools: postwalk, prewalk
using RecipesBase
using Distributed
using LatinHypercubeSampling
using BayesianOptimization, GaussianProcesses

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
    s === :minimizer && (return ho.history[argmin(ho.results)])
    s === :maximum && (return maximum(replace(ho.results, NaN => Inf)))
    s === :maximizer && (return ho.history[argmax(ho.results)])
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
    params = ntuple(i->params[i], length(params))
    state_ = sampler_.args[1] === :Hyperband ? :(state = nothing) : :(nothing)
    fun = quote
            $(Expr(:tuple, esc.(params)...)) -> begin
                $(esc(state_))
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

function macrobody(ex, params, candidates, sampler, ho_, objective)
    quote
        ho = ($(esc(ho_)) isa Hyperoptimizer) ? $(esc(ho_)) : Hyperoptimizer(iterations = $(esc(candidates[1])), params = $(esc(params[2:end])), candidates = $(Expr(:tuple, esc.(candidates[2:end])...)), sampler=$(esc(sampler)), objective = $(objective))
        ho.iterations = $(esc(candidates[1])) # if using existing ho, set the iterations to the new value.
        optimize(ho)
    end
end

macro hyperopt(ex)
    pre = preprocess_expression(ex)
    if pre[3].args[1] === :Hyperband
        macrobody_hyperband(ex, pre...)
    else
        macrobody(ex, pre...)
    end
end

function pmacrobody(ex, params, candidates, sampler_, ho_, objective)
    quote
        ho = ($(esc(ho_)) isa Hyperoptimizer) ? $(esc(ho_)) : Hyperoptimizer(iterations = $(esc(candidates[1])), params = $(esc(params[2:end])), candidates = $(Expr(:tuple, esc.(candidates[2:end])...)), sampler=$(esc(sampler_)), objective = $(objective))
        ho.iterations = $(esc(candidates[1])) # if using existing ho, set the iterations to the new value.
        ($(esc(ho_)) isa Hyperoptimizer) || init!(ho.sampler, ho)
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
end

macro phyperopt(ex)
    pre = preprocess_expression(ex)
    pre[3].args[1] === :GPSampler && error("We currently do not support running the GPSampler in parallel. If this is an issue, open an issue ;)")
    pmacrobody(ex, pre...)
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

include("plotting.jl")
end # module
