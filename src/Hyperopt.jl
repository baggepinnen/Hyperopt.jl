module Hyperopt

export Hyperoptimizer, @hyperopt, @phyperopt, @thyperopt, printmin, printmax
export RandomSampler, LHSampler, CLHSampler, hyperband, Hyperband, BOHB, Continuous, Categorical, UnorderedCategorical

using Base.Threads: threadid, nthreads
using LinearAlgebra, Statistics, Random
using ProgressMeter
using MacroTools
using MacroTools: postwalk, prewalk
using RecipesBase
using Distributed
using LatinHypercubeSampling
using ThreadPools
using Distributions: Normal, truncated
using MultiKDE

const HO_RNG = [MersenneTwister(rand(1:1000)) for _ in 1:nthreads()]

const DimensionType = LHCDimension

# # Types of dimensions
# const CategoricalDim = Categorical
# const ContinuousDim = Continuous
struct UnorderedCategorical <: DimensionType
    levels::Int64
end

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
    @showprogress "Hyperoptimizing" for nt = ho
        res = ho.objective(nt...)
        push!(ho.results, res)
        Base.CoreLogging.@logmsg Base.CoreLogging.BelowMinLevel "Hyperopt" progress=nt.i/ho.iterations  _id=id
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

include("plotting.jl")
end # module
