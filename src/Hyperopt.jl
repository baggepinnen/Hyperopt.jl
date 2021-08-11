module Hyperopt

export Hyperoptimizer, @hyperopt, @phyperopt, @thyperopt, printmin, printmax, warn_on_boundary
export RandomSampler, LHSampler, CLHSampler, hyperband, Hyperband, hyperoptim, BOHB, Continuous, Categorical, UnorderedCategorical

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
using Requires
using Printf

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
    s === :minimum && (return isempty(ho.results) ? NaN :  minimum(replace(ho.results, NaN => Inf)))
    s === :minimizer && (return isempty(ho.results) ? [] :  ho.history[argmin(replace(ho.results, NaN => Inf))])
    s === :maximum && (return isempty(ho.results) ? NaN :  maximum(replace(ho.results, NaN => Inf)))
    s === :maximizer && (return isempty(ho.results) ? [] :  ho.history[argmax(replace(ho.results, NaN => Inf))])
    return getfield(ho,s)
end

function Base.show(io::IO, ho::Hyperoptimizer)
    println(io, "Hyperoptimizer with")
    cands = ho.candidates
    candstring = map(keys(cands)) do k
        s = "  "*string(k)*" length: "
        if length(cands[k]) <= 3
            s*string(cands[k])
        else
            s*string(length(cands[k]))
        end
    end
    candstring = join(candstring, "\n")
    println(io, candstring)
    println(io, "  minimum / maximum: $((ho.minimum, ho.maximum))")
    println(io, "  minimizer:")
    for (i, v) in enumerate(ho.minimizer)
        @printf(io, "%9s ", string(ho.params[i]))
    end
    println(io)
    for (i, v) in enumerate(ho.minimizer)
        @printf(io, "%9.4g ", v)
    end
    println()
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


function Base.iterate(ho::Hyperoptimizer, state=1; update_history=true)
    state > ho.iterations && return nothing
    samples = ho.sampler(ho, state)
    update_history && push!(ho.history, samples)
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
        function $(funname)($(esc(params[1])), $(esc(:state)))
            $(esc(ex.args[2]))
        end
    end

    params, candidates, sampler_, ho_, fun
end


function optimize(ho::Hyperoptimizer)
    try
        @showprogress "Hyperoptimizing" for nt = ho
            res = ho.objective(nt...)
            push!(ho.results, res)
        end
    catch e
        if e isa InterruptException
            @info "Aborting hyperoptimization"
        else
            rethrow()
        end
    end
    ho
end

function create_ho(params, candidates, sampler, ho_, objective_)
    esccands = esc.(candidates)
    allcands = Expr(:tuple, esccands...)
    quote
        objective, iters = $(objective_), $(esc(candidates[1]))
        ho = Hyperoptimizer(; iterations = iters, params = $(esc(params))[2:end], candidates = $(allcands)[2:end], sampler=$(esc(sampler)), objective)
        if $(esc(ho_)) isa Hyperoptimizer # get info from existing ho. the objective function might be changed, due to variables are captured into the closure, so the type of ho also changed.
            ho.sampler = $(esc(ho_)).sampler
            ho.history = $(esc(ho_)).history # it's important to use the same array, not copy.
            ho.results = $(esc(ho_)).results
        else
            s = ho.sampler
            if s isa Hyperband
                ho.iterations = length($(esc(candidates[end])))
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

            # We use a `RemoteChannel` to coordinate access to a single Hyperoptimizer object
            # that lives on the manager process.
            ho_channel = RemoteChannel(() -> Channel{Hyperoptimizer}(1), 1)

            # We use a `deepcopy` to ensure we get the same semantics whether or not the code
            # ends up executing on a remote process or not (i.e. always a copy). See
            # <https://docs.julialang.org/en/v1/manual/distributed-computing/#Local-invocations>
            put!(ho_channel, deepcopy(ho))

            # We don't care about the results of the `pmap` since we update the hyperoptimizer
            # inside the loop.
            $(esc(pmap))(1:ho.iterations) do i
                # We take the hyperoptimizer out of the channel, and get our new parameter values
                local_ho = take!(ho_channel)
                # We use `update_history` because we want to only update the history once we've
                # finished the iteration and have a result to report back as well. Otherwise,
                # some processes may observe the Hyperoptimizer in an inconsistent state with
                # `length(ho.history) > length(ho.results)`. Moreover, if one run is very quick
                # the history and results could become out of order with respect to one another.
                $(Expr(:tuple, esc.(params)...)), _ = iterate(local_ho, i; update_history = false)
                # Now we put it back so another processor can use the hyperoptimizer
                put!(ho_channel, local_ho)

                # Now run the objective
                res = $(esc(ex.args[2])) # ex.args[2] = Body of the For loop

                # Now update the results; we again grab the one true hyperoptimizer,
                # and populate it's history and result.
                local_ho = take!(ho_channel)
                push!(local_ho.history, $(Expr(:tuple, esc.(params[2:end])...)))
                push!(local_ho.results, res)
                put!(ho_channel, local_ho)

                res
            end

            # What we get out of the channel is an updated copy of our original hyperoptimizer.
            # So now, back on the manager process, we take it out one last time and update
            # the original hyperoptimizer.
            updated_ho = take!(ho_channel)
            close(ho_channel)

            # Getting the history right is tricky. For some reason, we can't do `ho.history = updated_ho.history`.
            # Instead, we must mutate the existing `ho.history` vector. (Similarly for results).
            empty!(ho.history)
            append!(ho.history, updated_ho.history)

            empty!(ho.results)
            append!(ho.results, updated_ho.results)

            ho
        end
        workaround_function()
    end
end

"""
Same as `@hyperopt` but uses `Distributed.pmap` for parallel evaluation of the cost function.
"""
macro phyperopt(ex, pmap=pmap)
    pre = preprocess_expression(ex)
    ho_ = create_ho(pre...)
    pmacrobody(ex, pre[1], ho_, pmap)
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
