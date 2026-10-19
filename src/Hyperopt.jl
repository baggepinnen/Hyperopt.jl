module Hyperopt

export Hyperoptimizer, @hyperopt, @phyperopt, @thyperopt, printmin, printmax, warn_on_boundary
export RandomSampler, LHSampler, CLHSampler, hyperband, Hyperband, hyperoptim, BOHB, Continuous, Categorical, UnorderedCategorical

using Random
using ProgressMeter
using MacroTools
using RecipesBase
using Distributed
using LatinHypercubeSampling
using ThreadPools
using Printf

const DimensionType = LHCDimension

# # Types of dimensions
# const CategoricalDim = Categorical
# const ContinuousDim = Continuous
struct UnorderedCategorical <: DimensionType
    levels::Int64
end

abstract type Sampler end
Base.@kwdef mutable struct Hyperoptimizer{S<:Sampler, F, P, C}
    iterations::Int
    params::P
    candidates::C
    history = []
    results = []
    sampler::S = RandomSampler()
    objective::F = nothing
end

function Base.getproperty(ho::Hyperoptimizer, s::Symbol)
    if s === :minimum || s === :minimizer || s === :maximum || s === :maximizer
        idx = compared_indices(ho)
        isempty(idx) && return s === :minimum || s === :maximum ? NaN : []
        sense_min = s === :minimum || s === :minimizer
        results = replace(ho.results[idx], NaN => (sense_min ? Inf : -Inf))
        k = sense_min ? argmin(results) : argmax(results)
        return s === :minimum || s === :maximum ? results[k] : ho.history[idx[k]]
    end
    return getfield(ho,s)
end

# Indices of the results that `ho.minimum`, `ho.minimizer`, `ho.maximum` and `ho.maximizer` compare
compared_indices(ho::Hyperoptimizer) = eachindex(ho.results)

function Base.show(io::IO, ho::Hyperoptimizer)
    println(io, "Hyperoptimizer with")
    candstring = map(zip(ho.params, ho.candidates)) do (param, cand)
        s = "  "*string(param)*" length: "
        if length(cand) <= 3
            s*string(cand)
        else
            s*string(length(cand))
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
        if v isa Number
            @printf(io, "%9.4g ", v)
        else
            @printf(io, "%9s ", v)
        end
    end
    println(io)
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
Base.eltype(::Type{<:Hyperoptimizer}) = NamedTuple
# The field types of the yielded `NamedTuple` depend on the sampled values, `collect` therefore determines the element type from the elements.
Base.IteratorEltype(::Type{<:Hyperoptimizer}) = Base.EltypeUnknown()

"""
    iterate(ho::Hyperoptimizer, state = 1; update_history = true)

Draw the sample of iteration `state` from the sampler of `ho` and return it as a `NamedTuple` with the iteration index `i` and one entry per parameter, e.g., `(i = 1, a = 0.3, b = true)`.

If `update_history = true`, the sample is appended to `ho.history`; the corresponding result is to be appended to `ho.results` by the caller:
```julia
for (i, a, b) in ho
    push!(ho.results, f(a, b))
end
```
`ho.history[k]` and `ho.results[k]` then refer to the same evaluation. Entries at the end of `ho.history` without a corresponding entry in `ho.results`, e.g., the samples of an earlier iteration whose results were not recorded, are removed when the iteration starts (`state = 1`). Iterating over `ho` again after the results of all samples were recorded appends `ho.iterations` new samples, which corresponds to continuing the optimization.
"""
function Base.iterate(ho::Hyperoptimizer, state=1; update_history=true)
    state > ho.iterations && return nothing
    update_history && state == 1 && discard_unevaluated!(ho)
    samples = ho.sampler(ho, state)
    update_history && push!(ho.history, samples)
    nt = (; Pair.((:i, ho.params...), (state, samples...))...)
    nt, state+1
end

"""
    discard_unevaluated!(ho::Hyperoptimizer)

Remove the entries at the end of `ho.history` for which `ho.results` contains no corresponding result.
"""
function discard_unevaluated!(ho::Hyperoptimizer)
    n = length(ho.results)
    length(ho.history) > n && resize!(ho.history, n)
    ho
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
`fun` is a function `(i, pars...; state = nothing) -> val` that executes the macrobody

The expression `ex` is not modified.
"""
function preprocess_expression(ex)
    ex.head == :for || error("Wrong syntax, Use For-loop syntax, ex: @hyperopt for i=100, param=LinRange(1,10,100) ...")
    params     = []
    candidates = []
    sampler_ = :(RandomSampler())
    ho_ = nothing
    for arg in ex.args[1].args # ex.args[1] = the arguments to the For loop
        if @capture(arg, sampler = sam_) # A sampler was provided
            sampler_ = sam
        elseif @capture(arg, ho = h_)
            ho_ = h
        else
            @capture(arg, param_ = list_) || error("Wrong syntax, Use For-loop syntax, ex: @hyperopt for i=100, param=LinRange(1,10,100) ...")
            push!(params, param)
            push!(candidates, list)
        end
    end
    params = (params...,)
    funname = gensym(:hyperopt_objective)
    body = loop_body(ex)
    # The state is a keyword argument so that a new configuration and a resumed state are
    # not distinguished by the number of positional arguments, see `successive_halving`.
    fun = quote
        function $(funname)($(esc.(params)...); $(esc(:state)) = nothing)
            $(esc(body))
        end
    end

    params, candidates, sampler_, ho_, fun
end

"""
Hyperopt internal function

Returns the body of the for loop `ex` without a trailing `LineNumberNode`, so that the body does not evaluate to `nothing`, ref https://github.com/JuliaLang/julia/pull/41857
The expression `ex` is not modified.
"""
function loop_body(ex)
    body = ex.args[2]
    if Meta.isexpr(body, :block) && !isempty(body.args) && body.args[end] isa LineNumberNode
        body = Expr(:block, body.args[1:end-1]...)
    end
    body
end


"""
    Hyperopt.SHOW_PROGRESS[] = false

Disable the progress meters that `@hyperopt`, `hyperband` and `hyperoptim` display. The progress meters are enabled by default.
"""
const SHOW_PROGRESS = Ref(true)

function optimize(ho::Hyperoptimizer)
    try
        @showprogress desc="Hyperoptimizing" enabled=SHOW_PROGRESS[] for nt = ho
            res = ho.objective(nt...)
            push!(ho.results, res)
        end
    catch e
        discard_unevaluated!(ho)
        if e isa InterruptException
            @info "Aborting hyperoptimization"
        else
            rethrow()
        end
    end
    ho
end

function create_ho(params, candidates, sampler, ho_, objective_)
    quote
        objective = $(objective_)
        allcands = $(Expr(:tuple, esc.(candidates)...)) # The first element is the number of iterations
        ho = Hyperoptimizer(; iterations = allcands[1], params = $(esc(params))[2:end], candidates = allcands[2:end], sampler=$(esc(sampler)), objective)
        existing_ho = $(esc(ho_))
        if existing_ho isa Hyperoptimizer # get info from existing ho. the objective function might be changed, due to variables are captured into the closure, so the type of ho also changed.
            ho.sampler = existing_ho.sampler
            ho.history = existing_ho.history # it's important to use the same array, not copy.
            ho.results = existing_ho.results
        else
            s = ho.sampler
            if s isa Hyperband
                set_R!(s, allcands[1]) # The inner sampler is initialized by `hyperband`
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

            # pmap continues running a few lefterover jobs when interrupted, this is to stop that
            done_channel = RemoteChannel(() -> Channel{Bool}(1), 1)
            put!(done_channel, false)

            try
                # We don't care about the results of the `pmap` since we update the hyperoptimizer
                # inside the loop.
                $(esc(pmap))(1:ho.iterations) do i
                    isdone = take!(done_channel)
                    put!(done_channel, isdone)
                    isdone && return

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
                    res = $(esc(loop_body(ex)))

                    # Now update the results; we again grab the one true hyperoptimizer,
                    # and populate it's history and result.
                    local_ho = take!(ho_channel)
                    push!(local_ho.history, $(Expr(:tuple, esc.(params[2:end])...)))
                    push!(local_ho.results, res)
                    put!(ho_channel, local_ho)

                    res
                end
            catch e
                take!(done_channel)
                put!(done_channel, true)
                interrupt() # Stop all workers
                if e isa InterruptException
                    @info "Aborting hyperoptimization"
                else
                    rethrow()
                end
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

            # Do this last in case remaining runs are slow at starting so they can take the channel and stop
            # properly without erroring (happens if accessing channel after closing)
            close(done_channel)

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

"""
    minimum(ho::Hyperoptimizer)

Return the smallest result in `ho.results`, equivalent to `ho.minimum`. `NaN` results are treated as `Inf`. If `ho` contains no results, `NaN` is returned.
With a [`Hyperband`](@ref) sampler, only the results obtained with the largest budget are compared.
"""
Base.minimum(ho::Hyperoptimizer) = ho.minimum

"""
    maximum(ho::Hyperoptimizer)

Return the largest result in `ho.results`, equivalent to `ho.maximum`. `NaN` results are treated as `-Inf`. If `ho` contains no results, `NaN` is returned.
With a [`Hyperband`](@ref) sampler, only the results obtained with the largest budget are compared.
"""
Base.maximum(ho::Hyperoptimizer) = ho.maximum

"""
    printmin([io=stdout,] ho)

Prints the parameters that minimized the function.
"""
printmin(ho::Hyperoptimizer) = printmin(stdout, ho)
function printmin(io::IO, ho::Hyperoptimizer)
    for (param, value) in zip(ho.params, ho.minimizer)
        println(io, param, " = ", value)
    end
end

"""
    printmax([io=stdout,] ho)

Prints the parameters that maximized the function.
"""
printmax(ho::Hyperoptimizer) = printmax(stdout, ho)
function printmax(io::IO, ho::Hyperoptimizer)
    for (param, value) in zip(ho.params, ho.maximizer)
        println(io, param, " = ", value)
    end
end

"""
    warn_on_boundary(ho, sense = :min)

Emit a warning for each parameter where the optimum was obtained on an extreme point of the sampled space, and return the names of these parameters.

Example: If parameter `a` can take values in 1:10 and the optimum was obtained at
`a = 1`, it's an indication that the parameter was constrained by the search space.
The warning is effective even if the lowest value of `a` that was sampled was higher than 1,
but the optimum occurred on the lowest sampled value.

Only parameters whose candidates are real numbers with more than three distinct values are considered, since categorical candidates have no order.
"""
function warn_on_boundary(ho, sense = :min)
    m = sense == :min ? ho.minimizer : ho.maximizer
    on_boundary = Symbol[]
    for i in eachindex(m)
        c = unique(ho.candidates[i])
        (length(c) > 3 && all(x -> x isa Real, c)) || continue
        if m[i] ∈ extrema(h[i] for h in ho.history)
            @warn "Parameter $(ho.params[i]) obtained its optimum on an extremum of the sampled region: $(m[i])"
            push!(on_boundary, Symbol(ho.params[i]))
        end
    end
    on_boundary
end

include("plotting.jl")

"""
    hyperoptim(f, candidates, algorithm = Optim.NelderMead(), opts = Optim.Options(); R = 30, η = 3, inner = RandomSampler(), threads = true, verbose = false, Rmin = 0.1, kwargs...)

Multi-start optimization with Optim and Hyperband. The interface is similar to that of `Optim.optimize`, but the initial guess is replaced by a vector of candidate vectors. The resources are used to set the allowed time for optimization by `Optim.optimize`. An evaluation with a larger budget continues the optimization from the minimizer found by the previous evaluation of the same initial guess.

Returns the `Hyperoptimizer` `ho`. `ho.minimum` is the smallest value of `f` found with the largest budget, `ho.minimizer` is the initial guess from which this value was found, and `ho.sampler.minimum.state` is the corresponding minimizer of `f`.

This function requires Optim.jl to be loaded, i.e., `using Optim`.

# Arguments:
- `f` a function from `Rⁿ -> R`, i.e., it takes a vector of parameters.
- `candidates`: A vector of vectors
- `algorithm`: One of the Optim algorithms
- `opts`: `Optim.Options`. The field `time_limit` is replaced by the resources allocated to each call to `Optim.optimize`.
- `inner`: The inner sampler
- `threads`: Use threading to evaluate the cost function
- `verbose`: Print the resources and the minimum after each call to `Optim.optimize`.
- `Rmin`: The minimum amount of resources to use.
- `kwargs`: Are sent to `Optim.optimize`.

See `?hyperband` for the remaining arguments.
"""
function hyperoptim end

end # module
