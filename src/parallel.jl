"""
    Hyperopt.tmap(f, c...; ntasks = Threads.nthreads())

Multithreaded `map`: return the values of `f(x...)` for the elements `x` of `zip(c...)`, in the order of the elements.
The evaluations of `f` are distributed over `ntasks` tasks spawned with `Threads.@spawn`. A task takes the next element
when it has completed its previous evaluation, the collections are thus consumed in order and may be iterators of
unknown length.

If an evaluation of `f` throws an error, no further elements are taken, and a `TaskFailedException` is thrown after the
evaluations that have started have completed. If the calling task is interrupted while it waits, no further elements are
taken, and the `InterruptException` is rethrown without waiting for the evaluations that have started.
"""
function tmap(f, c...; ntasks::Integer = Threads.nthreads())
    itr = enumerate(zip(c...))
    lk = ReentrantLock()
    started = Ref(false)
    state = Ref{Any}()
    stopped = Threads.Atomic{Bool}(false)
    results = Dict{Int, Any}()
    # The next element `(k, args)`, or `nothing` if the iteration has ended or was stopped
    function next_element()
        lock(lk) do
            stopped[] && return nothing
            y = started[] ? iterate(itr, state[]) : iterate(itr)
            started[] = true
            if y === nothing
                stopped[] = true
                return nothing
            end
            state[] = y[2]
            y[1]
        end
    end
    tasks = map(1:ntasks) do _
        Threads.@spawn while (x = next_element()) !== nothing
            k, args = x
            r = try
                f(args...)
            catch
                stopped[] = true
                rethrow()
            end
            lock(() -> results[k] = r, lk)
        end
    end
    for t in tasks
        try
            wait(t)
        catch e
            e isa TaskFailedException && e.task === t && continue
            stopped[] = true
            rethrow()
        end
    end
    for t in tasks
        istaskfailed(t) && throw(TaskFailedException(t))
    end
    [results[k] for k in 1:length(results)]
end

# Iterator over the elements of `itr` that ends when `stopped[]` is true
struct Stoppable{I}
    itr::I
    stopped::Threads.Atomic{Bool}
end
Base.IteratorSize(::Type{<:Stoppable}) = Base.SizeUnknown()
Base.eltype(::Type{Stoppable{I}}) where {I} = eltype(I)
Base.iterate(s::Stoppable, state...) = s.stopped[] ? nothing : iterate(s.itr, state...)

"""
    stoppable_map(mapfun, f, itr)

Return `mapfun(f, s)`, where `s` is an iterator over the elements of `itr` of unknown length. `mapfun` is called in a
separate task. If the calling task is interrupted while it waits for this task, the iteration of `s` ends, so that
`mapfun` starts no further evaluations, and the `InterruptException` is rethrown without waiting for the evaluations
that have started. An error thrown by `mapfun` is rethrown.
"""
function stoppable_map(mapfun, f, itr)
    stopped = Threads.Atomic{Bool}(false)
    task = @async mapfun(f, Stoppable(itr, stopped))
    try
        return fetch(task)
    catch e
        stopped[] = true
        e isa TaskFailedException && e.task === task || rethrow()
    end
    # Thrown outside of the `catch` block, so that the `TaskFailedException` is not reported as its cause
    throw(task.exception)
end

# Whether `e` is an `InterruptException`, possibly wrapped by a remote call or a task
is_interrupt(e) = e isa InterruptException
is_interrupt(e::RemoteException) = is_interrupt(e.captured)
is_interrupt(e::CapturedException) = is_interrupt(e.ex)
is_interrupt(e::TaskFailedException) = is_interrupt(e.task.exception)

# Evaluates the objective for the pair `(i, configuration)` of an iteration index and a configuration, and puts
# `(i, result)` into the channel `results`. `pmap` serializes this function for every evaluation unless a
# `CachingPool` is used; it contains only the objective and a reference to the channel.
struct Evaluation{F, C} <: Function
    objective::F
    results::C
end

function (e::Evaluation)((i, configuration))
    put!(e.results, (i, e.objective(i, configuration...)))
    nothing
end

results_channel() = Channel{Tuple{Int, Any}}(Inf)
remote_results_channel() = RemoteChannel(results_channel)

"""
    optimize_parallel(ho, mapfun, channel)

Evaluate the iterations of `ho` with `stoppable_map(mapfun, f, itr)`, where `itr` iterates over the pairs
`(i, configuration)`. All configurations are drawn from the sampler on the calling task before the evaluations start.
`f` puts `(i, result)` into the channel created by `channel()`, and the results in the channel are recorded in the order
of the iterations when `mapfun` has returned, has thrown an error, or has been interrupted.
"""
function optimize_parallel(ho::Hyperoptimizer, mapfun, channel)
    discard_unevaluated!(ho)
    start_iteration!(ho.sampler, ho)
    configurations = [ho.sampler(ho, i) for i in 1:ho.iterations]
    results = channel()
    try
        stoppable_map(mapfun, Evaluation(ho.objective, results), enumerate(configurations))
    catch e
        record_results!(ho, configurations, results)
        is_interrupt(e) || rethrow()
        @info "Aborting hyperoptimization"
        return ho
    end
    record_results!(ho, configurations, results)
end

# Hyperband evaluates the configurations of each rung with `mapfun`
optimize_parallel(ho::Hyperoptimizer{<:Hyperband}, mapfun, channel) =
    optimize(ho, (f, c...) -> stoppable_map(mapfun, splat(f), zip(c...)))

function record_results!(ho, configurations, results)
    completed = Tuple{Int, Any}[]
    while isready(results)
        push!(completed, take!(results))
    end
    for (i, result) in sort!(completed; by = first)
        push!(ho.history, configurations[i])
        push!(ho.results, result)
    end
    ho
end

"""
    @phyperopt for i = iterations, [sampler = ..., ho = ...,] a = candidates_a, b = candidates_b, ...
        body
    end [pmap]

Same as `@hyperopt`, but the evaluations of the loop body are distributed over the worker processes with
`Distributed.pmap`. The functions that the loop body calls must be defined on all workers, e.g., with `@everywhere`.

All configurations are drawn from the sampler on the calling process before the evaluations start. For every
evaluation, a worker receives the function that evaluates the loop body, the iteration index and the configuration, and
returns only the result. The amount of data transferred per evaluation is thus independent of the number of evaluations.
The evaluations are recorded in `ho.history` and `ho.results` in the order of the iterations, with the same types as by
`@hyperopt`; with a seeded sampler, e.g., `RandomSampler(Xoshiro(0))`, `@phyperopt` records the same configurations as
`@hyperopt`.

The optional second argument is a function with the interface of `pmap`, which is called as `pmap(f, itr)`, where `itr`
is an iterator of unknown length. It can, e.g., pass keyword arguments to `pmap`, as in
`(f, itr) -> pmap(f, itr; on_error = e -> nothing)`, or use a `CachingPool`, as in
`(f, itr) -> pmap(f, CachingPool(workers()), itr)`, which sends `f` to each worker only once. The latter is useful if
the loop body refers to large local variables, e.g., if `@phyperopt` is called inside a function.

Errors and interruptions:
- If the loop body throws an error, no further evaluations are started, and the error is rethrown as a `RemoteException`
  when the evaluations that have started have completed. Other tasks on the workers are not interrupted. The results
  obtained so far are recorded in the `Hyperoptimizer`, which remains accessible if it was given with `ho = ...`.
- If `pmap` handles errors, e.g., with the keyword argument `on_error`, the failed evaluations are not recorded, and the
  values that `on_error` returns are ignored.
- If the calling task is interrupted, e.g., by Ctrl-C, no further evaluations are started, and the `Hyperoptimizer` with
  the results obtained so far is returned. The evaluations that are running are not interrupted; they complete in the
  background, and their results are discarded. `Distributed.interrupt()` interrupts them, together with any other task
  that runs on the workers.

With a `Hyperband` sampler, the configurations of each rung of successive halving are evaluated in parallel, see
`hyperband`. The value that `on_error` returns for a failed evaluation is then used as its result, e.g.,
`on_error = e -> Inf`, since successive halving requires a loss for every configuration of a rung.
"""
macro phyperopt(ex, pmap = pmap)
    pre = preprocess_expression(ex)
    ho_ = local_scope(create_ho(pre...))
    :(optimize_parallel($ho_, $(esc(pmap)), remote_results_channel))
end

# Wraps `ex` in a `let` block. The objective defined by `create_ho` is then a local function, which `Serialization`
# sends to the workers together with its methods, also if the macro is called at the top level of `Main`. A global
# function of `Main` is sent as a reference to a binding that does not exist on the workers.
local_scope(ex) = Expr(:let, Expr(:block), ex)

"""
    @thyperopt for i = iterations, [sampler = ..., ho = ...,] a = candidates_a, b = candidates_b, ...
        body
    end

Same as `@hyperopt`, but the loop body is evaluated in `Threads.nthreads()` tasks spawned with `Threads.@spawn`, see
`Hyperopt.tmap`. The loop body must therefore be safe to evaluate concurrently, and the memory consumption increases
with the number of concurrent evaluations.

The configurations are drawn and the evaluations are recorded as described for `@phyperopt`. Errors and interruptions
are also handled as described for `@phyperopt`, except that an error in the loop body is rethrown as a
`TaskFailedException`. With a `Hyperband` sampler, the configurations of each rung of successive halving are evaluated
in parallel.
"""
macro thyperopt(ex)
    pre = preprocess_expression(ex)
    ho_ = local_scope(create_ho(pre...))
    :(optimize_parallel($ho_, tmap, results_channel))
end
