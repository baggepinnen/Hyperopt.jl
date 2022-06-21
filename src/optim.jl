using Setfield

"""
    hyperoptim(f, candidates, algorithm = Optim.NelderMead(), opts = Optim.Options(); R = 30, η = 3, inner = RandomSampler(), threads = true, verbose = false, Rmin = 0.1, kwargs...)

Multi-start optimization with Optim and Hyperband. The interface is similar to that of `Optim.optimize`, but the initial guess is replaced by a vector of candidate vectors. The resources are used to set the allowed time for optimization by `Optim.optimize`.

# Arguments:
- `f` a function from `Rⁿ -> R`, i.e., it takes a vector of parameters.
- `candidates`: A vector of vectors
- `algorithm`: One of the Optim algorithms
- `opts`: `Optim.Options`
- `inner`: The inner sampler
- `threads`: Use threading to evaluate the cost function
- `verbose`: DESCRIPTION
- `Rmin`: The minimum amount of resources to use.
- `kwargs`: Are sent to `Optim.optimize`.

See `?hyperband` for the remaining arguments.
"""
function hyperoptim(f, candidates, algorithm = Optim.NelderMead(), opts = Optim.Options(); R=30, η=3, inner = RandomSampler(), threads=true, verbose = false, Rmin = 0.1, kwargs...)

    fun = function(r, pars)
        r = max(r, Rmin)
        @set! opts.time_limit = r
        res = Optim.optimize(f, pars, algorithm, opts; kwargs...)
        verbose && println("Resources: $(round(r, digits=3)), value: $(round(minimum(res), digits=6))")
        Optim.minimum(res), Optim.minimizer(res)
    end

    ho = hyperband(fun, candidates; R, η, inner, threads)
    ho
end

function multistart(f, candidates; N, algorithm = Optim.NelderMead(), opts = Optim.Options(), inner = RandomSampler(), threads=false)
    ho = Hyperoptimizer(;
        iterations = N,
        params = [Symbol("$i") for i in eachindex(candidates)],
        candidates,
        history = Vector{Any}(undef, N),
        results = Vector{Any}(undef, N),
        sampler = inner,
        objective = f,
    )
    if inner isa Union{LHSampler,CLHSampler}
        ho.iterations = length(candidates[1])
        init!(inner, ho)
    end
    sem = Base.Semaphore(threads)
    try
        @sync for i = 1:N
            pars = ho.sampler(ho, i)
            # nt = (; Pair.((:i, ho.params...), (state, samples...))...)
            # pars = [Base.tail(nt)...] # the first element is i

            if threads >= 2
                Base.acquire(sem)
                Threads.@spawn begin
                    res = Optim.optimize(f, pars, algorithm, opts)
                    ho.history[i] = res.minimizer
                    ho.results[i] = res.minimum
                    Base.release(sem)
                end

            else
                res = Optim.optimize(f, pars, algorithm, opts)
                ho.history[i] = res.minimizer
                ho.results[i] = res.minimum
            end
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