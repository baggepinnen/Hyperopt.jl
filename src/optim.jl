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