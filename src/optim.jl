using Setfield

function hyperoptim(f, candidates, algorithm = Optim.NelderMead(), opts = Optim.Options(); R=30, η=3, inner = RandomSampler(), threads=true, verbose = false, Rmin = 0.1, kwargs...)

    fun = function(r, pars)
        r = max(r, Rmin)
        @set! opts.time_limit = r
        res = Optim.optimize(f, pars, algorithm, opts)
        verbose && println("Resources: $(round(r, digits=3)), value: $(round(minimum(res), digits=6))")
        Optim.minimum(res), Optim.minimizer(res)
    end

    ho = hyperband(fun, candidates; R, η, inner, threads)
    ho
end