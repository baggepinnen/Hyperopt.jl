module HyperoptOptimExt

using Hyperopt: Hyperopt, RandomSampler, hyperband
using Optim: Optim

function Hyperopt.hyperoptim(f, candidates, algorithm = Optim.NelderMead(), opts = Optim.Options(); R=30, η=3, inner = RandomSampler(), threads=true, verbose = false, Rmin = 0.1, kwargs...)

    fun = function(r, pars)
        r = max(r, Rmin)
        res = Optim.optimize(f, pars, algorithm, with_time_limit(opts, r); kwargs...)
        verbose && println("Resources: $(round(r, digits=3)), value: $(round(Optim.minimum(res), digits=6))")
        Optim.minimum(res), Optim.minimizer(res)
    end

    ho = hyperband(fun, candidates; R, η, inner, threads)
    ho
end

# Optim v1 provides no constructor that copies an `Optim.Options` with modified fields.
function with_time_limit(opts::Optim.Options, time_limit)
    fields = map(fieldnames(typeof(opts))) do name
        name === :time_limit ? time_limit : getfield(opts, name)
    end
    typeof(opts)(fields...)
end

end
