function logscale(x)
    isempty(x) && return false
    all(v -> v isa Real, x) || return false
    lo, hi = extrema(x)
    lo > 0 && lo / hi < 2e-2
end

# Labels and indices of the groups of results that are plotted as separate series
result_groups(ho::Hyperoptimizer, idx) = ["Sampled points" => idx]

# The results of Hyperband are grouped by their budget, in ascending order of the budget
function result_groups(ho::Hyperoptimizer{Hyperband}, idx)
    budgets = ho.sampler.budgets
    length(budgets) == length(ho.results) || return ["Sampled points" => idx]
    map(sort!(unique(budgets[idx]))) do b
        "budget = $(round(b, sigdigits=3))" => filter(k -> budgets[k] == b, idx)
    end
end

# Results that are NaN or infinite, and history entries without a result, are not plotted.
@recipe function plot(ho::Hyperoptimizer)
    N = length(ho.params)
    layout --> N
    n = min(length(ho.history), length(ho.results))
    finite = [k for k in 1:n if ho.results[k] isa Real && isfinite(ho.results[k])]
    groups = result_groups(ho, finite)
    allresults = [ho.results[k] for k in finite]
    for i = 1:N
        allparams = [ho.history[k][i] for k in finite]
        realvalued = eltype(allparams) <: Real
        uniqueparams = realvalued ? nothing : sort(unique(Symbol.(allparams)))
        for (grouplabel, idx) in groups
            params = [ho.history[k][i] for k in idx]
            results = [ho.results[k] for k in idx]
            yguide --> "Function value"
            seriestype --> :scatter
            @series begin
                xguide --> ho.params[i]
                subplot --> i
                label --> grouplabel
                legend --> (length(groups) > 1 && i == 1 ? :best : false)
                yscale --> (logscale(allresults) ? :log10 : :identity)
                if realvalued
                    perm = sortperm(params)
                    xscale --> (logscale(allparams) ? :log10 : :identity)
                    params[perm], results[perm]
                else
                    xticks --> (1:length(uniqueparams), uniqueparams)
                    xscale --> :identity
                    map(x -> findfirst(==(Symbol(x)), uniqueparams), params), results
                end
            end
        end
    end
end
