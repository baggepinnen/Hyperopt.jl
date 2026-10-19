function logscale(x)
    isempty(x) && return false
    all(v -> v isa Real, x) || return false
    lo, hi = extrema(x)
    lo > 0 && lo / hi < 2e-2
end

# Results that are NaN or infinite, and history entries without a result, are not plotted.
@recipe function plot(ho::Hyperoptimizer)
    N = length(ho.params)
    layout --> N
    n = min(length(ho.history), length(ho.results))
    finite = [k for k in 1:n if ho.results[k] isa Real && isfinite(ho.results[k])]
    history = ho.history[finite]
    results = [ho.results[k] for k in finite]
    for i = 1:N
        params = getindex.(history, i)
        if eltype(params) <: Real
            perm = sortperm(params)
            yguide --> "Function value"
            seriestype --> :scatter
            @series begin
                xguide --> ho.params[i]
                subplot --> i
                label --> "Sampled points"
                legend --> false
                xscale --> (logscale(params) ? :log10 : :identity)
                yscale --> (logscale(results) ? :log10 : :identity)
                params[perm], results[perm]
            end
        else
            params = Symbol.(params)
            uniqueparams = sort(unique(params))
            paramidxs = map(x -> findfirst(==(x), uniqueparams), params)
            yguide --> "Function value"
            seriestype --> :scatter
            @series begin
                xguide --> ho.params[i]
                subplot --> i
                label --> "Sampled points"
                legend --> false
                xticks --> (1:length(uniqueparams), uniqueparams)
                xscale --> :identity
                yscale --> (logscale(results) ? :log10 : :identity)
                paramidxs, results
            end
        end
    end
end
