logscale(params::AbstractVector{T}) where T <: Real = /(extrema(params)...) < 2e-2 && minimum(params) > 0

logscale(params) = false

@recipe function plot(ho::Hyperoptimizer)
    N = length(ho.params)
    layout --> N
    for i = 1:N
        params = getindex.(ho.history, i)
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
                yscale --> (logscale(ho.results) ? :log10 : :identity)
                params[perm], ho.results[perm]
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
                yscale --> (logscale(ho.results) ? :log10 : :identity)
                paramidxs, ho.results
            end
        end
    end
end
