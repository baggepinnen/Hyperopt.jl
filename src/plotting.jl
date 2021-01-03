
logscale(params::AbstractVector{<:Real}) = /(extrema(params)...) < 2e-2 && minimum(params) > eps()

logscale(params) = false

@recipe function plot(ho::Hyperoptimizer)
    N = length(ho.params)
    layout --> N
    for i = 1:N
        params = getindex.(ho.history, i)
        perm = sortperm(params)
        yguide --> "Function value"
        seriestype --> :scatter
        @series begin
            xguide --> ho.params[i]
            subplot --> i
            label --> "Sampled points"
            legend --> false
            logscale(params) && (xscale --> :log10)
            logscale(ho.results) && (yscale --> :log10)
            params[perm], ho.results[perm]
        end

    end
end

# robust_acq(model, x::Vector{<:Real}) = robust_acq(model, [[x] for x in x]')
robust_acq(model, x::AbstractVector{<:Real}) = robust_acq(model, reshape(x,1,:))
tovec(x::Number) = [x]
tovec(x::AbstractVector) = x

function robust_acq(s, x=s.model.x)
    acqfunc = BayesianOptimization.acquisitionfunction(BayesianOptimization.ExpectedImprovement(maximum(s.model.y)), s.model)
    map(1:size(x,2)) do i
        try
            return acqfunc(tovec(x[:,i]))
        catch ex
            @warn("BayesianOptimization failed, error: ", ex); return -Inf
        end
    end
end

@recipe function plot(s::GPSampler)
    model = s.model
    ndims = size(model.x,1)
    ms, var = GaussianProcesses.predict_f(model, model.x)
    sig = sqrt.(var)
    layout := (ndims == 1 ? 2 : ndims)
    seriestype := :scatter
    xvals = from_logspace([model.x[i,:] for i in 1:ndims], s.logdims)
    xrange = [extrema(s.candidates[i]) for i in 1:ndims]
    for i = 1:ndims
        logscale([model.y[:]; Int(s.sense)*ms]) && (yscale --> :log10)
        @series begin
            subplot := i
            label := "Model function"
            yerror := sqrt.(var)
            xvals[i], Int(s.sense)*ms
        end
        @series begin
            subplot := i
            label := "Obs"
            xlims := xrange[i]
            markershape := :cross
            xvals[i], Int(s.sense)*model.y[:]
        end
    end
    if ndims == 1
        @series begin
            xa = LinRange(extrema(s.candidates[1])...,100)
            aqy = robust_acq(s, xa)
            title -> "Expected improvement"
            subplot := 2
            lab="Aq function"
            xa, aqy
        end
    end
end


# function extend_dim(x::AbstractMatrix,d,n=200)
#     new_dim = LinRange(extrema(x[d,:])..., n)
#     new_x = repeat(x,1,n)
#     new_x[d,:] = repeat(new_dim, 1, size(x,2))
#     new_x
# end
#
# function extend_dim(x::AbstractVector{<:AbstractVector},d,n=200)
#     new_dim = LinRange(extrema(x[d])..., n)
#     new_x = repeat(x,1,n)
#     new_x[d,:] = repeat(new_dim, 1, size(x,2))
#     new_x
# end
