
logscale(params) = /(extrema(params)...) < 2e-2 && minimum(params) > 0

@recipe function plot(ho::Hyperoptimizer)
    N = length(ho.params)
    layout --> N
    for i = 1:N
        params = getindex.(ho.history, i)
        perm = sortperm(params)
        ylabel --> "Function value"
        seriestype --> :scatter
        @series begin
            xlabel --> ho.params[i]
            subplot --> i
            label --> "Sampled points"
            legend --> false
            logscale(params) && (xscale --> :log10)
            logscale(ho.results) && (yscale --> :log10)
            params[perm], ho.results[perm]
        end

    end
end

function robust_acq(model)
    acqfunc = BayesianOptimization.acquisitionfunction(BayesianOptimization.ExpectedImprovement(maximum(model.y)), model)
    map(1:size(model.x,2)) do i
        try
            return acqfunc(model.x[:,i])
        catch
            return -Inf
        end
    end
end

@recipe function plot(s::GPSampler)
    model = s.model
    ndims = size(model.x,1)
    ms, var = GaussianProcesses.predict_f(model, model.x)
    sig = sqrt.(var)
    aqy = robust_acq(model)
    layout := 2ndims
    seriestype := :scatter
    xvals = from_logspace(eachrow(model.x), s.logdims)
     for i = 1:ndims
         @series begin
             subplot := i
             label := "Model function"
             yerror := sqrt.(var)
             xvals[i], Int(s.sense)*ms
         end
         @series begin
             logscale(Int(s.sense)*model.y[:]) && (yscale --> :log10)
             subplot := i
             label := "Obs"
             markershape := :cross
             xvals[i], Int(s.sense)*model.y[:]
         end
         @series begin
             subplot := i + ndims
             lab="Aq function"
             xvals[i], aqy
         end
     end
end
