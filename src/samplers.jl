"""
Sample a value For each parameter uniformly at random from the candidate vectors. Log-uniform sampling available by providing a log-spaced candidate vector.
"""
struct RandomSampler <: Sampler end
function (s::RandomSampler)(ho)
    [list[rand(1:length(list))] for list in ho.candidates]
end

"""
Try to spread out parameters as blue noise (only high frequency). Should sample the space better than random sampling. Use this if you intend to run less than, say, 2000.
"""
@with_kw mutable struct BlueNoiseSampler <: Sampler
    samples = zeros(0,0)
    orders = Int[]
end
function init!(s::BlueNoiseSampler, ho)
    s.samples != zeros(0,0) && return # Already initialized
    dims = length(ho.candidates)
    s.samples = bluenoise(dims = dims, nsamples = ho.iterations)
    s.orders = [sortperm(s.samples[dim,:]) for dim = 1:dims]
end

function (s::BlueNoiseSampler)(ho)
    init!(s, ho)
    iter = length(ho.history)+1
    [list[s.orders[dim][iter]] for (dim,list) in enumerate(ho.candidates)]
end

function bluenoise(;
    dims,
    nsamples,
    iters = 100,
    points = rand(dims, nsamples),
    α = 0.005
    )

    n          = nsamples = size(points,2)
    dims       = size(points,1)
    rdmax      = [1/(2n), sqrt(1/(2sqrt(3)*n)), (1/(4sqrt(2)*n))^(1/3), (1/(8n))^(1/4)]
    w          = 1 ./ rdmax
    w        ./= sum(w)
    fadelength = iters ÷ 2
    fadein(i)  = min(i,fadelength)*w[1]/fadelength
    points2    = copy(points)
    grad       = zeros(dims)
    d          = zeros(dims)
    for i = 1:iters
        i == iters ÷ 2 && (α *= 0.2)
        perms = [sortperm(points[dim,:]) for dim = 1:dims]
        for p1 = 1:nsamples
            grad .*= 0
            for dim = 1:dims
                optimal = (-0.5 + perms[dim][p1])/n
                grad[dim] += fadein(i)*sign(optimal - points[dim,p1])
            end
            for p2 = 1:nsamples
                d .= points[:,p1] .- points[:,p2]
                nd = norm(d)
                # nd > 0.2 && continue
                grad .+= w[dims]*d./n/(0.001 + nd)^3
            end
            points2[:,p1] .+= α*grad
        end
        clamp!(points2,0.,1.)
        copyto!(points, points2)
    end
    points
end




"""
Sample from a latin hypercube
"""
@with_kw mutable struct LHSampler <: Sampler
    samples = zeros(0,0)
end
function init!(s::LHSampler, ho)
    s.samples != zeros(0,0) && return # Already initialized
    ndims = length(ho.candidates)
    X, fit = LHCoptim(ho.iterations,ndims,1000)
    s.samples = copy(X')
end

function (s::LHSampler)(ho)
    init!(s, ho)
    iter = length(ho.history)+1
    [list[s.samples[dim,iter]] for (dim,list) in enumerate(ho.candidates)]
end

"""
    CLHSampler(dims=[Continuous(), Categorical(2), ...])
Sample from a categorical/continuous latin hypercube. All continuous variables must have the same length of the candidate vectors.
"""
@with_kw mutable struct CLHSampler <: Sampler
    samples = zeros(0,0)
    dims = []
end
function init!(s::CLHSampler, ho)
    s.samples != zeros(0,0) && return # Already initialized
    ndims = length(ho.candidates)
    initialSample = randomLHC(ho.iterations,s.dims)
    X,_ = LHCoptim!(initialSample, 500, dims=s.dims)
    s.samples = copy(X')
end

function (s::CLHSampler)(ho)
    init!(s, ho)
    iter = length(ho.history)+1
    [list[s.samples[dim,iter]] for (dim,list) in enumerate(ho.candidates)]
end
