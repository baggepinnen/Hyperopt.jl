export RandomSampler, TreeSampler, ForestSampler, BlueNoiseSampler, model


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
    s.samples != zeros(0,0) && return
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
    dims = 2,
    nsamples = 100,
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
```julia
random_init::Int = 5 # Number of initial function with random sampling before a tree is built
samples_per_leaf::Int = 3   # Average number of samples per leaf
n_tries::Int = 20    # How many random samples to query the tree model with before selecting a sample
```
"""
@with_kw struct TreeSampler <: Sampler
    random_init::Int = 5
    samples_per_leaf::Int = 3
    n_tries::Int = 20
end


function model(s::TreeSampler, ho::Hyperoptimizer)
    A = copy(hcat(ho.history...)')
    y = Float64.(ho.results)
    model = build_tree(y, A, s.samples_per_leaf)
end


@with_kw struct ForestSampler <: Sampler
    random_init::Int = 5
    n_tries::Int = 20
    n_trees::Int = 5
    n_subfeatures::Int = -1
    partial_sampling::Float64 = 0.7
    max_depth::Int = -1
    min_samples_leaf::Int = 5
    min_samples_split::Int = 2
    min_purity_increase::Float64 = 0.0
end


function model(s::ForestSampler, ho::Hyperoptimizer)
    A = copy(hcat(ho.history...)')
    y = Float64.(ho.results)
    model = build_forest(y, A, s.n_subfeatures, s.n_trees, s.partial_sampling, s.max_depth, s.min_samples_leaf, s.min_samples_split, s.min_purity_increase)
end

for T in [:TreeSampler, :ForestSampler]
    apply = T == :TreeSampler ? apply_tree : apply_forest
    @eval function (s::($(T)))(ho::Hyperoptimizer)
        rs =  RandomSampler()
        length(ho) < s.random_init && return rs(ho)

        model_ = model(s, ho)
        bestsample = rs(ho)
        bestres = $(apply)(model_, bestsample)
        for t in 2:s.n_tries
            sample = rs(ho)
            res = $(apply)(model_, sample)
            if res <= bestres
                bestsample = sample
                bestres = res
            end
        end
        return bestsample
    end
end
