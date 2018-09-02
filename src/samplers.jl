export RandomSampler, TreeSampler, ForestSampler, model


"""
Sample a value For each parameter uniformly at random from the candidate vectors. Log-uniform sampling available by providing a log-spaced candidate vector.
"""
struct RandomSampler <: Sampler end
function (s::RandomSampler)(ho)
    [list[rand(1:length(list))] for list in ho.candidates]
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
