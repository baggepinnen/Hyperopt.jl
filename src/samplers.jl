export RandomSampler, TreeSampler

abstract type Sampler end
"""
Sample a value for each parameter uniformly at random from the candidate vectors. Log-uniform sampling available by providing a log-spaced candidate vector.
"""
struct RandomSampler <: Sampler end
(s::RandomSampler)(ho) = [list[rand(1:length(list))] for list in ho.candidates]

"""
```julia
random_init::Int = 5 # Number of initial function with random sampling before a tree is built
n_samples::Int = 3   # Average number of samples per leaf
n_tries::Int = 20    # How many random samples to query the tree model with before selecting a sample
```
"""
@with_kw struct TreeSampler <: Sampler
    random_init::Int = 5
    n_samples::Int = 3
    n_tries::Int = 20
end

function (s::TreeSampler)(ho)
    rs =  RandomSampler()
    (length(ho) < s.random_init || length(ho) % s.random_init == 0) && return rs(ho)

    A = Float64.(hcat(ho.history...)')
    y = Float64.(ho.results)
    model = build_tree(y, A, s.n_samples)
    bestsample = rs(ho)
    bestres = apply_tree(model, bestsample)
    for t in 2:s.n_tries
        sample = rs(ho)
        res = apply_tree(model, sample)
        if res <= bestres
            bestsample = sample
            bestres = res
        end
    end
    return bestsample
end
