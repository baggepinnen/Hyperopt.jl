using Printf, Statistics
pp(x;kwargs...) = scatter(x[1,:], x[2,:]; kwargs...)

function dists(points)
    dims,n = size(points)
    dists = fill(Inf, n)
    for i = 1:n
        for j = 1:n
            i == j && continue
            d = norm(points[:,i] - points[:,j])
            dists[i] = min(d, dists[i])
        end
    end
    dists
end

function bluenoise(;
    dims = 2,
    nsamples = 100,
    iters = 10,
    points = rand(dims, nsamples)
    )
    dd = dists(points)
    @printf("Initial dist dist mean: %8.3g std: %8.3g\n", mean(dd), std(dd))
    hist1 = histogram(dd, bins=20, title= "Initial distance to closest")
    n = nsamples
    rdmax = [1/(2n), sqrt(1/(2sqrt(3)*n)), (1/(4sqrt(2)*n))^(1/3), (1/(8n))^(1/4)]
    w = 1 ./ rdmax
    w ./= sum(w)
    fadelength = iters ÷ 2
    fadein(i) = min(i,fadelength)*w[1]/fadelength
    fig1 = pp(points, title="Uniform random")
    scatter!(zeros(nsamples), points[2,:], m=(2,))
    scatter!(points[1,:], zeros(nsamples), m=(2,)) |> display
    points2 = copy(points)
    α = 0.005
    grad = zeros(dims)
    d = zeros(dims)
    @progress for i = 1:iters
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
        # if i % 2 == 0
        #     pp(points, xlims=(0,1), ylims=(0,1))
        #     scatter!(zeros(nsamples), points[2,:], m=(2,))
        #     scatter!(points[1,:], zeros(nsamples), m=(2,)) |> display
        # end
    end
    fig2 = pp(points, title="Optimized sample")
    scatter!(zeros(nsamples), points[2,:], m=(2,))
    scatter!(points[1,:], zeros(nsamples), m=(2,)) |> display
    dd = dists(points)
    @printf("Final dist dist mean: %8.3g std: %8.3g\n", mean(dd), std(dd))
    hist2 = histogram(dd, bins=20, title= "Optimized distance to closest")
    plot(fig1, fig2, hist1, hist2) |> display
    points
end

default(legend=false)
points = bluenoise(iters=100, nsamples=30)
# points = bluenoise(points=points, iters=10000, nsamples=20)

using StatPlots
histogram([std(dists(rand(1,20))) for i = 1:10000])

histogram([mean(dists(rand(1,20))) for i = 1:10000])
vline!([1/21 mean(diff((-0.5 .+ 1:20)./20)) mean(dists(points[1:1,:]))])
