using Test, Random
Random.seed!(0)


using Hyperopt
f(a,b=true;c=10) = sum(@. 100 + (a-3)^2 + (b ? 10 : 20) + (c-100)^2)

# res = map(1:30) do i
#     @info("Iteration ", i)
hor = @hyperopt for i=100, sampler=RandomSampler(), a = LinRange(1,5,50), b = [true, false], c = exp10.(LinRange(-1,3,50))
    # println(i, "\t", a, "\t", b, "\t", c)
    # print(i, " ")
    f(a,b,c=c)
end
@test minimum(hor)[2] < 300

horp = @phyperopt for i=100, sampler=RandomSampler(), a = LinRange(1,5,50), b = [true, false], c = exp10.(LinRange(-1,3,50))
    # println(i, "\t", a, "\t", b, "\t", c)
    # print(i, " ")
    f(a,b,c=c)
end
@test minimum(horp)[2] < 300

hob = @hyperopt for i=100, sampler=BlueNoiseSampler(), a = LinRange(1,5,100), b = repeat([true, false],50), c = exp10.(LinRange(-1,3,100))
    # println(i, "\t", a, "\t", b, "\t", c)
    # print(i, " ")
    f(a,b,c=c)
end
@test minimum(hob)[2] < 300


#     minimum.((hor,hob,hot,hof))
# end

# hor,hob,hot,hof = getindex.(res,1),getindex.(res,2),getindex.(res,3),getindex.(res,4)
# hor,hob,hot,hof = getindex.(hor,2),getindex.(hob,2),getindex.(hot,2),getindex.(hof,2)
# res = [hor hob hot hof]
# sum(log.(res), dims=1)

ho = Hyperoptimizer(10, a = range(1, stop=2, length=50), b = [true, false], c = randn(100))
for (i,a,b,c) in ho
    println(i, "\t", a, "\t", b, "\t", c)
end



ho = @hyperopt for i=100, a = LinRange(1,5,50), b = [true, false], c = exp10.(LinRange(-1,3,50)), d = [tanh, exp]
    # println(i, "\t", a, "\t", b, "\t", c)
    # print(i, " ")
    f(a,b,c=c) + d(a)
end
