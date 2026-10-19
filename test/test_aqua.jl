using Aqua

@testset "Aqua" begin
    # The check for type piracy is disabled until the methods defined on MultiKDE types are removed,
    # see https://github.com/baggepinnen/Hyperopt.jl/issues/104
    Aqua.test_all(Hyperopt; piracies = false)
end
