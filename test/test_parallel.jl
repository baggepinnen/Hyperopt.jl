using Test, Distributed, Random, Serialization, Hyperopt

# The workers added here are removed at the end of this file
added_workers = nprocs() < 3 ? addprocs(3 - nprocs()) : Int[]
@everywhere using Hyperopt

# Bytes read and written by the calling process, including the communication with other processes
@everywhere function parallel_test_io_bytes()
    n = 0
    for line in eachline("/proc/self/io")
        key, value = split(line, ": ")
        key in ("rchar", "wchar") && (n += parse(Int, value))
    end
    n
end

# Interrupts the calling task after `delay` seconds, as Ctrl-C does, unless `running[]` is false. The task that is
# created with `@async` runs on the thread of the calling task, and only while the calling task waits.
function interrupt_after(delay)
    caller = current_task()
    running = Ref(true)
    @async begin
        sleep(delay)
        running[] && schedule(caller, InterruptException(); error = true)
    end
    running
end

@testset "Parallel macros" begin
    @info "Testing parallel macros with $(nworkers()) workers and $(Threads.nthreads()) threads"

    @testset "tmap" begin
        @test Hyperopt.tmap(x -> 2x, 1:100) == 2 .* (1:100)
        @test Hyperopt.tmap(+, 1:10, 11:20) == collect(12:2:30)
        @test Hyperopt.tmap(identity, Int[]) == []
        # An iterator of unknown length is consumed in order
        @test Hyperopt.tmap(x -> x^2, Iterators.filter(iseven, 1:20)) == (2:2:20) .^ 2
        @test Hyperopt.tmap(x -> x, 1:10; ntasks = 3) isa Vector{Int}

        # After an error, no further elements are taken
        started = Threads.Atomic{Int}(0)
        err = try
            Hyperopt.tmap(1:100; ntasks = 2) do x
                Threads.atomic_add!(started, 1)
                sleep(0.01)
                x == 3 && error("failure")
                x
            end
        catch e
            e
        end
        @test err isa TaskFailedException
        @test err.task.exception isa ErrorException
        @test started[] < 10
    end

    @testset "History types and reproducibility" begin
        for sampler in (() -> RandomSampler(Xoshiro(1)), () -> LHSampler(rng = Xoshiro(1)))
            hos = map((:serial, :distributed, :threaded)) do kind
                s = sampler()
                if kind === :serial
                    @hyperopt for i = 30, sampler = s, a = LinRange(1, 5, 30), b = repeat([true, false], 15)
                        a + b
                    end
                elseif kind === :distributed
                    @phyperopt for i = 30, sampler = s, a = LinRange(1, 5, 30), b = repeat([true, false], 15)
                        a + b
                    end
                else
                    @thyperopt for i = 30, sampler = s, a = LinRange(1, 5, 30), b = repeat([true, false], 15)
                        a + b
                    end
                end
            end
            ho, hop, hot = hos
            @test hop.history == ho.history
            @test hot.history == ho.history
            @test hop.results == hot.results == ho.results
            @test typeof(hop.minimizer) == typeof(hot.minimizer) == typeof(ho.minimizer)
            @test all(h -> h isa Vector, hop.history)
            @test all(h -> h isa Vector, hot.history)
        end
    end

    @testset "Iteration index, state and evaluation on workers" begin
        ho = @phyperopt for i = 20, a = 1:20
            sleep(0.01)
            (i, state === nothing, myid())
        end
        @test [r[1] for r in ho.results] == 1:20 # Recorded in the order of the iterations
        @test all(r -> r[2], ho.results)
        @test issubset(Set(r[3] for r in ho.results), workers())
        @test length(unique(r[3] for r in ho.results)) == min(nworkers(), 20)

        ho = @thyperopt for i = 20, a = 1:20
            sleep(0.01)
            (i, state === nothing, Threads.threadid())
        end
        @test [r[1] for r in ho.results] == 1:20
        @test all(r -> r[2], ho.results)
        Threads.nthreads() > 1 && @test length(unique(r[3] for r in ho.results)) > 1

        # The loop body refers to a local variable of a function
        function phyperopt_inside_function(c)
            @phyperopt for i = 10, a = 1:10
                a + c
            end
        end
        ho = phyperopt_inside_function(100)
        @test ho.results == [h[1] + 100 for h in ho.history]

        # The custom pmap may use a `CachingPool`
        ho = @phyperopt for i = 10, a = 1:10
            a
        end (f, itr) -> pmap(f, CachingPool(workers()), itr)
        @test ho.results == first.(ho.history)
    end

    @testset "Hyperband" begin
        # With R = 9 and η = 3, Hyperband evaluates 17 new configurations and resumes 5 of them. The state is the
        # vector of the budgets with which the configuration has been evaluated.
        hos = map((:serial, :distributed, :threaded)) do kind
            s = Hyperband(η = 3, inner = RandomSampler(Xoshiro(2)))
            if kind === :serial
                @hyperopt for r = 9, sampler = s, a = LinRange(1, 5, 100), b = [1, 2]
                    budgets = state === nothing ? [r] : [state; r]
                    (a - 3)^2 + b / r, budgets
                end
            elseif kind === :distributed
                @phyperopt for r = 9, sampler = s, a = LinRange(1, 5, 100), b = [1, 2]
                    budgets = state === nothing ? [r] : [state; r]
                    (a - 3)^2 + b / r, budgets
                end
            else
                @thyperopt for r = 9, sampler = s, a = LinRange(1, 5, 100), b = [1, 2]
                    budgets = state === nothing ? [r] : [state; r]
                    (a - 3)^2 + b / r, budgets
                end
            end
        end
        ho, hop, hot = hos
        @test length(ho.results) == 22
        for h in (hop, hot)
            @test h.history == ho.history
            @test h.results == ho.results
            @test h.sampler.budgets == ho.sampler.budgets
            @test h.sampler.minimum == ho.sampler.minimum
            @test h.minimizer == ho.minimizer
        end
        @test ho.sampler.minimum.state in ([1.0, 3.0, 9.0], [3.0, 9.0], [9.0])

        # The rungs are evaluated with the custom pmap
        rungs = Ref(0)
        counting_pmap = (f, itr) -> (rungs[] += 1; pmap(f, itr))
        hop = @phyperopt for r = 9, sampler = Hyperband(η = 3), a = LinRange(1, 5, 100)
            (a - 3)^2 + 1 / r
        end counting_pmap
        @test rungs[] == 3 + 2 + 1
        @test length(hop.results) == 22

        # With `on_error`, the value returned for a failed evaluation is its result
        hop = @phyperopt for r = 9, sampler = Hyperband(η = 3, inner = RandomSampler(Xoshiro(3))), a = LinRange(1, 5, 100)
            a > 4 && error("failure")
            (a - 3)^2 + 1 / r
        end (f, itr) -> pmap(f, itr; on_error = e -> Inf)
        @test length(hop.results) == 22
        @test any(isinf, hop.results)
        @test all(k -> isinf(hop.results[k]) == (hop.history[k][1] > 4), eachindex(hop.results))

        # The function interface accepts `mapfun`
        fun = (r, pars) -> ((pars[1] - 3)^2 + 1 / r, pars)
        hof = hyperband(fun, [LinRange(1, 5, 100)]; R = 9, η = 3, mapfun = pmap)
        @test length(hof.results) == 22
        @test hof.minimum < 1

        # An interruption aborts Hyperband, the evaluations of the completed rungs are recorded
        hop = @test_logs (:info, "Aborting hyperband") match_mode = :any @phyperopt for r = 9, sampler = Hyperband(η = 3), a = LinRange(1, 5, 100)
            r == 9 && throw(InterruptException())
            (a - 3)^2 + 1 / r
        end
        @test length(hop.history) == length(hop.results) == length(hop.sampler.budgets) == 9 + 3
        hot = @test_logs (:info, "Aborting hyperband") match_mode = :any @thyperopt for r = 9, sampler = Hyperband(η = 3), a = LinRange(1, 5, 100)
            r == 9 && throw(InterruptException())
            (a - 3)^2 + 1 / r
        end
        @test length(hot.history) == length(hot.results) == length(hot.sampler.budgets) == 9 + 3
    end

    @testset "Errors do not interrupt other work" begin
        # A computation on the worker w1 that is unrelated to the optimizations, which use only the worker w2 or threads.
        # The computation creates the file `started` and runs until the file `stop` exists. The allocations are
        # safepoints at which an interrupt of w1 would be delivered.
        w1, w2 = workers()[1:2]
        started, stop = tempname(), tempname()
        unrelated = remotecall(w1, started, stop) do started, stop
            touch(started)
            t0 = time()
            while !isfile(stop) && time() - t0 < 120
                rand(10)
            end
            isfile(stop) ? :completed : :timeout
        end
        timedwait(() -> isfile(started), 60)
        @test_throws RemoteException @phyperopt for i = 10, a = 1:10
            error("failure")
        end (f, itr) -> pmap(f, WorkerPool([w2]), itr)
        @test_throws TaskFailedException @thyperopt for i = 10, a = 1:10
            error("failure")
        end
        touch(stop)
        @test fetch(unrelated) === :completed
        rm(started); rm(stop)

        ho = @phyperopt for i = 10, a = 1:10
            a
        end
        @test_throws RemoteException @phyperopt for i = 20, ho = ho, a = 1:10
            sleep(0.01)
            i == 5 && error("failure")
            a
        end
        # The results obtained before the error are recorded
        @test 10 <= length(ho.results) < 30
        @test ho.results == first.(ho.history)

        @test_throws TaskFailedException @thyperopt for i = 20, ho = ho, a = 1:10
            sleep(0.01)
            i == 5 && error("failure")
            a
        end
        @test ho.results == first.(ho.history)

        # Failed evaluations are not recorded if the custom pmap handles errors
        ho = @phyperopt for i = 20, a = 1:10
            isodd(i) && error("failure")
            a
        end (f, itr) -> pmap(f, itr; on_error = e -> nothing)
        @test length(ho.results) == 10
        @test ho.results == first.(ho.history)
        @test nworkers() >= 2
    end

    @testset "Interruption" begin
        # An InterruptException in the loop body aborts the optimization
        ho = @test_logs (:info, "Aborting hyperoptimization") match_mode = :any @phyperopt for i = 50, a = 1:50
            i == 10 && throw(InterruptException())
            a
        end
        @test length(ho.results) < 50
        @test ho.results == first.(ho.history)
        ho = @test_logs (:info, "Aborting hyperoptimization") match_mode = :any @thyperopt for i = 50, a = 1:50
            i == 10 && throw(InterruptException())
            a
        end
        @test length(ho.results) < 50
        @test ho.results == first.(ho.history)

        # An interruption of the calling task stops the optimization before all evaluations have completed, and no
        # further evaluations start. `pmap` may have received one configuration that it has not yet sent to a worker.
        @everywhere parallel_test_started = Ref(0)
        @everywhere parallel_test_count!() = (parallel_test_started[] += 1; nothing)
        duration = 10 # Duration of a complete run in seconds
        N = round(Int, duration * nworkers() / 0.05)
        running = interrupt_after(1)
        t = @elapsed ho = @phyperopt for i = N, a = 1:N
            parallel_test_count!()
            sleep(0.05)
            a
        end
        running[] = false
        @test length(ho.results) < N
        @test ho.results == first.(ho.history)
        @test t < duration / 2
        sleep(0.5) # The running evaluations complete
        started = sum(w -> remotecall_fetch(() -> parallel_test_started[], w), workers())
        sleep(0.5)
        @test sum(w -> remotecall_fetch(() -> parallel_test_started[], w), workers()) == started
        @test started <= length(ho.results) + nworkers() + 1

        started = Threads.Atomic{Int}(0)
        N = round(Int, duration * Threads.nthreads() / 0.05)
        running = interrupt_after(1)
        t = @elapsed ho = @thyperopt for i = N, a = 1:N
            Threads.atomic_add!(started, 1)
            sleep(0.05)
            a
        end
        running[] = false
        @test length(ho.results) < N
        @test ho.results == first.(ho.history)
        @test t < duration / 2
        sleep(0.5)
        @test started[] <= length(ho.results) + Threads.nthreads()
    end

    @testset "Resumption after an incomplete iteration" begin
        for kind in (:distributed, :threaded)
            ho = Hyperoptimizer(5, a = 1:10)
            for (i, a) in ho
                push!(ho.results, a)
            end
            iterate(ho) # A sample without a result, e.g., of a manual iteration that was stopped
            @test length(ho.history) == length(ho.results) + 1
            if kind === :distributed
                @phyperopt for i = 5, ho = ho, a = 1:10
                    a
                end
            else
                @thyperopt for i = 5, ho = ho, a = 1:10
                    a
                end
            end
            @test length(ho.history) == length(ho.results) == 10
            @test ho.results == first.(ho.history)
        end
    end

    @testset "Communication per evaluation" begin
        # The function and the elements that pmap sends to the workers have sizes that are independent of the number
        # of evaluations
        serialized_size(x) = (io = IOBuffer(); serialize(io, x); position(io))
        recorded = Dict{Int, Tuple{Int, Int}}()
        for N in (10, 1000)
            fsize = Ref(0)
            elementsizes = Int[]
            recording_pmap = function (f, itr)
                fsize[] = serialized_size(f)
                pmap(f, Iterators.map(x -> (push!(elementsizes, serialized_size(x)); x), itr))
            end
            ho = @phyperopt for i = N, a = LinRange(1, 5, 1000), b = [true, false]
                a + b
            end recording_pmap
            @test length(ho.results) == N
            recorded[N] = (fsize[], maximum(elementsizes))
        end
        @test recorded[10][1] == recorded[1000][1]
        @test recorded[1000][2] <= recorded[10][2] + 4 # The iteration index may require more bytes

        # The data that the workers receive and send per evaluation does not increase with the number of evaluations
        if Sys.islinux()
            function bytes_per_evaluation(N)
                before = [remotecall_fetch(parallel_test_io_bytes, w) for w in workers()]
                ho = @phyperopt for i = N, a = LinRange(1, 5, 1000), b = [true, false]
                    a + b
                end
                after = [remotecall_fetch(parallel_test_io_bytes, w) for w in workers()]
                @test length(ho.results) == N
                sum(after .- before) / N
            end
            bytes_per_evaluation(10) # Compilation
            b100, b1000 = bytes_per_evaluation(100), bytes_per_evaluation(1000)
            @info "Data transferred by the workers per evaluation" b100 b1000
            @test 100 < b1000 < 1.5 * b100
            @test b1000 < 5000
        end
    end
end

isempty(added_workers) || rmprocs(added_workers)
