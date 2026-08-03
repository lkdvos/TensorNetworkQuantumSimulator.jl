# The memory-bounded "blocked" message update, on the host.
#
# `Algorithm("contract")` is the definition of the message this is supposed to be computing, so that
# is what it is checked against -- across vertex degrees, index orders and bond dimensions, which is
# where the kernel's three source cases and its scheduling live. `test_gpu_paths.jl` covers the same
# kernel on a device array type but cannot afford this breadth.
@eval module $(gensym())
using ITensors: ITensors, Index, dim, inds, permute
using Random
using TensorNetworkQuantumSimulator
using Test: @test, @testset
const TNQS = TensorNetworkQuantumSimulator
using TensorNetworkQuantumSimulator: Algorithm, norm

# Replay a schedule symbolically, assert every op is legal where it is issued, and return the number
# of permutations. This is the contract the kernel relies on: a gemm's leg is at an end of the current
# layout, a permute is consistent with the layout it claims to produce, the close sees no unabsorbed
# messages and a layout it accepts, and nothing reads the ket's block as a matrix unless that block is
# a contiguous run of storage -- which it is only when the cut leg is trailing.
function replay(n::Int, sliced::Int, msglegs::Vector{Int})
    ops = TNQS._message_schedule(n, sliced, msglegs)
    layout = ntuple(identity, n)
    remaining = Set(eachindex(msglegs))
    insrc, srcusable, npermutes = true, sliced == n, 0
    korder = Tuple(q for q in 1:n if q != sliced)
    for op in ops
        if op.kind === :permute
            @test sort(collect(op.perm)) == collect(1:n)
            @test ntuple(k -> layout[op.perm[k]], n) == op.layout
            layout, insrc = op.layout, false
            npermutes += 1
        elseif op.kind === :gemm
            @test !insrc || srcusable
            @test layout[op.side === :front ? 1 : n] == msglegs[op.msg]
            @test op.msg in remaining
            delete!(remaining, op.msg)
        else
            @test op.kind === :close
            @test isempty(remaining)
            @test !insrc || srcusable
            @test layout == (op.side === :kb ? (korder..., sliced) : (sliced, korder...))
        end
    end
    @test isempty(remaining)
    @test ops[end].kind === :close
    return npermutes
end

# Worst relative deviation from the "contract" path over every edge, for a few block widths. b = 3
# does not divide the bond dimensions used below, so the trailing partial block is covered too.
function worst_error(bpc, bs = (1, 3); edges = TNQS.edges(bpc))
    worst = 0.0
    for e in edges
        ref, _ = TNQS.updated_message(TNQS.set_default_kwargs(Algorithm("contract"), bpc), bpc, e)
        for b in bs
            got, _ = TNQS.updated_message(
                TNQS.set_default_kwargs(Algorithm("blocked"; b), bpc), bpc, e
            )
            @test issetequal(inds(got), inds(ref))
            worst = max(worst, norm(got - ref) / norm(ref))
        end
    end
    return worst
end

# A state whose bonds do not all have the same dimension, cycling through `dims` over the edges.
function ragged_state(elt, g, dims)
    s = siteinds("S=1/2", g)
    es = collect(TNQS.edges(g))
    l = Dict(e => Index(dims[mod1(i, length(dims))]) for (i, e) in enumerate(es))
    merge!(l, Dict(reverse(e) => l[e] for e in es))
    vs = collect(TNQS.vertices(g))
    ts = TNQS.Dictionary(
        vs, [
            ITensors.random_itensor(
                    elt, vcat(s[v], [l[TNQS.NamedEdge(v => vn)] for vn in TNQS.neighbors(g, v)])
                ) for v in vs
        ]
    )
    return TensorNetworkState(TensorNetwork(ts, g), s)
end

# Randomise the stored index order of every vertex tensor: that is what selects between the kernel's
# three source cases (contiguous block / strided gather / aligned copy).
function shuffle_orders!(bpc)
    tn = TNQS.network(bpc)
    for v in TNQS.vertices(tn)
        is = collect(inds(tn[v]))
        TNQS.setindex_preserve!(tn, permute(tn[v], shuffle(is)...), v)
    end
    return bpc
end

@testset "Blocked message update" begin
    # Labels are stored positions; 4 modes stands for a degree-3 vertex with one site index. The
    # permutation counts are the point of the schedule -- the unscheduled version did three per block
    # plus a factor-sized copy -- so they are asserted, not just the schedules' validity.
    @testset "schedule" begin
        @test replay(4, 4, [1, 3]) == 1        # (l_a, s, l_b, l_e): absorb l_a free, fuse l_b
        @test replay(4, 4, [1, 2]) == 2        # (l_a, l_b, s, l_e): l_b is boxed in, no fusion
        @test replay(4, 4, [2, 3]) == 2        # (s, l_a, l_b, l_e): nothing is free to start with
        @test replay(4, 1, [2, 4]) == 2        # (l_e, l_a, s, l_b): the block must be gathered first
        @test replay(3, 3, [1, 2]) == 1        # degree 3, no site index
        @test replay(2, 2, [1]) == 0           # degree 2, nothing to move
        @test replay(4, 4, Int[]) == 0         # degree 1: the close reads the block directly
        @test replay(5, 5, [1, 2, 4]) == 2     # degree 4
    end

    # A cut leg in the middle of the stored order cannot be matricized for the close, so the ket is
    # permuted into a copy -- and that order is ours to choose, so it is chosen to leave the block loop
    # with one permutation rather than two.
    @testset "aligned order" begin
        for (n, sliced, legs) in ((4, 2, [1, 3]), (4, 3, [1, 2]), (5, 3, [1, 2, 4]))
            perm = TNQS._align_perm(n, sliced, legs)
            @test sort(perm) == collect(1:n)
            @test perm[end] == sliced
            mapped = Int[findfirst(==(q), perm) for q in legs]
            @test replay(n, n, mapped) == (length(legs) > 2 ? 2 : 1)
        end
    end

    @testset "memory bound" begin
        # `max_scratch` is a fraction of one factor and the two block buffers are 2b/χ_e of one, so
        # b = max_scratch·χ_e/2 -- at least 1, at most χ_e.
        @test TNQS.blocked_blocksize(1 / 8, 1024) == 64
        @test TNQS.blocked_blocksize(1 / 8, 64) == 4
        @test TNQS.blocked_blocksize(1 / 8, 4) == 1          # would round to 0
        @test TNQS.blocked_blocksize(4.0, 8) == 8            # would exceed χ_e

        # What the peak actually is: two block buffers and the output, and a factor-sized aligned copy
        # only on the edges whose cut leg cannot be matricized. S=2, χ=1024, degree 3, b=64.
        nket, chie, b = 2 * 1024^3, 1024, 64
        nblock = div(nket, chie) * b
        @test TNQS.message_scratch_length(nket, nblock, chie, false) == 2 * nblock + chie^2
        @test TNQS.message_scratch_length(nket, nblock, chie, true) ==
            nket + 2 * nblock + chie^2
        @test 2 * nblock / nket == 1 / 8                     # the requested fraction, as promised
    end

    @testset "agreement with contract" begin
        Random.seed!(1234)
        hex = named_hexagonal_lattice_graph(2, 2)            # degrees 2 and 3
        for elt in (ComplexF64, ComplexF32)
            tol = real(elt) == Float32 ? 1.0f-4 : 1.0e-12
            ψ = random_tensornetworkstate(elt, hex; bond_dimension = 4)
            bpc = update(BeliefPropagationCache(ψ); maxiter = 3, tolerance = nothing)
            @test worst_error(bpc) < tol
            @test worst_error(shuffle_orders!(bpc)) < tol
        end

        # Degree 4: three messages to absorb, so the schedule permutes more than once and the
        # ping-pong between the two block buffers actually turns over.
        ψ = random_tensornetworkstate(ComplexF64, named_grid((3, 3)); bond_dimension = 3)
        bpc = update(BeliefPropagationCache(ψ); maxiter = 3, tolerance = nothing)
        @test worst_error(bpc) < 1.0e-12

        # Unequal bond dimensions, which the previous kernel refused outright.
        bpc = update(
            BeliefPropagationCache(ragged_state(ComplexF64, hex, (2, 3, 5)));
            maxiter = 3, tolerance = nothing
        )
        @test length(unique(dim(only(TNQS.virtualinds(TNQS.network(bpc), e))) for e in TNQS.edges(bpc))) > 1
        @test worst_error(bpc) < 1.0e-12
    end

    @testset "fallbacks" begin
        Random.seed!(7)
        # A single-layer network is a different contraction (its messages are rank 1), so this has to
        # hand back to "contract" -- bit-identically, since it *is* the contract path -- rather than
        # misread the factors.
        tn = random_tensornetwork(ComplexF64, named_hexagonal_lattice_graph(2, 2); bond_dimension = 3)
        bpc = update(BeliefPropagationCache(tn); maxiter = 2, tolerance = nothing)
        @test worst_error(bpc, (2,); edges = collect(TNQS.edges(bpc))[1:2]) == 0.0

        # Block-sparse storage breaks both the flat-buffer views and `dag` being plain conjugation.
        # The package builds no QN networks, so the guard is checked on the tensor directly.
        qi = Index([ITensors.QN(0) => 1, ITensors.QN(1) => 1])
        @test !TNQS._is_dense(ITensors.random_itensor(ComplexF64, qi, ITensors.dag(qi')))
        @test TNQS._is_dense(ITensors.random_itensor(ComplexF64, Index(2), Index(2)))
    end

    @testset "full update" begin
        Random.seed!(13)
        ψ = random_tensornetworkstate(
            ComplexF64, named_hexagonal_lattice_graph(2, 2); bond_dimension = 4
        )
        bpc = BeliefPropagationCache(ψ)
        plain = update(bpc; maxiter = 4, tolerance = nothing)
        # Once with a block size that forces many blocks, once with the default, which has to work
        # without being told anything.
        for alg in (Algorithm("blocked"; b = 2), Algorithm("blocked"))
            blocked = update(bpc; maxiter = 4, tolerance = nothing, message_update_alg = alg)
            @test maximum(
                TNQS.message_diff(TNQS.message(blocked, e), TNQS.message(plain, e))
                    for e in TNQS.edges(blocked)
            ) < 1.0e-12
        end
    end
end
end
