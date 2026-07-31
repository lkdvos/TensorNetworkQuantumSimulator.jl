# The memory-bounded "blocked" message update, on the host.
#
# The kernel is a schedule of gemms and permutedims over blocks of one open leg, so there are two
# separate things to test: the schedule builder is pure index bookkeeping and is checked exhaustively
# below without any tensors, and the kernel itself is checked against `Algorithm("contract")`, which
# is the definition of the message it is supposed to be computing.
#
# `test_gpu_paths.jl` covers the same kernel on a device array type; what it cannot cover is the
# breadth here (index orders, degrees, fallbacks), which is cheap on the host.
@eval module $(gensym())
using ITensors: ITensors, Index, dim, inds, permute
using Random
using TensorNetworkQuantumSimulator
using Test: @test, @testset, @test_throws
const TNQS = TensorNetworkQuantumSimulator
using TensorNetworkQuantumSimulator: Algorithm, norm

# Replay a schedule symbolically and assert every op is legal at the point it is issued. This is the
# invariant the kernel relies on: a gemm's leg is at an end of the current layout, a permute is a
# permutation consistent with the layout it claims to produce, the close sees no unabsorbed messages
# and a layout the closing gemm accepts, and nothing reads the ket's slice as a matrix unless that
# slice is a contiguous run of storage (which it is only when the cut leg is trailing).
function replay(n::Int, sliced::Int, msglegs::Vector{Int})
    ops = TNQS._message_schedule(n, sliced, msglegs)
    ops === nothing && return nothing
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

# All messages as (l, l') matrices and the ket in its stored order, straight out of the cache.
function kernel_inputs(bpc, edge)
    tn = TNQS.network(bpc)
    v = TNQS.src(edge)
    T = tn[v]
    le = only(TNQS.virtualinds(tn, edge))
    is = collect(inds(T))
    ms = TNQS.incoming_messages(bpc, v; ignore_edges = (reverse(edge),))
    legs = [only(ITensors.commoninds(m, T)) for m in ms]
    return (
        A = ITensors.array(T),
        dims = Tuple(dim(i) for i in is),
        sliced = findfirst(==(le), is),
        msglegs = Int[findfirst(==(l), is) for l in legs],
        mats = [ITensors.array(ms[i], legs[i], ITensors.prime(ITensors.dag(legs[i]))) for i in eachindex(ms)],
        chie = dim(le),
    )
end

# Worst relative deviation from the "contract" path over every edge, for several block widths.
function worst_error(bpc, bs = (1, 3, 1000); alg_kwargs = (;))
    worst = 0.0
    for e in TNQS.edges(bpc)
        ref, _ = TNQS.updated_message(TNQS.set_default_kwargs(Algorithm("contract"), bpc), bpc, e)
        for b in bs
            got, _ = TNQS.updated_message(
                TNQS.set_default_kwargs(Algorithm("blocked"; b, alg_kwargs...), bpc), bpc, e
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

# Randomise the stored index order of every vertex tensor, which is what selects between the three
# source cases in the kernel (contiguous slice / strided gather / aligned copy).
function shuffle_orders!(bpc)
    tn = TNQS.network(bpc)
    for v in TNQS.vertices(tn)
        is = collect(inds(tn[v]))
        TNQS.setindex_preserve!(tn, permute(tn[v], shuffle(is)...), v)
    end
    return bpc
end

@testset "Blocked message update" begin
    @testset "schedule builder" begin
        # Exhaustive over mode counts, cut-leg positions and message-leg placements. Every case must
        # produce a legal schedule -- the kernel has no other way to handle a vertex.
        for n in 2:5, sliced in 1:n
            others = [q for q in 1:n if q != sliced]
            for nmsg in 0:(n - 1), legs in Iterators.product(ntuple(_ -> others, nmsg)...)
                v = collect(Int, legs)
                allunique(v) || continue
                @test replay(n, sliced, v) !== nothing
            end
        end

        # The counts that matter for cost, versus three block permutations per block before this was
        # scheduled. A trailing cut leg lets the first message absorb straight out of the ket's
        # storage; one further permutation then both exposes the second message leg and lands on a
        # layout the close accepts, but only if that leg sits at an end of the uncut order.
        @test replay(4, 4, [1, 3]) == 1          # (l_a, s, l_b, l_e)
        @test replay(4, 4, [1, 2]) == 2          # (l_a, l_b, s, l_e): l_b is boxed in
        @test replay(4, 1, [2, 4]) == 2          # (l_e, l_a, s, l_b): the slice must be gathered
        @test replay(2, 2, [1]) == 0             # degree 2, nothing to move
        @test replay(3, 3, [1, 2]) == 1
        @test replay(4, 4, Int[]) == 0           # degree 1: the close reads the slice directly

        # A vertex whose modes cannot be scheduled within the permutation budget must say so rather
        # than return something the kernel would misinterpret.
        @test TNQS._message_schedule(4, 4, [1, 3]; maxpermutes = 0) === nothing
    end

    @testset "aligned order is chosen by cost" begin
        # A cut leg in the middle of the stored order needs a permuted copy, and *that* order is ours
        # to pick -- so it should be picked to leave the block loop with as little to move as
        # possible. Every degree-3 placement must come out at one block permutation, which the
        # previous rule-based choice (message leg first, rest in stored order) did not manage.
        for n in 3:5, sliced in 2:(n - 1)
            others = [q for q in 1:n if q != sliced]
            for nmsg in 0:(n - 1), legs in Iterators.product(ntuple(_ -> others, nmsg)...)
                v = collect(Int, legs)
                allunique(v) || continue
                perm, ops = TNQS._align_schedule(n, sliced, v)
                @test ops !== nothing
                @test sort(perm) == collect(1:n)
                @test perm[end] == sliced          # the close has to be able to matricize it
                # Revalidate through the replay, on the aligned mode numbering.
                remapped = Int[findfirst(==(q), perm) for q in v]
                @test replay(n, n, remapped) == count(op -> op.kind === :permute, ops)
                n == 4 && nmsg == 2 && @test count(op -> op.kind === :permute, ops) == 1
            end
        end
    end

    @testset "block size from max_scratch" begin
        # `max_scratch` is a fraction of one factor and the two block buffers are 2b/χ_e of one, so
        # b = max_scratch·χ_e/2 -- clamped to at least 1 and at most χ_e.
        @test TNQS.blocked_blocksize(1 / 8, 1024) == 64
        @test TNQS.blocked_blocksize(1 / 8, 64) == 4
        @test TNQS.blocked_blocksize(1 / 8, 4) == 1          # would round to 0
        @test TNQS.blocked_blocksize(4.0, 8) == 8            # would exceed χ_e
        @test TNQS.blocked_blocksize(1 / 2, 32) == 8
    end

    @testset "agreement with contract" begin
        Random.seed!(1234)
        for g in (
                named_hexagonal_lattice_graph(2, 2),   # degrees 2 and 3
                named_grid((3, 3)),                    # degrees 2, 3 and 4
            )
            for elt in (Float64, ComplexF64, ComplexF32)
                tol = real(elt) == Float32 ? 1.0f-4 : 1.0e-12
                ψ = random_tensornetworkstate(elt, g; bond_dimension = 4)
                bpc = update(BeliefPropagationCache(ψ); maxiter = 3, tolerance = nothing)
                @test worst_error(bpc) < tol
                # b = 3 above does not divide χ = 4, so the trailing partial block is covered too.
                @test worst_error(shuffle_orders!(bpc)) < tol
            end
        end
    end

    @testset "unequal bond dimensions" begin
        Random.seed!(5)
        g = named_hexagonal_lattice_graph(2, 2)
        # A network the old kernel refused outright: it required every bond of a vertex to have the
        # same dimension, and inferred the site dimension from χ³ dividing the element count.
        ψ = ragged_state(ComplexF64, g, (2, 3, 5))
        bpc = update(BeliefPropagationCache(ψ); maxiter = 3, tolerance = nothing)
        dims = unique(dim(only(TNQS.virtualinds(TNQS.network(bpc), e))) for e in TNQS.edges(bpc))
        @test length(dims) > 1
        @test worst_error(bpc) < 1.0e-12
    end

    @testset "fallbacks" begin
        Random.seed!(7)
        g = named_hexagonal_lattice_graph(2, 2)

        # A single-layer network is a different contraction (its messages are rank 1), so the
        # algorithm has to hand back to "contract" rather than misread the factors.
        tn = random_tensornetwork(ComplexF64, g; bond_dimension = 3)
        bpc = update(BeliefPropagationCache(tn); maxiter = 2, tolerance = nothing)
        @test worst_error(bpc, (2,)) == 0.0        # bit-identical: it *is* the contract path

        # Block-sparse storage breaks both the flat-buffer view and `dag` being plain conjugation.
        # The package builds no QN networks, so the guard is checked on the tensor directly.
        qi = Index([ITensors.QN(0) => 1, ITensors.QN(1) => 1])
        @test !TNQS._is_dense(ITensors.random_itensor(ComplexF64, qi, ITensors.dag(qi')))
        @test TNQS._is_dense(ITensors.random_itensor(ComplexF64, Index(2), Index(2)))
    end

    @testset "kernel writes only its buffers" begin
        Random.seed!(11)
        g = named_hexagonal_lattice_graph(2, 2)
        ψ = random_tensornetworkstate(ComplexF64, g; bond_dimension = 4)
        bpc = update(BeliefPropagationCache(ψ); maxiter = 2, tolerance = nothing)

        for e in TNQS.edges(bpc)
            inp = kernel_inputs(bpc, e)
            n = length(inp.dims)
            inp.sliced == 1 || inp.sliced == n || continue   # no aligned copy in these cases
            b = 2
            nket = prod(inp.dims)
            nblock = div(nket, inp.chie) * b
            len = TNQS.message_scratch_length(nket, nblock, inp.chie, false)
            # The whole point of the layout invariant: two block buffers and the output, no
            # factor-sized term, whatever the degree.
            @test len == 2 * nblock + inp.chie^2

            ops = TNQS._message_schedule(n, inp.sliced, inp.msglegs)
            s = zeros(ComplexF64, len)
            ket = vec(inp.A)
            buf1 = view(s, 1:nblock)
            buf2 = view(s, (nblock + 1):(2 * nblock))
            outmat = reshape(view(s, (2 * nblock + 1):len), inp.chie, inp.chie)
            before = copy(inp.A)

            TNQS.blocked_message!(                                # warm up
                outmat, buf1, buf2, ket, inp.dims, inp.sliced, inp.mats, ops, b
            )
            # A buffer of exactly the advertised length is enough, the ket is only read, and the
            # steady-state call allocates only the handful of views it builds per block.
            allocs = @allocated TNQS.blocked_message!(
                outmat, buf1, buf2, ket, inp.dims, inp.sliced, inp.mats, ops, b
            )
            @test allocs < 4096
            @test inp.A == before
        end
    end

    @testset "full update" begin
        Random.seed!(13)
        g = named_hexagonal_lattice_graph(2, 2)
        ψ = random_tensornetworkstate(ComplexF64, g; bond_dimension = 4)
        bpc = BeliefPropagationCache(ψ)
        blocked = update(
            bpc; maxiter = 4, tolerance = nothing,
            message_update_alg = Algorithm("blocked"; b = 2)
        )
        plain = update(bpc; maxiter = 4, tolerance = nothing)
        @test maximum(
            TNQS.message_diff(TNQS.message(blocked, e), TNQS.message(plain, e))
                for e in TNQS.edges(blocked)
        ) < 1.0e-12

        # The default block size has to work without being told anything.
        defaulted = update(
            bpc; maxiter = 4, tolerance = nothing, message_update_alg = Algorithm("blocked")
        )
        @test maximum(
            TNQS.message_diff(TNQS.message(defaulted, e), TNQS.message(plain, e))
                for e in TNQS.edges(blocked)
        ) < 1.0e-12
    end
end
end
