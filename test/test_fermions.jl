@eval module $(gensym())
# Spinless free fermions on a graph: quench from a Fock product state under
# nearest-neighbour hopping, checked against the exact free-fermion (Gaussian)
# result. Because the model is quadratic, the many-body evolution under the exact
# same sequence of two-site hopping gates is reproduced level-by-level by the
# single-particle propagator acting on the correlation matrix C -> u C u'. That
# makes the comparison exact (no Trotter error), so any mismatch is a fermionic
# sign error.
#
# The fermionic operators come from TensorKitTensors.FermionOperators (fℤ₂ graded,
# signs maintained upstream); TNQS wraps them in `ITensorKit.fermions`. States are
# built here directly as graded `ITensorMap`s (an occupied site carries odd
# fermion parity, which needs a charge leg — see `fermion_state`).

using LinearAlgebra: diagm, diag, exp
using TensorKit
using TensorKitTensors.FermionOperators: fermion_space
using TensorNetworkQuantumSimulator
const TNQS = TensorNetworkQuantumSimulator
using TensorNetworkQuantumSimulator:
    named_hexagonal_lattice_graph, named_grid, vertices, edges, src, dst, neighbors,
    fermion_tensornetworkstate, norm_sqr, expect, symmetric_gauge,
    BeliefPropagationCache, update, apply_gates, network
using TensorNetworkQuantumSimulator.ITensorKit:
    Index, ITensor, ITensorMap, fermion_siteind, number_op, hopping_gate, contract, dag, noprime, scalar
using Test: @testset, @test

const V = fermion_space()
const Vodd = Vect[fℤ₂](1 => 1)

# A definite-occupation Fock state |n₁,…,n_N⟩ as a graded `ITensorMap` over `sites`.
# Codomain = the N site legs; if the total parity is odd, a dim-1 odd "charge" leg
# is appended (an odd state is not representable without one, since `V ← I` holds
# only the even sector).
function fermion_state(occs::Vector{Int}, sites)
    N = length(occs)
    odd = isodd(sum(occs))
    cod = reduce(⊗, ntuple(_ -> V, N))
    data = odd ? zeros(ComplexF64, cod ← Vodd) : zeros(ComplexF64, cod)
    target = Tuple(fℤ₂.(occs))
    found = false
    for (f1, f2) in fusiontrees(data)
        if f1.uncoupled == target
            data[f1, f2] .= 1
            found = true
        end
    end
    found || error("occupation $occs not found among fusion trees")
    legs = odd ? (sites..., Index(dual(Vodd))) : Tuple(sites)
    return ITensorMap(data, legs)
end

# ⟨n_v⟩ on site index `sites[v]`, as a ratio so the (fermionic) norm sign cancels.
function density(ket, sites, v)
    opψ = noprime(contract(number_op(sites[v]), ket))
    num = scalar(contract(dag(ket), opψ))
    den = scalar(contract(dag(ket), ket))
    return real(num / den)
end

norm_sq(ket) = real(scalar(contract(dag(ket), ket)))

# Single-particle reference: apply the same two-site hopping gate to the N×N
# correlation matrix, C -> u C u', with u = exp(-iθ K), K the a↔b hop.
function sp_gate!(C, a, b, θ)
    N = size(C, 1)
    K = zeros(ComplexF64, N, N)
    K[a, b] = 1
    K[b, a] = 1
    u = exp(-im * θ * K)
    C .= u * C * u'
    return C
end

# Run `nsteps` Trotter sweeps of `edgelist` (each edge a hopping gate of angle θ)
# on both the many-body graded state and the single-particle correlation matrix;
# return (many-body densities, reference densities, norm²).
function quench(vlist, edgelist, occs; nsteps, θ)
    N = length(vlist)
    idx = Dict(v => i for (i, v) in enumerate(vlist))
    sites = [fermion_siteind() for _ in 1:N]
    ket = fermion_state(occs, sites)
    C = ComplexF64.(diagm(occs))
    for _ in 1:nsteps, e in edgelist
        a, b = idx[e[1]], idx[e[2]]
        ket = noprime(contract(hopping_gate(sites[a], sites[b], θ), ket))
        sp_gate!(C, a, b, θ)
    end
    return [density(ket, sites, i) for i in 1:N], real.(diag(C)), norm_sq(ket)
end

# BFS 2-colouring -> occupations filling one sublattice (a charge-density-wave
# Fock state on a bipartite graph). `parity_target` forces an even total so the
# state needs no global charge leg.
function cdw_occupations(g; parity_target = 0)
    vs = collect(vertices(g))
    colour = Dict{eltype(vs), Int}()
    for s in vs
        haskey(colour, s) && continue
        colour[s] = 0
        queue = [s]
        while !isempty(queue)
            u = popfirst!(queue)
            for w in neighbors(g, u)
                if !haskey(colour, w)
                    colour[w] = 1 - colour[u]
                    push!(queue, w)
                end
            end
        end
    end
    occ = [colour[v] == 0 ? 1 : 0 for v in vs]
    if (sum(occ) % 2) != parity_target
        occ[findfirst(==(1), occ)] = 0   # drop one particle to hit target parity
    end
    return vs, occ
end

# Package-level quench: build the state with `fermion_tensornetworkstate`, evolve with
# `simple_update` (empty environments, no truncation -> exact), measure with alg="exact".
function pipeline_quench(g, occf, edgelist; nsteps, θ)
    vs = collect(vertices(g))
    vindex = Dict(v => i for (i, v) in enumerate(vs))
    ψ = fermion_tensornetworkstate(occf, g)
    sind(v) = only(TNQS.siteinds(ψ, v))
    C = ComplexF64.(diagm([occf(v) for v in vs]))
    for _ in 1:nsteps, e in edgelist
        u, v = e
        gate = hopping_gate(sind(u), sind(v), θ)
        upd, _, _ = TNQS.simple_update(gate, ITensor[ψ[u], ψ[v]];
            envs = ITensor[], normalize_tensors = false, cutoff = 0.0, maxdim = 4096)
        ψ[u] = upd[1]; ψ[v] = upd[2]
        sp_gate!(C, vindex[u], vindex[v], θ)
    end
    mb = [real(expect(ψ, ("N", v); alg = "exact")) for v in vs]
    return mb, real.(diag(C)), real(norm_sqr(ψ; alg = "exact"))
end

# BP quench: evolve with `apply_gates` through a `BeliefPropagationCache` (simple update using
# belief-propagation environments), then measure with alg="bp" and, on the evolved network,
# alg="exact". Because there is no truncation the evolved state stays exact, so alg="exact" must
# match the free-fermion reference; alg="bp" additionally tests the belief-propagation contraction.
# Returns (bp densities, exact densities, reference densities, bp norm²).
function bp_quench(g, occf, edgelist; nsteps, θ)
    vs = collect(vertices(g))
    vindex = Dict(v => i for (i, v) in enumerate(vs))
    ψ = fermion_tensornetworkstate(occf, g)
    sind(v) = only(TNQS.siteinds(ψ, v))
    C = ComplexF64.(diagm([occf(v) for v in vs]))
    gates = ITensor[]
    gate_vs = Vector{eltype(vs)}[]
    for _ in 1:nsteps, e in edgelist
        u, v = e
        push!(gates, hopping_gate(sind(u), sind(v), θ))
        push!(gate_vs, [u, v])
        sp_gate!(C, vindex[u], vindex[v], θ)
    end
    ψ_bpc = BeliefPropagationCache(ψ)
    ψ_bpc = update(ψ_bpc; maxiter = 1)
    ψ_bpc, _ = apply_gates(
        gates, ψ_bpc; gate_vertices = gate_vs,
        apply_kwargs = (; cutoff = 0.0, maxdim = 4096, normalize_tensors = false), update_cache = true,
    )
    ψev = network(ψ_bpc)
    mb_bp = [real(expect(ψ_bpc, ("N", v); alg = "bp")) for v in vs]
    mb_ex = [real(expect(ψev, ("N", v); alg = "exact")) for v in vs]
    return mb_bp, mb_ex, real.(diag(C)), real(norm_sqr(ψ_bpc; alg = "bp"))
end

@testset "Fermions (free-fermion quench)" begin

    @testset "(a) two sites, one particle" begin
        s1, s2 = fermion_siteind(), fermion_siteind()
        ψ0 = fermion_state([1, 0], (s1, s2))
        @test density(ψ0, (s1, s2), 1) ≈ 1 atol = 1e-12
        @test density(ψ0, (s1, s2), 2) ≈ 0 atol = 1e-12
        # one hopping gate is the exact evolution of a 2-site single particle:
        # n₁ = cos²θ, n₂ = sin²θ.
        for θ in (0.0, 0.3, 0.7, π / 4, 1.2, π / 2)
            ψ = noprime(contract(hopping_gate(s1, s2, θ), ψ0))
            @test density(ψ, (s1, s2), 1) ≈ cos(θ)^2 atol = 1e-10
            @test density(ψ, (s1, s2), 2) ≈ sin(θ)^2 atol = 1e-10
        end
    end

    @testset "(b) chain and ring vs single-particle propagator" begin
        # 4-site chain, 2 particles (even parity -> no charge leg, norm² = 1).
        mb, sp, nrm = quench(1:4, [(i, i + 1) for i in 1:3], [1, 0, 0, 1]; nsteps = 6, θ = 0.35)
        @test nrm ≈ 1 atol = 1e-10
        @test mb ≈ sp atol = 1e-10

        # 6-site ring: the closed loop is where fermionic loop-braiding signs bite.
        ring = vcat([(i, i + 1) for i in 1:5], [(6, 1)])
        mb, sp, nrm = quench(1:6, ring, [1, 0, 1, 0, 0, 0]; nsteps = 6, θ = 0.4)
        @test nrm ≈ 1 atol = 1e-10
        @test mb ≈ sp atol = 1e-10
    end

    @testset "(c) honeycomb patch, charge-density-wave quench" begin
        g = named_hexagonal_lattice_graph(1, 1)
        vs, occ = cdw_occupations(g; parity_target = 0)
        es = [(src(e), dst(e)) for e in edges(g)]
        mb, sp, nrm = quench(vs, es, occ; nsteps = 5, θ = 0.3)
        @test nrm ≈ 1 atol = 1e-10
        @test mb ≈ sp atol = 1e-10
    end

    @testset "(d) package pipeline: state + exact norm/expect + evolution" begin
        # honeycomb patch, charge-density-wave Fock state through the TNQS API
        g = named_hexagonal_lattice_graph(1, 1)
        vs, occ = cdw_occupations(g; parity_target = 0)
        occd = Dict(vs[i] => occ[i] for i in eachindex(vs))
        ψ0 = fermion_tensornetworkstate(v -> occd[v], g)
        @test real(norm_sqr(ψ0; alg = "exact")) ≈ 1 atol = 1e-10
        for v in vs
            @test real(expect(ψ0, ("N", v); alg = "exact")) ≈ occd[v] atol = 1e-10
        end
        es = [(src(e), dst(e)) for e in edges(g)]
        mb, sp, nrm = pipeline_quench(g, v -> occd[v], es; nsteps = 4, θ = 0.3)
        @test nrm ≈ 1 atol = 1e-10
        @test mb ≈ sp atol = 1e-10

        # 1D chain via named_grid, CDW filling
        gp = named_grid((4, 1))
        vsp = collect(vertices(gp))
        occp = Dict(vsp[i] => (isodd(i) ? 1 : 0) for i in eachindex(vsp))
        esp = [(src(e), dst(e)) for e in edges(gp)]
        mb, sp, nrm = pipeline_quench(gp, v -> occp[v], esp; nsteps = 5, θ = 0.4)
        @test nrm ≈ 1 atol = 1e-10
        @test mb ≈ sp atol = 1e-10
    end

    @testset "(e) belief propagation: simple update + BP environments" begin
        # 1D chain (tree): BP is exact, so alg="bp" == alg="exact" == reference to machine precision.
        gp = named_grid((4, 1))
        vsp = collect(vertices(gp))
        occp = Dict(vsp[i] => (isodd(i) ? 1 : 0) for i in eachindex(vsp))
        esp = [(src(e), dst(e)) for e in edges(gp)]
        mb_bp, mb_ex, sp, nrm_bp = bp_quench(gp, v -> occp[v], esp; nsteps = 5, θ = 0.4)
        @test nrm_bp ≈ 1 atol = 1e-8          # well-defined BP norm (+1, even parity)
        @test mb_ex ≈ sp atol = 1e-9          # BP-evolved state is exact (no truncation)
        @test mb_bp ≈ mb_ex atol = 1e-9       # BP is exact on a tree

        # Honeycomb patch (has loops): the untruncated state stays exact so alg="exact" matches the
        # free-fermion reference; BP itself is approximate on loops but stays well-defined and, at a
        # small Trotter angle, close to exact.
        g = named_hexagonal_lattice_graph(1, 1)
        vs, occ = cdw_occupations(g; parity_target = 0)
        occd = Dict(vs[i] => occ[i] for i in eachindex(vs))
        es = [(src(e), dst(e)) for e in edges(g)]
        mb_bp, mb_ex, sp, nrm_bp = bp_quench(g, v -> occd[v], es; nsteps = 4, θ = 0.1)
        @test nrm_bp ≈ 1 atol = 1e-6          # well-defined BP norm (+1), not the −1 fermionic sign
        @test mb_ex ≈ sp atol = 1e-9          # evolution is exact
        @test mb_bp ≈ mb_ex atol = 1e-2       # BP approximate on loops but close at small θ
    end

    @testset "(f) boundary-MPS contraction (graded product state)" begin
        # A fermionic charge-density-wave product state on a grid, contracted with the boundary-MPS
        # algorithm. For a product state the graded interpartition MPS is exact, so norm² = 1 and the
        # occupations are recovered exactly.
        g = named_grid((3, 3))
        vs, occ = cdw_occupations(g; parity_target = 0)
        occd = Dict(vs[i] => occ[i] for i in eachindex(vs))
        ψ0 = fermion_tensornetworkstate(v -> occd[v], g)
        @test real(norm_sqr(ψ0; alg = "boundarymps", mps_bond_dimension = 16)) ≈ 1 atol = 1e-8
        for v in vs
            nb = real(expect(ψ0, ("N", v); alg = "boundarymps", mps_bond_dimension = 16, gauge_state = false))
            @test nb ≈ occd[v] atol = 1e-8
        end
    end

    @testset "(g) boundary-MPS contraction (graded entangled/evolved state)" begin
        # One layer of hopping entangles the 3×3 charge-density-wave state (the ket virtual bonds
        # now carry both fℤ₂ sectors). The graded boundary MPS goes through the doubled SVD zip-up,
        # which mints correctly-sectored bonds, so alg="boundarymps" → alg="exact" as the bond
        # dimension grows (norm² stays +1). Because the model is quadratic the untruncated evolved
        # state is exact, so alg="exact" also matches the single-particle reference C.
        g = named_grid((3, 3))
        vs, occ = cdw_occupations(g; parity_target = 0)
        occd = Dict(vs[i] => occ[i] for i in eachindex(vs))
        vindex = Dict(v => i for (i, v) in enumerate(vs))
        ψ = fermion_tensornetworkstate(v -> occd[v], g)
        sind(v) = only(TNQS.siteinds(ψ, v))
        C = ComplexF64.(diagm([occd[v] for v in vs]))
        θ = 0.3
        for e in [(src(x), dst(x)) for x in edges(g)]
            u, w = e
            gate = hopping_gate(sind(u), sind(w), θ)
            upd, _, _ = TNQS.simple_update(gate, ITensor[ψ[u], ψ[w]];
                envs = ITensor[], normalize_tensors = false, cutoff = 0.0, maxdim = 4096)
            ψ[u] = upd[1]; ψ[w] = upd[2]
            sp_gate!(C, vindex[u], vindex[w], θ)
        end

        # untruncated evolution is exact ⇒ alg="exact" matches the free-fermion reference
        mb_ex = [real(expect(ψ, ("N", v); alg = "exact")) for v in vs]
        @test mb_ex ≈ real.(diag(C)) atol = 1e-9

        # The graded symmetric (Vidal) gauge preserves the physical state: gauged ⟨N⟩ and the
        # norm² sign are unchanged (a wrong duality/sign would break this).
        ψg = symmetric_gauge(ψ)
        @test [real(expect(ψg, ("N", v); alg = "exact")) for v in vs] ≈ mb_ex atol = 1e-9
        @test real(norm_sqr(ψg; alg = "exact")) ≈ 1 atol = 1e-9

        # boundary MPS: well-defined norm (+1), converges to exact and improves with bond dimension
        occ_bmps(χ) = [real(expect(ψ, ("N", v); alg = "boundarymps", mps_bond_dimension = χ, gauge_state = false)) for v in vs]
        err(χ) = maximum(abs.(occ_bmps(χ) .- mb_ex))
        @test real(norm_sqr(ψ; alg = "boundarymps", mps_bond_dimension = 16)) ≈ 1 atol = 1e-8
        @test err(16) < 1e-6            # converged to exact at χ = 16
        @test err(16) < err(4)          # larger bond dimension is strictly better

        # The default gauge_state=true path applies the graded symmetric gauge as an accuracy
        # preconditioner. It must converge to exact and, at a fixed moderate bond dimension, be at
        # least as accurate as the ungauged boundary MPS.
        occ_gauged(χ) = [real(expect(ψ, ("N", v); alg = "boundarymps", mps_bond_dimension = χ, gauge_state = true)) for v in vs]
        errg(χ) = maximum(abs.(occ_gauged(χ) .- mb_ex))
        @test errg(16) < 1e-6              # gauged converges to exact (also the gauge_state=true default)
        @test errg(8) <= err(8) + 1e-10    # gauged at least as accurate as ungauged at fixed χ
    end

end
end
