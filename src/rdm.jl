# Assemble an RDM into its canonical form: the ket legs (prime level 0) of the requested `verts`
# form the codomain in `verts` order, their bra partners (`dag(prime(·))`) the domain in the same
# order, giving a `W ← W` endomorphism. The fermionic `twist` is then baked onto the physical ket
# legs (cf. Canopy's `twist!(ρ, physical_ket_legs)`), so afterwards a *plain* trace `tr(ρ)` — and
# `tr(ρ ∘ O)` — are the correct physical (twisted) quantities with no further sign handling. On a
# self-dual dense leg the twist is the identity, so this leaves the dense RDM unchanged.
function order_rdm(ρ::ITensor, state, verts)
    ket = reduce(vcat, [collect(siteinds(state, v)) for v in verts])
    return order_rdm(ρ, ket)
end

# Self-inferring variant: the ket legs are `ρ`'s prime-level-0 legs (first-appearance order) and the
# bra legs their `dag(prime(·))` partners. Same canonical twisted layout as the `verts` form; used
# where the requested sites are implicit (e.g. the single-site RDMs built during sampling).
order_rdm(ρ::ITensor) = order_rdm(ρ, collect(inds(ρ; plev = 0)))

function order_rdm(ρ::ITensor, ket::AbstractVector{<:Index})
    isempty(ket) && return ρ
    bra = dag.(prime.(ket))
    return twist(permute(ρ, ket, bra), ket)
end

# Trace-normalize an RDM by its (plain) trace. `ρ` is expected to be an `order_rdm` output, i.e.
# already the twisted density matrix, so no twist is applied here.
normalize_rdm(ρ::ITensor) = ρ / tr(ρ)

"""
    reduced_density_matrix(ψ, verts; alg = nothing, kwargs...)

Compute the reduced density matrix on the vertices `verts` of the tensor network state `ψ`.

# Arguments
- `ψ::Union{TensorNetworkState, BeliefPropagationCache, BoundaryMPSCache}`: The tensor network state or its associated cache.
- `verts`: The vertices over which to compute the reduced density matrix. Can be a single vertex or a collection of vertices.

# Keyword Arguments
- `alg::Union{String, Nothing}`: The contraction algorithm to use. If not provided, defaults based on the type of `ψ`. Supported algorithms are `"exact"`, `"bp"`, and `"boundarymps"`.
- `normalize::Bool = true`: Whether to normalize the reduced density matrix so that its trace is 1.
- `kwargs...`: Additional keyword arguments specific to the chosen algorithm.

# Returns
- An `ITensor` representing the reduced density matrix on the specified vertices.
"""
function reduced_density_matrix(ψ::Union{TensorNetworkState, BeliefPropagationCache, BoundaryMPSCache}, verts; alg::Union{String, Nothing} = default_alg(ψ), kwargs...)
    algorithm_check(ψ, "rdm", alg)
    verts = collect_vertices(verts, graph(ψ))
    return reduced_density_matrix(Algorithm(alg), ψ, verts; kwargs...)
end

function reduced_density_matrix(
        alg::Algorithm"exact",
        ψ::TensorNetworkState,
        verts::Vector;
        contraction_sequence_kwargs = (; alg = "omeinsum", optimizer = GreedyMethod()),
        normalize = true
    )
    disable_warn_order()
    op_string_f = v -> v ∈ verts ? "ρ" : "I"
    ρ_tensors = norm_factors(ψ, collect(vertices(ψ)); op_strings = op_string_f)
    seq = contraction_sequence(ρ_tensors; contraction_sequence_kwargs...)
    ρ = contract(ρ_tensors; sequence = seq)
    ρ = order_rdm(ρ, ψ, verts)
    if normalize
        ρ = normalize_rdm(ρ)
    end
    return ρ
end


function reduced_density_matrix(
        alg::Algorithm"bp",
        cache::BeliefPropagationCache,
        vs::Vector;
        normalize = true
    )
    steiner_vs = length(vs) == 1 ? vs : collect(vertices(steiner_tree(network(cache), vs)))
    incoming_ms = incoming_messages(cache, steiner_vs)

    op_string_f = v -> v ∈ vs ? "ρ" : "I"

    #TODO: If there are a lot of tensors here, (more than 100 say), we need to think about defining a custom sequence as optimal may be too slow
    ρ_tensors = norm_factors(network(cache), steiner_vs; op_strings = op_string_f)
    append!(ρ_tensors, incoming_ms)
    seq = contraction_sequence(ρ_tensors; alg = "optimal")
    ρ = contract(ρ_tensors; sequence = seq)
    ρ = order_rdm(ρ, network(cache), vs)

    if normalize
        ρ = normalize_rdm(ρ)
    end
    return ρ
end

function reduced_density_matrix(
        alg::Algorithm"boundarymps",
        cache::BoundaryMPSCache,
        vs::Vector;
        normalize = true,
        bmps_messages_up_to_date = false,
    )

    op_string_f = v -> v ∈ vs ? "ρ" : "I"
    ρ, _ = path_contract(cache, vs, op_string_f; bmps_messages_up_to_date)
    ρ = order_rdm(ρ, network(cache), vs)

    if normalize
        ρ = normalize_rdm(ρ)
    end
    return ρ
end

function reduced_density_matrix(
        alg::Algorithm"bp",
        ψ::TensorNetworkState,
        verts::Vector;
        cache_update_kwargs = default_bp_update_kwargs(ψ),
        kwargs...,
    )
    ψ_bpc = BeliefPropagationCache(ψ)
    ψ_bpc = update(ψ_bpc; cache_update_kwargs...)

    return reduced_density_matrix(alg, ψ_bpc, verts; kwargs...)
end

function reduced_density_matrix(
        alg::Algorithm"boundarymps",
        ψ::TensorNetworkState,
        verts::Vector;
        cache_update_kwargs = default_bmps_update_kwargs(ψ),
        mps_bond_dimension::Integer,
        partition_by::String = boundarymps_partitioning(verts),
        kwargs...,
    )
    ψ_bpc = BoundaryMPSCache(ψ, mps_bond_dimension; partition_by)
    ψ_bpc = update(ψ_bpc; cache_update_kwargs...)

    return reduced_density_matrix(alg, ψ_bpc, verts; kwargs...)
end

function boundarymps_partitioning(vs::Vector)
    allequal(first.(vs)) && return "row"
    allequal(last.(vs)) && return "col"
    error("Vertices must be aligned in either the same column or the same row to do BoundaryMPS.")
end

const rdm = reduced_density_matrix
