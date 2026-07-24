# Scalar extraction and dense-array conversion for `ITensorMap`.

export scalar, array

"""
    scalar(t::ITensorMap)
    t[]

The single number held by a 0-leg tensor (e.g. the result of a full contraction).
Errors if `t` has any legs.
"""
function TensorKit.scalar(t::ITensorMap)
    numind(t) == 0 || throw(ArgumentError("scalar: tensor has $(numind(t)) legs, expected 0"))
    return scalar(t.data)
end
Base.getindex(t::ITensorMap) = scalar(t)

"""
    array(t::ITensorMap) -> Array

A dense Julia `Array` of `t`'s entries with axes in `inds(t)` order. Inverse of
[`itensor`](@ref): `itensor(array(t), inds(t)) == t`.
"""
function array(t::ITensorMap)
    numind(t) == 0 && return fill(scalar(t))
    A = convert(Array, t.data)
    return reshape(A, map(dim, inds(t)))
end

"""
    diag(t::ITensorMap)

The diagonal populations of a density-matrix-like tensor: the ket legs (prime level 0) are paired
with their bra partners (higher prime level, matched by `id`) and repartitioned into an endomorphism
`W ← W` (ket = codomain, matching bra = domain), exactly as in [`tr`](@ref). The diagonal is read
per-sector via TensorKit's `diagview` — no dense `convert(Array, …)`, so the grading is kept intact
(no categorical-property warning) — and the sector blocks are concatenated in canonical order (the
same order the dense basis uses). For a fermionic reduced density matrix already carrying the
physical-ket twist (see `order_rdm`) these are the physical populations `p_k = ⟨k|ρ|k⟩`, with
`sum(diag(t)) == tr(t)`. Falls back to the dense diagonal when the legs do not split evenly into
ket/bra pairs.
"""
function LinearAlgebra.diag(t::ITensorMap)
    ket = filter(i -> plev(i) == 0, t.inds)
    bra = filter(i -> plev(i) != 0, t.inds)
    (length(ket) == length(bra) && !isempty(ket)) || return LinearAlgebra.diag(array(t))
    braord = map(k -> bra[findfirst(b -> _id(b) == _id(k), bra)], ket)
    m = permute(t, ket, braord).data                   # endomorphism W ← W
    return mapreduce(collect, vcat, values(diagview(m)))
end

"""
    tr(t::ITensorMap)

The (regular) trace of a density-matrix-like tensor: the ket legs (prime level 0) are traced
against their bra partners (higher prime level, matched by `id`). The legs are repartitioned
into an endomorphism `W ← W` (ket = codomain, matching bra = domain) and traced with TensorKit's
`tr`. This is a *plain* categorical trace with no twist of its own — a fermionic reduced density
matrix must already carry the physical-ket-leg twist (see `order_rdm`). Falls back to the dense
matrix trace when the legs do not split evenly into ket/bra pairs.
"""
function LinearAlgebra.tr(t::ITensorMap)
    ket = filter(i -> plev(i) == 0, t.inds)
    bra = filter(i -> plev(i) != 0, t.inds)
    (length(ket) == length(bra) && !isempty(ket)) || return LinearAlgebra.tr(array(t))
    # pair each bra leg to its ket leg by identity, then repartition to `W ← W` and trace
    braord = map(k -> bra[findfirst(b -> _id(b) == _id(k), bra)], ket)
    return LinearAlgebra.tr(permute(t, ket, braord).data)
end
