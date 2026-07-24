# Factorizations of `ITensorMap`.
#
# ITensor-style verbs (`svd`, `qr`, `lq`, `eigen`, `factorize`, `left_null`, `right_null`)
# that take *which legs go left* (the codomain) and split the tensor across a fresh internal
# bond. TensorKit 0.17's factorizations are positional (no leg-partition argument), so each
# wrapper:
#
#   1. permutes the data so the chosen left legs form the codomain, the rest the domain;
#   2. calls the underlying TensorKit factorization (`svd_trunc`, `left_orth`, …);
#   3. re-wraps each factor, minting a fresh bond `Index` from the factor's bond *space* and
#      placing its `dag` on the neighbouring factor.
#
# Because `dag(i) == i` (the match key `(id, plev)` ignores duality) while the spaces are
# mutually dual, the bond legs of adjacent factors contract — so `contract`ing the factors
# back together reconstructs the original tensor. Truncation is exposed verbatim through
# TensorKit's composable `trunc=` schemes (`truncrank`, `trunctol`, `truncerror`, …).

# --- leg partitioning -------------------------------------------------------
function _findleg(A::ITensorMap, i::Index)
    p = findfirst(==(i), A.inds)
    p === nothing && throw(ArgumentError("factorization: index $i is not a leg of the tensor"))
    return p
end

# Permute `A.data` so `left_inds` form the codomain and the remaining legs (in their current
# order) the domain. Returns the permuted data and the ordered left / right index tuples.
# Accept a single `Index` or any collection of indices as a leg specification.
_astuple(i::Index) = (i,)
_astuple(is) = Tuple(is)

function _partition(A::ITensorMap, left_inds)
    # Keep only the requested left indices that are actually legs of `A` (ITensor's
    # `factorize`/`svd` ignore extra indices, e.g. a bond already contracted away).
    lefts = Tuple(i for i in _astuple(left_inds) if hasind(A, i))
    rights = Tuple(i for i in A.inds if !hasind(lefts, i))
    pL = map(i -> _findleg(A, i), lefts)
    pR = map(i -> _findleg(A, i), rights)
    return permute(A.data, (pL, pR)), lefts, rights
end

# Explicit left/right partition (for `eigen`, which needs a square codomain/domain split).
function _partition(A::ITensorMap, left_inds, right_inds)
    lefts, rights = _astuple(left_inds), _astuple(right_inds)
    length(lefts) + length(rights) == numind(A) || throw(
        ArgumentError("left_inds and right_inds must together cover all $(numind(A)) legs")
    )
    pL = map(i -> _findleg(A, i), lefts)
    pR = map(i -> _findleg(A, i), rights)
    return permute(A.data, (pL, pR)), lefts, rights
end

"""
    permute(t::ITensorMap, cod_inds, dom_inds) -> ITensorMap

Repartition `t` into a `TensorMap` with codomain `cod_inds` and domain `dom_inds` (each a
single `Index` or a collection). Together they must cover every leg of `t`. The underlying
data is permuted accordingly (inserting the fermionic braiding signs), so a subsequent
`tr`/`svd`/adjoint sees the intended codomain/domain split. Relabels nothing — the returned
legs are exactly `(cod_inds..., dom_inds...)`.
"""
function TensorKit.permute(t::ITensorMap, cod_inds, dom_inds)
    cods, doms = _astuple(cod_inds), _astuple(dom_inds)
    length(cods) + length(doms) == numind(t) || throw(
        ArgumentError("permute: codomain+domain legs ($(length(cods))+$(length(doms))) must cover all $(numind(t)) legs")
    )
    pC = map(i -> _findleg(t, i), cods)
    pD = map(i -> _findleg(t, i), doms)
    data = permute(t.data, (pC, pD))
    return unsafe_itensormap(data, (cods..., doms...))
end

"""
    twist(t::ITensorMap, is) -> ITensorMap

Apply the fermionic `twist` to the leg(s) matching `is` (a single `Index` or a collection),
returning a relabelled-identical `ITensorMap` whose data carries the per-sector parity sign on
those legs. On a self-dual (dense) leg `twist` is the identity. Used to bake the physical-ket-leg
twist into a reduced density matrix so a subsequent `tr` needs no twist of its own.
"""
function TensorKit.twist(t::ITensorMap, is)
    data = t.data
    for i in _astuple(is)
        data = twist(data, _findleg(t, i))
    end
    return unsafe_itensormap(data, t.inds)
end

# Wrap a two-factor split `X * Y` (X: cod ← bond, Y: bond ← dom) sharing one fresh bond.
function _wrap2(X, Y, lefts, rights)
    b = Index(space(X, numind(X)))               # X's bond leg (its last, in the domain)
    return unsafe_itensormap(X, (lefts..., b)), unsafe_itensormap(Y, (dag(b), rights...))
end

# --- SVD --------------------------------------------------------------------
"""
    svd(A::ITensorMap, left_inds; trunc=nothing) -> U, S, V

Singular value decomposition splitting the legs `left_inds` (codomain) from the rest, with
`A ≈ U * S * V`. `S` is a positive diagonal map carrying two fresh bond legs; `U`/`V` are
isometric. Pass a TensorKit truncation scheme via `trunc` (e.g. `truncrank(64)`,
`trunctol(1e-12)`, composed with `&`).
"""
function LinearAlgebra.svd(A::ITensorMap, left_inds; trunc = nothing)
    data, lefts, rights = _partition(A, left_inds)
    U, S, Vᴴ = svd_trunc(data; trunc)
    bu = Index(space(U, numind(U)))              # U–S bond (U's domain leg)
    bv = Index(space(S, numind(S)))              # S–V bond (S's domain leg)
    return unsafe_itensormap(U, (lefts..., bu)),
        unsafe_itensormap(S, (dag(bu), bv)),
        unsafe_itensormap(Vᴴ, (dag(bv), rights...))
end

# --- ITensor-style truncated SVD factorization ------------------------------
export factorize_svd, TruncationSpec

"""
    TruncationSpec

Result metadata from [`factorize_svd`](@ref). Carries `truncerr`, the relative
discarded weight `‖discarded singular values‖² / ‖A‖²` (matching ITensor's
`spec.truncerr`).
"""
struct TruncationSpec
    truncerr::Float64
end

# Map ITensor truncation kwargs to a TensorKit truncation scheme. `maxdim` caps the
# rank; `cutoff` is ITensor's relative discarded squared weight, i.e. a relative
# 2-norm error of `sqrt(cutoff)`.
function _svd_trunc_scheme(maxdim, cutoff)
    schemes = Any[]
    isnothing(maxdim) || push!(schemes, truncrank(maxdim))
    isnothing(cutoff) || push!(schemes, truncerror(; rtol = sqrt(cutoff), p = 2))
    isempty(schemes) && return notrunc()
    return reduce(&, schemes)
end

"""
    factorize_svd(A::ITensorMap, left_inds; ortho="none", maxdim=nothing, cutoff=nothing,
                  singular_values!=nothing) -> R1, R2, spec

Truncated SVD factorization splitting the `left_inds` (codomain) from the rest, with
`R1 * R2 ≈ A`. `ortho` controls where the singular values go: `"none"` (balanced,
`R1 = U√S`, `R2 = √S V`), `"left"` (`R1 = U`, `R2 = S V`), or `"right"`. If a
`singular_values!` `Ref` is given it is filled with the diagonal singular-value
tensor `S`. `maxdim`/`cutoff` truncate (see [`TruncationSpec`](@ref) for the error).
"""
function factorize_svd(
        A::ITensorMap, left_inds; ortho::AbstractString = "none",
        maxdim = nothing, cutoff = nothing, singular_values! = nothing, kwargs...
    )
    data, lefts, rights = _partition(A, left_inds)
    U, S, Vᴴ, ϵ = svd_trunc(data; trunc = _svd_trunc_scheme(maxdim, cutoff))
    nrm2 = norm(data)^2
    truncerr = iszero(nrm2) ? 0.0 : ϵ^2 / nrm2

    bu = Index(space(U, numind(U)))
    bv = Index(space(S, numind(S)))
    Uitm = unsafe_itensormap(U, (lefts..., bu))
    Sitm = unsafe_itensormap(S, (dag(bu), bv))
    Vitm = unsafe_itensormap(Vᴴ, (dag(bv), rights...))
    isnothing(singular_values!) || (singular_values![] = Sitm)

    if ortho == "none"
        sqrtS = map_diag(sqrt, Sitm)                  # legs (dag(bu), bv)
        R1 = Uitm * sqrtS                             # (lefts..., bv)
        R2 = sqrtS * Vitm                             # (dag(bu), rights...)
        R2 = replaceind(R2, dag(bu), dag(bv))         # share the single bond bv
        return R1, R2, TruncationSpec(truncerr)
    elseif ortho == "left"
        return Uitm, Sitm * Vitm, TruncationSpec(truncerr)
    elseif ortho == "right"
        return Uitm * Sitm, Vitm, TruncationSpec(truncerr)
    else
        throw(ArgumentError("factorize_svd: `ortho` must be \"none\"/\"left\"/\"right\", got $(repr(ortho))"))
    end
end

# --- QR / LQ ----------------------------------------------------------------
"""
    qr(A::ITensorMap, left_inds) -> Q, R

QR split with `A == Q * R`, `Q` isometric over the `left_inds` codomain.
"""
function LinearAlgebra.qr(A::ITensorMap, left_inds)
    data, lefts, rights = _partition(A, left_inds)
    Q, R = left_orth(data; alg = :qr)
    return _wrap2(Q, R, lefts, rights)
end

"""
    lq(A::ITensorMap, left_inds) -> L, Q

LQ split with `A == L * Q`, `Q` isometric over the right (domain) legs.
"""
function LinearAlgebra.lq(A::ITensorMap, left_inds)
    data, lefts, rights = _partition(A, left_inds)
    L, Q = right_orth(data; alg = :lq)
    return _wrap2(L, Q, lefts, rights)
end

# --- generic orthogonal factorization ---------------------------------------
"""
    factorize(A::ITensorMap, left_inds; ortho=:left, trunc=nothing) -> X, Y

Split `A == X * Y` with one orthogonal factor: `ortho=:left` makes `X` isometric (QR, or SVD
when `trunc` is given), `ortho=:right` makes `Y` isometric (LQ / SVD).
"""
function LinearAlgebra.factorize(A::ITensorMap, left_inds; ortho = :left, trunc = nothing)
    data, lefts, rights = _partition(A, left_inds)
    ortho = Symbol(ortho)
    # `alg=nothing` (the default) picks a QR/LQ split when `trunc` is absent and an SVD-based
    # split when a truncation scheme is supplied.
    if ortho === :left
        X, Y = left_orth(data; trunc)
    elseif ortho === :right
        X, Y = right_orth(data; trunc)
    else
        throw(ArgumentError("factorize: `ortho` must be :left or :right, got $(repr(ortho))"))
    end
    return _wrap2(X, Y, lefts, rights)
end

# --- eigendecomposition -----------------------------------------------------
"""
    eigen(A::ITensorMap, left_inds, right_inds; ishermitian=false, trunc=nothing) -> D, U

Eigendecomposition of the square map with codomain `left_inds` and domain `right_inds`,
satisfying `A * U ≈ U * D`. `D` is diagonal (two fresh bond legs); `U` carries the left legs
and the shared bond. Set `ishermitian=true` for the Hermitian path (real `D`, unitary `U`).
"""
function LinearAlgebra.eigen(
        A::ITensorMap, left_inds, right_inds; ishermitian::Bool = false, trunc = nothing
    )
    data, lefts, rights = _partition(A, left_inds, right_inds)
    # MatrixAlgebraKit's Hermitian eigendecomposition rejects matrices that are only
    # approximately Hermitian (numerical noise), so project onto the Hermitian part.
    ishermitian && (data = (data + data') / 2)
    D, U = if isnothing(trunc)
        ishermitian ? eigh_full(data) : eig_full(data)
    else
        d, u, = ishermitian ? eigh_trunc(data; trunc) : eig_trunc(data; trunc)
        d, u
    end
    bu = Index(space(U, numind(U)))              # U–D bond (U's domain leg)
    bv = Index(space(D, numind(D)))              # D's outer bond
    return unsafe_itensormap(D, (dag(bu), bv)), unsafe_itensormap(U, (lefts..., bu))
end

# --- nullspaces -------------------------------------------------------------
"""
    left_null(A::ITensorMap, left_inds; alg=:qr) -> N

Isometry `N` spanning the left nullspace: `contract(dag(N), A) ≈ 0`, with `N`'s codomain the
`left_inds`.
"""
function MatrixAlgebraKit.left_null(A::ITensorMap, left_inds; alg = :qr)
    data, lefts, _ = _partition(A, left_inds)
    N = left_null(data; alg)
    return unsafe_itensormap(N, (lefts..., Index(space(N, numind(N)))))
end

"""
    right_null(A::ITensorMap, left_inds; alg=:lq) -> Nᴴ

Isometry `Nᴴ` spanning the right nullspace: `contract(A, dag(Nᴴ)) ≈ 0`, with `Nᴴ`'s domain
the legs not in `left_inds`.
"""
function MatrixAlgebraKit.right_null(A::ITensorMap, left_inds; alg = :lq)
    data, _, rights = _partition(A, left_inds)
    Nᴴ = right_null(data; alg)
    return unsafe_itensormap(Nᴴ, (Index(space(Nᴴ, 1)), rights...))
end
