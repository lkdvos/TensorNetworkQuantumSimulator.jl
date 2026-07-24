function symmetric_gauge!(bp_cache::BeliefPropagationCache; regularization = 10 * eps(real(scalartype(bp_cache))), kwargs...)
    tn = network(bp_cache)
    !(tn isa TensorNetworkState) && error("Can only transform TensorNetworkStates to the symmetric gauge")
    for e in edges(tn)
        # Graded (fermionic) bonds are not self-dual, so the bond bookkeeping differs; the two
        # paths coincide on a dense self-dual backend (`dag`/duality are no-ops there).
        if hasqns(tn[src(e)])
            _symmetric_gauge_graded_edge!(bp_cache, e; kwargs...)
        else
            _symmetric_gauge_dense_edge!(bp_cache, e; regularization, kwargs...)
        end
    end

    return bp_cache
end

# Dense (self-dual `CartesianSpace`) symmetric gauge for a single edge. Unchanged from the
# original implementation: every bond relabel assumes `dag == identity` and `sim` preserves
# the (dimension-only) space.
function _symmetric_gauge_dense_edge!(bp_cache::BeliefPropagationCache, e; regularization, kwargs...)
    tn = network(bp_cache)
    vsrc, vdst = src(e), dst(e)
    ψvsrc, ψvdst = tn[vsrc], tn[vdst]

    edge_ind = commoninds(ψvsrc, ψvdst)
    edge_ind_sim = sim(edge_ind)

    me, mer = message(bp_cache, e), message(bp_cache, reverse(e))
    # Hermitian eigendecomposition of each message (legs split unprimed/primed);
    # `eigendecomp` returns (U, D, Udag) with `U * D * Udag ≈ message`.
    X_U, X_D, X_Udag = eigendecomp(me, filter(i -> plev(i) == 0, inds(me)), filter(i -> plev(i) != 0, inds(me)); ishermitian = true)
    Y_U, Y_D, Y_Udag = eigendecomp(mer, filter(i -> plev(i) == 0, inds(mer)), filter(i -> plev(i) != 0, inds(mer)); ishermitian = true)
    X_D, Y_D = map_diag(x -> x + regularization, X_D),
        map_diag(x -> x + regularization, Y_D)

    rootX = X_U * map_diag(x -> sqrt(x), X_D) * X_Udag
    rootY = Y_U * map_diag(x -> sqrt(x), Y_D) * Y_Udag
    inv_rootX = X_U * map_diag(x -> inv(sqrt(x)), X_D) * X_Udag
    inv_rootY = Y_U * map_diag(x -> inv(sqrt(x)), Y_D) * Y_Udag

    ψvsrc, ψvdst = noprime(ψvsrc * inv_rootX), noprime(ψvdst * inv_rootY)

    Ce = rootX
    Ce = Ce * replaceinds(rootY, edge_ind, edge_ind_sim)

    U, S, V = svd(Ce, edge_ind; kwargs...)

    new_edge_ind = Index[Index(dim(only(commoninds(S, U))))]

    ψvsrc = replaceinds(ψvsrc * U, commoninds(S, U), new_edge_ind)
    ψvdst = replaceinds(ψvdst, edge_ind, edge_ind_sim)
    ψvdst = replaceinds(ψvdst * V, commoninds(V, S), new_edge_ind)


    S = replaceinds(
        S,
        [commoninds(S, U)..., commoninds(S, V)...] =>
            [new_edge_ind..., prime(new_edge_ind)...],
    )

    sqrtS = map_diag(sqrt, S)
    ψvsrc = noprime(ψvsrc * sqrtS)
    ψvdst = noprime(ψvdst * sqrtS)
    setindex_preserve!(bp_cache, ψvsrc, vsrc)
    setindex_preserve!(bp_cache, ψvdst, vdst)

    setmessage!(bp_cache, e, S)
    setmessage!(bp_cache, reverse(e), dag(S))
    return bp_cache
end

# Swap the two prime levels of a 2-leg (Hermitian) map: the plev-0 leg `i0` becomes `prime(i0)`
# and the plev-1 leg `i1` becomes `noprime(i1)`, keeping each leg's space (and the data). Routed
# through a fresh temporary so the two legs never transiently share a match key `(id, plev)` —
# `replaceind` matches on `(id, plev)` alone, so a direct swap would relabel the wrong leg.
function _swap_plevs(t::ITensor, i0::Index, i1::Index)
    tmp = sim(i1)
    t = replaceind(t, i1, tmp)
    t = replaceind(t, i0, prime(i0))
    t = replaceind(t, tmp, noprime(i1))
    return t
end

# Graded (fermionic, `Vect[fℤ₂]`) symmetric gauge for a single edge.
#
# The bonds are *not* self-dual: on a directed edge `src → dst` the ket bond leg `l` (space `W`)
# lives on `ψvsrc`, its dual `dag(l)` (space `dual W`) on `ψvdst`, and the BP messages carry
# `{l, prime(dag(l))}` (native duality on plev 0, dual on plev 1) — the same orientation as the
# BP simple-update messages. TensorKit contractions require the two paired legs to be mutually
# dual, so the whole procedure is expressed as *relabels* (no data conjugation, so no stray
# fermion signs): the env square roots are absorbed by swapping the two prime levels (so a `W`
# vertex leg meets the env's `dual W` leg), the bond matrix `Ce` is formed by contracting the
# plev-0 (ket) legs, and the balanced `factorize_svd` mints a fresh, self-consistent dual bond
# for the two isometry-·-√S factors.
function _symmetric_gauge_graded_edge!(bp_cache::BeliefPropagationCache, e; kwargs...)
    tn = network(bp_cache)
    vsrc, vdst = src(e), dst(e)
    ψvsrc, ψvdst = tn[vsrc], tn[vdst]

    l = only(commoninds(ψvsrc, ψvdst))       # ψvsrc bond leg, space W (plev 0)
    ls = sim(l)                               # fresh id, same space W

    me, mer = message(bp_cache, e), message(bp_cache, reverse(e))
    xl = only(filter(i -> plev(i) == 0, inds(me)))    # == l              (space W)
    xr = only(filter(i -> plev(i) != 0, inds(me)))    # == prime(dag(l))  (space dual W)
    yl = only(filter(i -> plev(i) == 0, inds(mer)))   # == dag(l)         (space dual W)
    yr = only(filter(i -> plev(i) != 0, inds(mer)))   # == prime(l)       (space W)

    # Hermitian but fermion-*indefinite* square roots (complex principal branch, see
    # `pseudo_sqrt_inv_sqrt`). Legs are inherited from the message: {xl, xr} / {yl, yr}.
    rootX, inv_rootX = pseudo_sqrt_inv_sqrt(me)
    rootY, inv_rootY = pseudo_sqrt_inv_sqrt(mer)

    # --- gauge the vertex tensors (strip the BP env sqrt) ---
    # Swap the env's two prime levels so its *dual*-space leg lands on plev 0 to contract the
    # vertex's native bond leg; the surviving (opposite-duality) leg becomes the bond after
    # `noprime`. ψvsrc keeps `l` (W); ψvdst keeps `dag(l)` (dual W).
    inv_rootX_c = _swap_plevs(inv_rootX, xl, xr)
    inv_rootY_c = _swap_plevs(inv_rootY, yl, yr)
    ψvsrc = noprime(ψvsrc * inv_rootX_c)
    ψvdst = noprime(ψvdst * inv_rootY_c)

    # --- bond matrix Ce = rootX · rootY, contracting the ket (plev-0) legs ---
    # `sim` the dst-side plev-1 leg so only the plev-0 legs pair (otherwise the plev-1 legs
    # would contract too, collapsing Ce to a scalar).
    rootY_c = replaceind(rootY, yr, prime(ls))
    Ce = rootX * rootY_c                       # legs {xr (dual W), prime(ls) (W)}

    # Balanced SVD: R1 = U√S (src side), R2 = √S·V (dst side), sharing a fresh dual bond.
    singular_values! = Ref{ITensor}()
    R1, R2, _ = factorize_svd(Ce, [xr]; ortho = "none", singular_values!, kwargs...)
    S = singular_values![]
    b = only(commoninds(R1, R2))               # new bond, R1 side (ψvsrc will carry it)

    # --- absorb the factors into the vertices ---
    # R1's src leg is `xr = prime(dag(l))` (plev 1); drop it to plev 0 `dag(l)` to meet ψvsrc's `l`.
    R1_c = replaceind(R1, xr, noprime(xr))
    ψvsrc = ψvsrc * R1_c                       # bond becomes `b`
    # R2's dst leg is `prime(ls)` (plev 1); drop to `ls`, and relabel ψvdst's `dag(l)` → `dag(ls)`.
    ψvdst = replaceind(ψvdst, dag(l), dag(ls))
    R2_c = replaceind(R2, prime(ls), ls)
    ψvdst = ψvdst * R2_c                       # bond becomes `dag(b)`

    setindex_preserve!(bp_cache, ψvsrc, vsrc)
    setindex_preserve!(bp_cache, ψvdst, vdst)

    # Singular-value message on the new bond in the default `{a, prime(dag(a))}` convention
    # (`a = b` is ψvsrc's new bond leg). `S = {dag(bu), b}` is a square diagonal map, so its
    # other leg has space `dual(space(b))` and relabels cleanly onto `prime(dag(b))`.
    b_other = only(filter(i -> i != b, inds(S)))
    S_msg = replaceind(S, b_other, prime(dag(b)))
    setmessage!(bp_cache, e, S_msg)
    setmessage!(bp_cache, reverse(e), dag(S_msg))
    return bp_cache
end

function symmetric_gauge(bp_cache::BeliefPropagationCache; kwargs...)
    bp_cache = copy(bp_cache)
    return symmetric_gauge!(bp_cache; kwargs...)
end

function symmetric_gauge(tns::TensorNetworkState; cache_update_kwargs = (; maxiter = 40), kwargs...)
    bp_cache = BeliefPropagationCache(tns)
    bp_cache = update(bp_cache; cache_update_kwargs...)
    bp_cache = symmetric_gauge(bp_cache; kwargs...)
    return network(bp_cache)
end

function symmetrize_and_normalize(bp_cache::BeliefPropagationCache; kwargs...)
    bp_cache = rescale(bp_cache)
    bp_cache = symmetric_gauge(bp_cache; kwargs...)
    return bp_cache
end

function symmetrize_and_bpnormalize(tns::TensorNetworkState; cache_update_kwargs = (; maxiter = 40), kwargs...)
    bp_cache = BeliefPropagationCache(tns)
    bp_cache = update(bp_cache; cache_update_kwargs...)
    bp_cache = symmetrize_and_normalize(bp_cache; kwargs...)
    return network(bp_cache)
end

gauge_and_scale(tns::TensorNetworkState; kwargs...) = symmetrize_and_bpnormalize(tns::TensorNetworkState; kwargs...)