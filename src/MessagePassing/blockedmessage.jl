using ITensors: Algorithm, array, dim, hasqns
using ITensors.NDTensors: NDTensors

#
# A memory-bounded message update for a double-layer (norm) network, registered as the "blocked"
# message-update algorithm:
#
#     update(bpc; maxiter, tolerance, message_update_alg = Algorithm("blocked"))
#
# It specialises `TensorNetworkState` norm networks with dense storage -- any vertex degree, any
# stored index order, any mix of bond dimensions -- and falls back to "contract" for everything else,
# so it is safe to enable globally.
#
# The stock "contract" path holds the ket, the bra and both factor-sized intermediates at once,
# roughly 6 × one factor, where a factor is the S·χ^d elements of one vertex tensor. This holds
#
#     the network's own tensor + two block buffers + the output
#         = (1 + 2b/χ_e) × one factor
#
# for two reasons. First, the bra is never materialised: `bp_factors` builds it as `dag(prime(T))`
# with the site indices replaced back to unprimed, which is exactly `conj(T)` *in the ket's own
# stored index order*, so `adjoint` supplies it to the closing gemm as a BLAS/cuBLAS 'C' flag.
# Second, the contraction is blocked over the outgoing leg, so no factor-sized intermediate is ever
# formed -- only a block's worth, and only ever two at a time, whatever the degree. At χ=1024 in
# ComplexF32 that is 19 GiB rather than 100 GiB.
#
# One case costs a further factor: an outgoing leg that sits neither first nor last in the stored
# order cannot be matricized for the closing gemm, so the ket is permuted into a copy first. A
# degree-3 vertex always has one such edge among its three, and the scratch is one buffer sized for
# the worst edge, so in practice the buffer is (2 + 2b/χ_e) factors -- the block bound is what is
# tight, not the total. Dropping the copy needs the close to accumulate over several gemms instead of
# one; `blocked_message!` is where that would go.
#
# `b` trades peak against very little: the closing gemm's arithmetic intensity is b flops/byte, so
# b = 64 is already compute-bound in fp32 and raising it only grows the peak. It is chosen from
# `max_scratch` (a fraction of one factor) unless given explicitly, and is clamped to χ_e.
#
# Runs on CPU or GPU unchanged -- only mul!, permutedims!, reshape and view, all on flat contiguous
# buffers, with no scalar indexing.
#

# --------------------------------------------------------------------------------------------------
# Scheduling
# --------------------------------------------------------------------------------------------------
#
# The contraction to perform, for the message on edge `e = v -> w` with the ket `A = array(tn[v])`
# in whatever index order the ITensor happens to have:
#
#     out[l_e', l_e] = Σ conj(A)[s, {l_i'}, l_e'] Π_i m_i[l_i, l_i'] A[s, {l_i}, l_e]
#
# A "layout" is a tuple of mode labels, where label `q` names the mode sitting at position `q` of
# `A`'s stored order and `sliced` is the label of the leg the block loop cuts. A buffer holding
# layout `(l_1, …, l_n)` stores mode `l_1` fastest, and always holds A's modes with the sliced one
# narrowed to the block width -- so a buffer is exactly `|A|·b/χ_e` elements, whatever the degree.
#
# Three ops act on such a buffer:
#
#     front gemm   transpose(m) * reshape(X, χ, :)     needs layout[1] to be m's leg
#     back gemm    reshape(X, :, χ) * m                needs layout[end] to be m's leg
#     permutedims! into the other buffer               any target layout
#
# A gemm writes the primed leg back into the slot it consumed, so it leaves the *layout* unchanged.
# That is the whole trick: absorbing a message costs one gemm and no data movement whenever its leg
# happens to sit at either end, and the closing gemm is a single gemm against the network's own
# array as soon as the layout is A's stored order with the sliced leg moved to one end -- which is
# what makes the `adjoint` bra valid, since positions then correspond.
#
# So the only thing to decide is where to put `permutedims!` calls, and each one should do *all* the
# reordering the remaining gemms need rather than one permute per gemm. Greedily:
#
#   1. absorb every message whose leg already sits at an end -- free, and it cannot hurt, since a
#      gemm leaves the layout alone;
#   2. spend one permutation bringing the next one or two legs to the two ends;
#   3. when one message is left, permute to the layout the close wants instead, so that the last
#      absorption lands on it -- possible exactly when that leg is at an end of the uncut order,
#      which is why step 2 saves such legs for last.
#
# For the degree-3 vertex this is one permutation per block, against three in the unscheduled version.

# `kind` is `:gemm`, `:permute` or `:close`.
#   :gemm    -- absorb message `msg` from `side` (`:front`/`:back`); `layout` is unchanged.
#   :permute -- `permutedims!` with `perm` into `layout`.
#   :close   -- contract against `conj(A)`; `side` is the close form, see `blocked_message!`.
struct MessageOp{N}
    kind::Symbol
    msg::Int
    side::Symbol
    perm::NTuple{N, Int}
    layout::NTuple{N, Int}
end

# The uncut modes in A's stored order, and the two layouts the closing gemm accepts: that order with
# the block index at either end. `:kb` is `(k…, b)` and `:bk` is `(b, k…)`; both close in a single
# gemm with two BLAS flags, so neither is preferred -- and having both is what lets a last absorption
# land on one of them instead of needing a permutation of its own.
function _close_layouts(n::Int, sliced::Int)
    korder = Tuple(q for q in 1:n if q != sliced)
    return korder, (korder..., sliced), (sliced, korder...)
end

# A layout with `f` leading and `l` trailing (0 for "anywhere"), everything else in stored order.
function _ends_layout(n::Int, f::Int, l::Int)
    mid = Tuple(q for q in 1:n if q != f && q != l)
    return ((f == 0 ? () : (f,))..., mid..., (l == 0 ? () : (l,))...)
end

# Absorbing a message whose leg is already at an end is free and never harmful: a gemm does not touch
# the layout, so doing it now can only loosen the constraints on what follows. One pass over the two
# ends is enough, for the same reason.
function _free_gemms!(ops::Vector{MessageOp{N}}, layout::NTuple{N, Int}, remaining, msgof) where {N}
    for (pos, side) in ((1, :front), (N, :back))
        i = msgof[layout[pos]]
        if i != 0 && remaining[i]
            push!(ops, MessageOp(:gemm, i, side, layout, layout))
            remaining[i] = false
        end
    end
    return ops
end

# Where to permute next, given the legs still to absorb.
#
# One leg left at an end of the uncut order is the good case: the layout the close wants already has
# that leg at an end, so the final gemm needs no permutation of its own. Otherwise bring the next one
# or two legs to the ends, preferring to leave an end-of-uncut leg for last so that fusion applies.
#
# With nothing left to absorb this is only reached when the block still has to be gathered (a
# strided source cannot be read as a matrix), so it gathers into a close-ready layout -- into the
# current one if that is already close-ready, which makes the gather a plain copy.
function _permute_target(n::Int, korder, kb, bk, layout, left::Vector{Int})
    if isempty(left)
        return (layout == kb || layout == bk) ? layout : kb
    elseif length(left) == 1
        only(left) == korder[end] && return bk
        only(left) == korder[1] && return kb
        return _ends_layout(n, only(left), 0)
    end
    order = sort(left; by = q -> q == korder[1] || q == korder[end])
    return _ends_layout(n, order[1], order[2])
end

"""
Schedule the ops for a ket with `n` modes whose cut leg sits at position `sliced` and whose message
legs sit at positions `msglegs`. Pure index bookkeeping -- no tensors and no sizes are involved, and
it is cheap enough to run per edge.
"""
function _message_schedule(n::Int, sliced::Int, msglegs::Vector{Int})
    korder, kb, bk = _close_layouts(n, sliced)
    msgof = zeros(Int, n)
    for (i, q) in enumerate(msglegs)
        msgof[q] = i
    end
    layout = ntuple(identity, n)
    ops = MessageOp{n}[]
    remaining = trues(length(msglegs))
    # The block is a contiguous run of the ket's storage only when the cut leg is trailing. Otherwise
    # it is strided, so nothing can be read from it as a matrix and the schedule has to open with the
    # gather that fixes that.
    insrc, srcusable = true, sliced == n

    while true
        canread = !insrc || srcusable
        canread && _free_gemms!(ops, layout, remaining, msgof)
        left = [msglegs[i] for i in eachindex(remaining) if remaining[i]]
        if isempty(left) && canread && (layout == kb || layout == bk)
            push!(ops, MessageOp(:close, 0, layout == kb ? :kb : :bk, layout, layout))
            return ops
        end
        target = _permute_target(n, korder, kb, bk, layout, left)
        perm = ntuple(k -> something(findfirst(==(target[k]), layout)), n)
        push!(ops, MessageOp(:permute, 0, :none, perm, target))
        layout, insrc = target, false
    end
    return
end

# --------------------------------------------------------------------------------------------------
# Scratch
# --------------------------------------------------------------------------------------------------

# One flat buffer, carved into the optional aligned copy, two block buffers and the output. A gemm
# cannot alias its input, so two block buffers is the floor -- and it is also the ceiling, because
# every op consumes one buffer and produces the other, so they simply swap.
#
# `BeliefPropagationCacheMPI` carries it in a `scratch` field, so it is grown once and then reused
# across every edge and every sweep -- the hot path allocates nothing. Any other cache type has no
# such field and falls back to allocating per call: the *peak* is unchanged (same buffer, same size)
# but it churns the allocator, so the reusing path is what the MPI runs should use.
function message_scratch_length(nket::Int, nblock::Int, chie::Int, needs_align::Bool)
    return (needs_align ? nket : 0) + 2 * nblock + chie^2
end

message_scratch(::AbstractBeliefPropagationCache) = Base.RefValue{Any}(Bool[])

# Grow the buffer to fit. The type check also catches a change of element type or device,
# in which case the old buffer is unusable and is replaced.
#
# The old buffer is dropped *before* the replacement is allocated. Otherwise both are live across
# the `similar`, which at S=4, χ=1024 is 32 GiB held while 36 GiB is requested -- and since the
# buffer is regrown every time a bond dimension climbs, that doubling happens repeatedly through a
# circuit. Releasing first lets the allocator hand back the same block.
function scratch_buffer!(ref::Base.RefValue{Any}, proto::AbstractVector, n::Int)
    s = ref[]
    if !(s isa typeof(proto)) || length(s) < n
        ref[] = Bool[]
        s = similar(proto, n)
        ref[] = s
    end
    return s
end

# Called once a BP solve is done. The scratch is only needed between the first and last message
# update of a sweep sequence; holding it afterwards means a factor-sized buffer squatting while
# gate application allocates its own.
release_message_scratch!(bpc::AbstractBeliefPropagationCache) = bpc

# --------------------------------------------------------------------------------------------------
# Kernel
# --------------------------------------------------------------------------------------------------

# The dimensions a buffer holding `layout` has, with the cut mode narrowed to the current block.
function _layoutdims(layout::NTuple{N, Int}, dims::NTuple{N, Int}, sliced::Int, nb::Int) where {N}
    return ntuple(k -> layout[k] == sliced ? nb : dims[layout[k]], Val(N))
end

# The block of the ket as a strided view. Only reached when the cut leg is leading, where the block
# is *not* a contiguous run of storage and therefore has to be gathered rather than reshaped.
function _slicedview(ket, dims::NTuple{N, Int}, sliced::Int, cols) where {N}
    idx = ntuple(k -> k == sliced ? cols : Colon(), Val(N))
    return view(reshape(ket, dims), idx...)
end

"""
    blocked_message!(outmat, buf1, buf2, ket, dims, sliced, mats, ops, b)

Run the schedule `ops` over blocks of width `b` of mode `sliced`.

`ket` is the flat storage of the ket tensor (or of its aligned copy) and `dims` its dimensions in
stored order; `mats` holds the incoming messages as χ×χ matrices oriented `(l, l')`; `outmat` is
χ_e×χ_e. `buf1`/`buf2` are block buffers of `prod(dims) ÷ dims[sliced] * b` elements each.

Nothing is allocated and nothing outside the three buffers is written.
"""
function blocked_message!(
        outmat, buf1, buf2, ket, dims::NTuple{N, Int}, sliced::Int, mats, ops, b::Int
    ) where {N}
    chie = dims[sliced]
    # Elements of the ket per unit of the cut leg; also the row count of the closing gemm.
    slab = div(prod(dims), chie)
    srcusable = sliced == N
    # Unless the cut leg is trailing, the block is strided in the ket's storage and cannot be
    # reshaped into a gemm operand, so the schedule has to open with the gather. Checked rather than
    # assumed: the failure mode is a wrong message, not an error.
    srcusable || first(ops).kind === :permute ||
        throw(ArgumentError("schedule must gather first when the cut leg is not trailing"))

    for lo in 1:b:chie
        nb = min(b, chie - lo + 1)
        cols = lo:(lo + nb - 1)
        nwritten = 0
        curlayout = ntuple(identity, Val(N))
        # A contiguous run of the ket when the cut leg is trailing. Otherwise this is a placeholder
        # of the right type that the first op -- a gather, which `_message_schedule` guarantees --
        # never reads.
        off = srcusable ? (lo - 1) * slab : 0
        cur = view(ket, (off + 1):(off + slab * nb))

        for op in ops
            if op.kind === :permute
                nwritten += 1
                dst = isodd(nwritten) ? buf1 : buf2
                ddims = _layoutdims(op.layout, dims, sliced, nb)
                dest = reshape(view(dst, 1:prod(ddims)), ddims)
                if nwritten == 1 && !srcusable
                    permutedims!(dest, _slicedview(ket, dims, sliced, cols), op.perm)
                else
                    sdims = _layoutdims(curlayout, dims, sliced, nb)
                    permutedims!(dest, reshape(cur, sdims), op.perm)
                end
                cur = view(dst, 1:prod(ddims))
                curlayout = op.layout
            elseif op.kind === :gemm
                nwritten += 1
                dst = isodd(nwritten) ? buf1 : buf2
                ddims = _layoutdims(curlayout, dims, sliced, nb)
                ntot = prod(ddims)
                m = mats[op.msg]
                if op.side === :front
                    chi = ddims[1]
                    rest = div(ntot, chi)
                    mul!(
                        reshape(view(dst, 1:ntot), chi, rest),
                        transpose(m), reshape(cur, chi, rest)
                    )
                else
                    chi = ddims[N]
                    rest = div(ntot, chi)
                    mul!(
                        reshape(view(dst, 1:ntot), rest, chi),
                        reshape(cur, rest, chi), m
                    )
                end
                cur = view(dst, 1:ntot)
            else
                # The close. `conj(A)` is the ket's own storage read with an `adjoint` flag, so the
                # bra is never materialised; which of the four forms applies is fixed by where the
                # cut leg sits in the ket (which end can be matricized) and by the block's layout.
                if srcusable
                    ket_mat = reshape(ket, slab, chie)
                    out = view(outmat, :, cols)                        # (l_e', l_e)
                    if op.side === :kb
                        mul!(out, adjoint(ket_mat), reshape(cur, slab, nb))
                    else
                        mul!(out, adjoint(ket_mat), transpose(reshape(cur, nb, slab)))
                    end
                else
                    ket_mat = reshape(ket, chie, slab)
                    out = view(outmat, cols, :)                        # (l_e, l_e')
                    if op.side === :kb
                        mul!(out, transpose(reshape(cur, slab, nb)), adjoint(ket_mat))
                    else
                        mul!(out, reshape(cur, nb, slab), adjoint(ket_mat))
                    end
                end
            end
        end
    end
    return outmat
end

# --------------------------------------------------------------------------------------------------
# Algorithm
# --------------------------------------------------------------------------------------------------

# The block buffers are the whole overhead above the network's own tensor: 2·|A|·b/χ_e elements. So
# `b` has to scale with χ_e to hold a memory bound -- a constant 64 is 12.5% of a factor at χ=1024
# but 200% of one at χ=64. `max_scratch` is that fraction directly, so the overhead is the same
# everywhere and the block size follows.
default_max_scratch(::Algorithm"blocked") = 1 / 8
blocked_blocksize(max_scratch::Real, chie::Integer) = clamp(floor(Int, max_scratch * chie / 2), 1, chie)
default_normalize(::Algorithm"blocked") = true

function set_default_kwargs(alg::Algorithm"blocked", bp_cache::AbstractBeliefPropagationCache)
    normalize = get(alg.kwargs, :normalize, default_normalize(alg))
    max_scratch = get(alg.kwargs, :max_scratch, default_max_scratch(alg))
    # `nothing` defers to `max_scratch`, which needs χ_e -- only known per edge in `updated_message`.
    b = get(alg.kwargs, :b, nothing)
    return Algorithm("blocked"; normalize, b, max_scratch)
end

_is_dense(t::ITensor) = ITensors.storage(t) isa NDTensors.Dense && !hasqns(t)

# The order to permute into when the cut leg is neither leading nor trailing. Unlike the stored order
# this one is ours to choose, and the choice is forced by what the schedule can then do: the cut leg
# goes last, since the close has to be able to matricize it and every block is then a contiguous run
# of the copy, which leaves the two slots a gemm can consume for free -- the front, and the position
# just before the cut leg. A message leg in each is what gets a degree-3 vertex to one block
# permutation: the first absorbs for free, and the second ends up at the end of the uncut order, so it
# lands straight on a layout the close accepts.
function _align_perm(n::Int, sliced::Int, msglegs::Vector{Int})
    f = isempty(msglegs) ? 0 : first(msglegs)
    l = length(msglegs) >= 2 ? last(msglegs) : 0
    mid = [q for q in 1:n if q != sliced && q != f && q != l]
    return Int[(f == 0 ? () : (f,))..., mid..., (l == 0 ? () : (l,))..., sliced]
end

function updated_message(
        alg::Algorithm"blocked", bp_cache::AbstractBeliefPropagationCache, edge::AbstractEdge
    )
    fallback() = updated_message(
        set_default_kwargs(Algorithm("contract"; normalize = alg.kwargs.normalize), bp_cache),
        bp_cache, edge
    )

    tn = network(bp_cache)
    # This is what makes the bra `conj` of the ket in the ket's own stored order, which the whole
    # kernel rests on: `bp_factors` is `[T, dag(prime(T))]` with the site legs replaced back to
    # unprimed, and neither `prime` nor `dag` reorders indices. Anything else -- a single-layer
    # network, a form with an operator factor -- is a different contraction.
    tn isa TensorNetworkState || return fallback()

    les = virtualinds(tn, edge)                   # an edge may carry several virtual inds
    length(les) == 1 || return fallback()
    le = only(les)

    v = src(edge)
    T = tn[v]
    # The flat matrix views assume a dense buffer whose length is the product of the dimensions, and
    # `dag` being plain conjugation. Neither holds for block-sparse or QN tensors.
    _is_dense(T) || return fallback()

    ms = incoming_messages(bp_cache, v; ignore_edges = (reverse(edge),))
    all(m -> ndims(m) == 2 && _is_dense(m), ms) || return fallback()

    is = collect(inds(T))
    n = length(is)
    sliced = findfirst(==(le), is)
    isnothing(sliced) && return fallback()

    # Each message must be a χ×χ matrix on one of the ket's legs and its prime -- that is the only
    # shape for which absorbing it leaves the layout, and hence the bra correspondence, intact.
    msglegs = Int[]
    mlegs = Index[]
    for m in ms
        mis = inds(m)
        c = commoninds(m, T)
        length(c) == 1 || return fallback()
        l = only(c)
        other = mis[1] == l ? mis[2] : mis[1]
        (other == prime(dag(l)) && dim(other) == dim(l)) || return fallback()
        q = findfirst(==(l), is)
        (isnothing(q) || q == sliced || q in msglegs) && return fallback()
        push!(msglegs, q)
        push!(mlegs, l)
    end

    chie = dim(le)
    b = alg.kwargs.b
    b = clamp(isnothing(b) ? blocked_blocksize(alg.kwargs.max_scratch, chie) : b, 1, chie)

    A = array(T)                                  # dims follow inds(T); a view when dense
    # Only a leading or trailing cut leg can be matricized for the closing gemm. Anything in between
    # needs one factor-sized permuted copy, which then plays the part of the ket throughout.
    needs_align = !(sliced == 1 || sliced == n)
    if needs_align
        alignperm = _align_perm(n, sliced, msglegs)
        dims = Tuple(dim(is[alignperm[k]]) for k in 1:n)
        msglegs = Int[something(findfirst(==(q), alignperm)) for q in msglegs]
        sliced = n
    else
        alignperm = nothing
        dims = Tuple(dim(i) for i in is)
    end
    ops = _message_schedule(n, sliced, msglegs)

    nket = prod(dims)
    nblock = div(nket, chie) * b
    s = scratch_buffer!(
        message_scratch(bp_cache), vec(A),
        message_scratch_length(nket, nblock, chie, needs_align)
    )
    o = 0
    # Unwrapped where it can be: fewer array wrappers to see through is fewer chances of a device
    # array falling off its fast path and into a scalar-indexing fallback.
    if needs_align
        ket = view(s, 1:nket)
        permutedims!(reshape(ket, dims), A, alignperm)
        o = nket
    else
        ket = vec(A)
    end
    buf1 = view(s, (o + 1):(o + nblock))
    buf2 = view(s, (o + nblock + 1):(o + 2 * nblock))
    outmat = reshape(view(s, (o + 2 * nblock + 1):(o + 2 * nblock + chie^2)), chie, chie)

    # Orient each message as (l, l') so `transpose` inside the kernel gives (l', l).
    mats = [array(ms[i], mlegs[i], prime(dag(mlegs[i]))) for i in eachindex(ms)]

    out = blocked_message!(outmat, buf1, buf2, ket, dims, sliced, mats, ops, b)

    # Label rather than transpose: the close writes columns of `out` when the cut leg is trailing and
    # rows when it is leading, so the two orientations differ only in which index goes first. Copy so
    # the message does not alias the scratch that the next edge overwrites.
    m = sliced == n ? itensor(copy(out), prime(dag(le)), le) : itensor(copy(out), le, prime(dag(le)))
    if alg.kwargs.normalize
        message_norm = sum(m)
        if !iszero(message_norm)
            m = m / message_norm
        end
    end
    # No contraction sequence is used, so `seq_changed = false` leaves the sequence cache
    # untouched.
    return m, (v => edge, nothing, false)
end
