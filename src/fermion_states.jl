# Fermionic (fℤ₂) product states as bond-dimension-1 tensor networks.
#
# An occupied site carries odd fermion parity, so a fermionic product state cannot
# be a plain tensor product of local kets: the parity has to be routed through the
# virtual bonds (the Jordan–Wigner string). We route it along a spanning tree,
# building leaves -> root and carrying the cumulative subtree parity toward the
# root; every tree bond's arrow points to the root. Non-tree ("loop-closing") edges
# carry trivial (even) charge at bond dimension 1, so no per-loop sign is needed.
# The root closes to the trivial sector, which requires an even total parity.

export fermion_tensornetworkstate

_occupation(x::Integer) = Int(x)
_occupation(x::Bool) = Int(x)
function _occupation(x::AbstractString)
    x in ("1", "↑", "occ", "occupied", "full") && return 1
    x in ("0", "↓", "empty", "vac", "vacuum") && return 0
    error("unrecognized fermion occupation string \"$x\" (use \"0\"/\"1\")")
end

"""
    fermion_tensornetworkstate([elt=ComplexF64,] f, g::AbstractGraph; root=first(vertices(g)))

Construct a spinless-fermion product `TensorNetworkState` on graph `g`. `f(v)` gives
the occupation of vertex `v` (`0`/`1`, `Bool`, or `"0"`/`"1"`). The state is built via
a spanning tree rooted at `root`, with fℤ₂-graded virtual bonds carrying the parity
string toward the root; the total occupation must be even (odd total is not
representable without an open charge leg).
"""
function fermion_tensornetworkstate(elt::Type{<:Number}, f, g::AbstractGraph; root = first(collect(vertices(g))))
    vs = collect(vertices(g))
    V = vertextype(g)
    occ = Dictionary{V, Int}(vs, [_occupation(f(v)) for v in vs])
    iseven(sum(occ)) || error("fermion_tensornetworkstate requires an even total occupation (got $(sum(occ)))")

    # --- spanning tree (BFS from root) ---
    parent = Dict{V, Union{V, Nothing}}(root => nothing)
    order = V[root]
    queue = V[root]
    treeedges = Set{Tuple{V, V}}()   # (child, parent)
    while !isempty(queue)
        u = popfirst!(queue)
        for w in neighbors(g, u)
            if !haskey(parent, w)
                parent[w] = u
                push!(order, w); push!(queue, w)
                push!(treeedges, (w, u))
            end
        end
    end
    children = Dict{V, Vector{V}}(v => V[] for v in vs)
    for (c, p) in treeedges
        push!(children[p], c)
    end

    # --- cumulative subtree parity (postorder = reverse BFS order) ---
    subpar = Dict{V, Int}()
    for v in reverse(order)
        subpar[v] = mod(occ[v] + sum(Int[subpar[c] for c in children[v]]; init = 0), 2)
    end

    # --- shared bond indices ---
    # tree edge (child c -> parent p): index minted on the PARENT as a codomain leg
    # (space = parity_space(subpar[c])); the child uses `dag(.)` as its to-root domain leg.
    treeind = Dict((c, p) => Index(parity_space(subpar[c])) for (c, p) in treeedges)
    sites = Dictionary{V, Vector{<:Index}}(vs, [Index[fermion_siteind()] for _ in vs])
    # non-tree edges: trivial (even) dim-1 bond.
    nontree = Tuple{V, V}[]
    for e in edges(g)
        a, b = src(e), dst(e)
        ((a, b) in treeedges || (b, a) in treeedges) && continue
        push!(nontree, (a, b))
    end
    ntind = Dict(e => Index(parity_space(0)) for e in nontree)

    # --- per-site tensors ---
    tensors = Dictionary{V, ITensor}()
    for v in vs
        cod_inds = Index[only(sites[v])]
        cod_par = Int[occ[v]]
        for c in children[v]
            push!(cod_inds, treeind[(c, v)]); push!(cod_par, subpar[c])
        end
        for e in nontree
            if e[1] == v
                push!(cod_inds, ntind[e]); push!(cod_par, 0)
            elseif e[2] == v
                push!(cod_inds, dag(ntind[e])); push!(cod_par, 0)
            end
        end
        dom_ind = parent[v] === nothing ? nothing : dag(treeind[(v, parent[v])])
        set!(tensors, v, fermion_site_tensor(elt, cod_inds, cod_par, dom_ind))
    end

    return TensorNetworkState(TensorNetwork(tensors, g), sites)
end

fermion_tensornetworkstate(f, g::AbstractGraph; kwargs...) = fermion_tensornetworkstate(ComplexF64, f, g; kwargs...)
