# Spinless-fermion building blocks over a fermion-parity (fℤ₂) graded space.
#
# The operators come from TensorKitTensors.FermionOperators (sign conventions
# maintained upstream); this file only wraps them into the `ITensorMap` world and
# supplies the local Hilbert space and Fock states.
#
# A wrapped operator on sites `(s₁,…,sₙ)` is a raw `Vⁿ ← Vⁿ` TensorKit map with
# legs assigned so the ITensor operator layout `(s₁',…,sₙ', s₁,…,sₙ)` holds:
# codomain (outputs) get `prime(sₖ)` (space `V`), domain (inputs) get `dag(sₖ)`
# (space `dual(V)`, same `(id,plev)=(sₖ.id,0)` so it contracts with a ket on `sₖ`).

using TensorKitTensors.FermionOperators: FermionOperators, fermion_space, f_num, f_hop

export fermion_space, fermion_siteind, number_op, hopping_gate, parity_space, fermion_site_tensor

"""
    fermion_siteind(; plev=0) -> Index

A fresh site `Index` over the spinless-fermion local space
`Vect[fℤ₂](0 => 1, 1 => 1)` (even = empty `|0⟩`, odd = occupied `|1⟩`).
"""
fermion_siteind(; plev::Integer = 0) = Index(fermion_space(); plev)

"""
    parity_space(p::Integer) -> GradedSpace

The 1-dimensional fℤ₂ space in parity sector `p` (mod 2): `Vect[fℤ₂](0 => 1)` (even)
or `Vect[fℤ₂](1 => 1)` (odd). Used for bond-dimension-1 fermionic virtual bonds
carrying a definite parity (the Jordan–Wigner string).
"""
parity_space(p::Integer) = Vect[fℤ₂](mod(p, 2) => 1)

"""
    fermion_site_tensor(elt, cod_inds, cod_parities, dom_ind) -> ITensorMap

A bond-dimension-1 fermionic site tensor with the single nonzero block set to `1`.
`cod_inds` are the codomain leg `Index`es (typically the physical leg first, then the
child/loop virtual bonds) and `cod_parities[k]` is the fℤ₂ sector (`0`/`1`) selected on
`cod_inds[k]`. `dom_ind` is the single domain (to-root) leg `Index`, or `nothing` for the
tree root. The domain sector is fixed by parity conservation (`xor` of `cod_parities`).
"""
function fermion_site_tensor(elt::Type{<:Number}, cod_inds, cod_parities, dom_ind)
    cod_inds = Tuple(cod_inds)
    cod = reduce(⊗, map(space, cod_inds))
    if dom_ind === nothing
        sum(cod_parities) % 2 == 0 ||
            throw(ArgumentError("root site tensor must have even total parity"))
        data = zeros(elt, cod)
        legs = cod_inds
    else
        data = zeros(elt, cod ← dual(space(dom_ind)))
        legs = (cod_inds..., dom_ind)
    end
    want = collect(Int.(mod.(cod_parities, 2)))
    found = false
    for (f1, f2) in fusiontrees(data)
        if [s == fℤ₂(1) ? 1 : 0 for s in f1.uncoupled] == want
            data[f1, f2] .= one(elt)
            found = true
        end
    end
    found || throw(ArgumentError("no fusion tree matches codomain parities $want"))
    return ITensorMap(data, legs)
end

# Fermionic operator lookup for graded (fℤ₂) site legs. Routed here from the generic
# `op` so `expect`/`norm_factors` can request ("N", v) on a fermion site. The dense
# ITensors spin bridge (opcatalogue.jl) is left untouched for `CartesianSpace` legs.
function op(name::AbstractString, i::Index{<:GradedSpace}; kwargs...)
    lname = lowercase(name)
    (lname == "n" || name == "N") && return number_op(i)
    (lname == "i" || lname == "id") && return _op_itensormap(id(fermion_space()), (i,))
    throw(ArgumentError("no fermionic operator '$name' for graded site legs"))
end

# Wrap a raw `Vⁿ ← Vⁿ` operator TensorMap onto ITensorKit site legs `sites`.
# Codomain legs (effective space `V`) ← `prime(sₖ)`; domain legs (effective space
# `dual(V)`) ← `dag(sₖ)`.
function _op_itensormap(data::AbstractTensorMap, sites)
    sites = Tuple(sites)
    n = length(sites)
    numout(data) == n && numin(data) == n ||
        throw(ArgumentError("expected an n-in/n-out operator for $n sites, got $(numout(data))/$(numin(data))"))
    legs = (map(prime, sites)..., map(dag, sites)...)
    return ITensorMap(data, legs)
end

"""
    number_op([elt=ComplexF64,] s::Index) -> ITensorMap

The fermion number operator `n = c†c` on site `s`, as an operator `ITensorMap`
with legs `(s', s)`. Sourced from `TensorKitTensors.FermionOperators.f_num`.
"""
number_op(elt::Type{<:Number}, s::Index) = _op_itensormap(f_num(elt), (s,))
number_op(s::Index) = number_op(ComplexF64, s)

"""
    hopping_gate([elt=ComplexF64,] s1::Index, s2::Index, θ::Number) -> ITensorMap

The two-site hopping gate `exp(-iθ (c†₁c₂ + c†₂c₁))` on sites `(s1, s2)`, as an
operator `ITensorMap` with legs `(s1', s2', s1, s2)`. The hopping term is
`f_hop = f⁺f⁻ - f⁻f⁺` from `TensorKitTensors.FermionOperators`; the fermionic
signs live in the graded braiding.
"""
function hopping_gate(elt::Type{<:Number}, s1::Index, s2::Index, θ::Number)
    h = f_hop(complex(elt))
    gate = exp(-im * θ * h)
    return _op_itensormap(gate, (s1, s2))
end
hopping_gate(s1::Index, s2::Index, θ::Number) = hopping_gate(ComplexF64, s1, s2, θ)
