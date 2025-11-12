# We have the dimensionless 1D time-independent Schrodinger equation
# - d^2 psi / d(xi)^2 + V(xi) psi = Epsi
# with potential V(xi) = - 1 / cosh(xi)^2
#
# Overview:
# - We discretize the second derivative with a central finite difference on a uniform grid.
# - Dirichlet boundary conditions psi(+/- L/2) = 0 are enforced by using only the N interior points.
# - This yields a symmetric tridiagonal Hamiltonian H (discrete operator) that we diagonalize.
# - The "workspace" struct allows reusing buffers to minimize allocations.
#
# Notation:
# - L: total domain length; h: grid spacing; N approx L/h - 1 interior grid points.
# - q: coordinate scaling factor; potential is V(xi) = -sech(q*xi)^2.
# - p and q are related by q = 1/sqrt(p) (helpers provided).
#
# Performance notes:
# - @inbounds avoids bounds checks in tight loops (assumes indices are valid).
# - @simd allows the compiler to vectorize loops when safe.
# - Views (@view) take slices without allocating new arrays.

using LinearAlgebra

# -- Helper Functions --

"""
Return the coordinate scaling factor q = 1/sqrt(p) for a given p.
"""
function q_of_p(p::T) where T<:Number
    return 1 / sqrt(p)
end

"""
Inverse of `q_of_p`: return p = 1/q^2.
"""
function p_of_q(q::T) where T<:Number
    return 1 / (q^2)
end

"""
    xi_to_index(x, L, N; mode=:floor) -> idx::Int

Map a coordinate x in [-L/2, L/2] to the index of the nearest interior
grid point on a uniform grid with N interior points. The grid spacing is
dx = L/(N+1) and interior points are xi_j = -L/2 + j*dx for j=1..N.

Arguments
- x: coordinate (Real)
- L: domain length (Real)
- N: number of interior points (Int)

Keyword arguments
- mode: :floor (default) uses trunc((x+L/2)/dx)+1 which matches the
  indexing used elsewhere; :nearest uses round((x+L/2)/dx) to pick the
  closest grid point.

The returned index is clamped to 1..N.
"""
function xi_to_index(x::Real, L::Real, N::Int; mode::Symbol = :floor)
    dx = L / (N + 1)
    t = (x + 0.5 * L) / dx
    idx = mode == :nearest ? round(Int, t) : trunc(Int, t) + 1
    return clamp(idx, 1, N)
end


function prob_density(psi::AbstractVector{<:ComplexF64})
    return abs2.(psi)
end

"""
Compute the percent error between an analytical and a numerical value.
```jldoctest
julia> percent_error(1.00, 0.98)
2.0
```
"""
function percent_error(analytical::T, numerical::T) where T<:AbstractFloat
    return abs((analytical - numerical) / analytical) * 100
end

# -- Analytical Solutions --

"""
    analytical_energy_levels(p) -> levels::Vector{Float64}

Compute the bound-state energy levels for the Poschl-Teller potential
`V(xi) = -sech(q*xi)^2`.

"""
function analytical_energy_levels(p::T) where T<:Number
    n_max = floor(Int, (sqrt(1 + 4*p) - 1) / 2)
    levels = Vector{Float64}(undef, n_max + 1)
    sqrt_term = sqrt(1 + 4*p)
    @inbounds @simd for n in 0:n_max
        levels[n + 1] = -((sqrt_term - (2*n + 1))^2) / (4 * p)
    end
    return levels
end

# -- Numerical Methods --

"""
    cda_first(f2, f0, a)

Second-order central difference approximation for the first derivative:
    f'(x) approx (f(x+a) - f(x-a)) / (2a).

```jldoctest
julia> cda_first(2.0, 0.0, 1.0)
1.0
```    
"""
function cda_first(f2::T, f0::T, a::T) where T<:AbstractFloat
    return (f2 - f0) / (2 * a)
end

"""
    cda_second(f0, f1, f2, a)

Second-order central difference approximation for the second derivative:
    `f''(x) approx (f(x+a) - 2f(x) + f(x-a)) / a^2`.
```jldoctest
julia> cda_second(1.0, 2.0, 1.0, 1.0)
0.0
```
"""
function cda_second(f0::T, f1::T, f2::T, a::T) where T<:AbstractFloat
    return (f2 - 2*f1 + f0) / (a^2)
end

"""
    struct SolverWorkspace

Preallocated buffers to assemble the tridiagonal Hamiltonian without reallocations.
Fields:
- xi: grid coordinates (interior points only)
- diag: diagonal entries of H
- offdiag: off-diagonal entries of H
"""
struct SolverWorkspace
    xi::Vector{Float64}
    diag::Vector{Float64}
    offdiag::Vector{Float64}
end

"""
    SolverWorkspace(maxN)

Create a workspace capable of handling up to `maxN` interior grid points.
Throws if `maxN < 1`.
"""
function SolverWorkspace(maxN::Int)
    maxN >= 1 || throw(ArgumentError("maxN must be positive, got $(maxN)"))
    xi = Vector{Float64}(undef, maxN)
    diag = Vector{Float64}(undef, maxN)
    offdiag = Vector{Float64}(undef, max(maxN - 1, 0))
    return SolverWorkspace(xi, diag, offdiag)
end

"""
    assemble_hamiltonian!(L, h, q, ws) -> (N, H::SymTridiagonal)

Assemble the symmetric tridiagonal Hamiltonian H for the Poschl-Teller potential
on the N interior points of the uniform grid in (-L/2, L/2), with spacing h.

# Notes:
- Dirichlet BCs at the boundaries are implicit by excluding endpoints.
- Uses [`SolverWorkspace`](@ref) views to avoid allocations.
 - Returns N = number of interior points, and the SymTridiagonal H (or nothing if N <= 0).
"""
function assemble_hamiltonian!(L::Float64, h::Float64, q::Float64, ws::SolverWorkspace)
    N = round(Int, L / h) - 1              # number of interior points
    # Early-outs and validation.
    # - If N <= 0, the domain has no interior points at this resolution.
    # - If N exceeds workspace capacity, instruct the user to enlarge it.
    if N <= 0
        return 0, nothing
    elseif N > length(ws.xi)
        throw(ArgumentError("workspace too small for N = $(N); increase maxN"))
    end

    # Take non-allocating views into the workspace buffers for the active size N.
    xi = @view ws.xi[1:N]
    diag = @view ws.diag[1:N]
    offdiag = if N > 1
        @view ws.offdiag[1:(N - 1)]
    else
        @view ws.offdiag[1:0]  # empty view, no allocation when N == 1
    end

    half_L = 0.5 * L
    inv_h2 = 1.0 / (h * h)                 # reuse 1/h^2

    # Fill grid coordinates of interior points: -L/2 + h, ..., L/2 - h.
    @inbounds @simd for i in 1:N
        xi[i] = -half_L + h * i
    end
    # Discrete -d^2/dxi^2 contributes 2/h^2 to the diagonal and -1/h^2 to neighbors.
    @inbounds @simd for i in 1:N
        diag[i] = 2.0 * inv_h2
    end
    if N > 1
        @inbounds @simd for i in 1:(N - 1)
            offdiag[i] = -inv_h2
        end
    end
    # Add diagonal potential term V(xi) = -sech(q * xi)^2.
    @inbounds @simd for i in 1:N
        diag[i] += -1.0 / cosh(xi[i] * q)^2
    end

    # Return the assembled symmetric tridiagonal matrix.
    return N, SymTridiagonal(diag, offdiag)
end



"""
    compute_lowest_energies!(dest, L, h, q, ws) -> dest

Compute the smallest eigenvalues of the Hamiltonian and store them into `dest`.
- The number of values written is min(length(dest), N).
- Remaining entries (if any) are filled with NaN.
"""
function compute_lowest_energies!(dest::Vector{Float64}, L::Float64, h::Float64, q::Float64, ws::SolverWorkspace)
    # Assemble H on the current grid.
    N, H = assemble_hamiltonian!(L, h, q, ws)
    if N == 0
        fill!(dest, NaN)
        return dest
    end

    # Extract eigenvalues of the symmetric tridiagonal matrix efficiently.
    eigvals = LinearAlgebra.eigvals(H)
    # Copy as many as requested into `dest`; pad with NaN if fewer are available.
    limit = min(length(dest), length(eigvals))
    @inbounds @simd for i in 1:limit
        dest[i] = eigvals[i]
    end
    if limit < length(dest)
        @inbounds @simd for i in (limit + 1):length(dest)
            dest[i] = NaN
        end
    end
    return dest
end

# -- Solver Interface --

"""
    solve_schrodinger(L, h, q, ws) -> (eigenvalues, eigenvectors, xi)

Build the Hamiltonian on a uniform grid with spacing h over (-L/2, L/2),
apply Dirichlet BCs (interior points only), and compute its full eigendecomposition.

This overload reuses the provided `ws::SolverWorkspace` to avoid allocations
when scanning parameters or repeatedly solving.

Returns:
- eigenvalues::Vector{Float64}
- eigenvectors::Matrix{Float64} (columns are eigenvectors on interior grid)
- xi::Vector{Float64} interior grid coordinates (copied out from workspace)
"""
function solve_schrodinger(L::Float64, h::Float64, q::Float64, ws::SolverWorkspace)
    N, H = assemble_hamiltonian!(L, h, q, ws)
    if N == 0
        return [Inf], nothing, Float64[]
    end
    eig = eigen(H)
    xi_view = @view ws.xi[1:N]
    return eig.values, eig.vectors, copy(xi_view)
end

"""
    solve_schrodinger(L, h, q) -> (eigenvalues, eigenvectors, xi)

Convenience overload that allocates a temporary [`SolverWorkspace`](@ref) sized for this grid
and delegates to the workspace-based method.
"""
function solve_schrodinger(L::Float64, h::Float64, q::Float64)
    N = round(Int, L / h) - 1
    if N <= 0
        return [Inf], nothing, Float64[]
    end
    ws = SolverWorkspace(N)
    return solve_schrodinger(L, h, q, ws)
end

"""
    solve_schrodinger(N, L, q)

Convenience overload: compute h = L/(N+1) for N interior points, then call the main solver.
"""
function solve_schrodinger(N::Int, L::Float64, q::Float64)
    h = L / (N + 1)
    return solve_schrodinger(L, h, q)
end


"""
    evolve_coeffs!(c0, E, dt)
Evolves the coefficients c0 in the eigenbasis over time dt given eigenvalues E from
a *time-independent Hamiltonian H*.

# Example:
```jldoctest
julia> c = complex.([1.0, 0.0, 0.0])
E = [-1.0, 0.0, 1.0] # assumed to be eigenvalues from a time-independent Hamiltonian H #
dt = pi/2
julia> evolve_coeffs!(c, E, dt);
julia> c
3-element Vector{ComplexF64}:
 0.0 + 1.0im
 0.0 + 0.0im
 0.0 - 1.0im
```
"""
function evolve_coeffs!(c0::Vector{ComplexF64}, E::Vector{Float64}, dt::Float64)
    @inbounds @simd for i in 1:length(c0)
        c0[i] *= exp(-im * E[i] * dt)
    end
end