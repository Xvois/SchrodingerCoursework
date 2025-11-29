# Performance notes:
# - @inbounds avoids bounds checks in tight loops (assumes indices are valid).
# - @simd allows the compiler to vectorize loops when safe.
# - Views (@view) take slices without allocating new arrays.

using LinearAlgebra
using FFTW

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


"""
    struct SolverWorkspace

Preallocated buffers to assemble the tridiagonal time independent Hamiltonian without reallocations.
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
    assemble_static_hamiltonian!(L, h, V, ws) -> (N, H::SymTridiagonal)

Assemble the symmetric tridiagonal time independent Hamiltonian H for the potential V(xi)
on the N interior points of the uniform grid in (-L/2, L/2), with spacing h using CDA.

# Notes:
- Dirichlet BCs at the boundaries are implicit by excluding endpoints.
- Uses [`SolverWorkspace`](@ref) views to avoid allocations.
 - Returns N = number of interior points, and the SymTridiagonal H (or nothing if N <= 0).
"""
function assemble_static_hamiltonian!(L::Float64, h::Float64, V::Function, ws::SolverWorkspace)
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

    # Add diagonal potential term V(xi).
    @inbounds @simd for i in 1:N
        diag[i] += V(xi[i])
    end

    # Return the assembled symmetric tridiagonal matrix.
    return N, SymTridiagonal(diag, offdiag)
end

# -- Solver Interface --

"""
    solve_static_schrodinger(L, h, V, ws) -> (eigenvalues, eigenvectors, xi)

Build the Hamiltonian on a uniform grid with spacing h over (-L/2, L/2),
apply Dirichlet BCs (interior points only), and compute its full eigendecomposition.

This overload reuses the provided `ws::SolverWorkspace` to avoid allocations
when scanning parameters or repeatedly solving.

Returns:
- eigenvalues::Vector{Float64}
- eigenvectors::Matrix{Float64} (columns are eigenvectors on interior grid)
- xi::Vector{Float64} interior grid coordinates (copied out from workspace)
"""
function solve_static_schrodinger(L::Float64, h::Float64, V::Function, ws::SolverWorkspace)
    N, H = assemble_static_hamiltonian!(L, h, V, ws)
    if N == 0
        return [Inf], nothing, Float64[]
    end
    eig = eigen(H, -1, 0) # compute valid energies and eigenvectors only
    xi_view = @view ws.xi[1:N]
    return eig.values, eig.vectors, copy(xi_view)
end

"""
    solve_static_schrodinger(L, h, V) -> (eigenvalues, eigenvectors, xi)

Convenience overload that allocates a temporary [`SolverWorkspace`](@ref) sized for this grid
and delegates to the workspace-based method.
"""
function solve_static_schrodinger(L::Float64, h::Float64, V::Function)
    N = round(Int, L / h) - 1
    if N <= 0
        return [Inf], nothing, Float64[]
    end
    ws = SolverWorkspace(N)
    return solve_static_schrodinger(L, h, V, ws)
end

"""
    solve_static_schrodinger(N, L, V)

Convenience overload: compute h = L/(N+1) for N interior points, then call the main solver.
"""
function solve_static_schrodinger(N::Int, L::Float64, V::Function)
    h = L / (N + 1)
    return solve_static_schrodinger(L, h, V)
end

"""
    evolve_dynamic_coeffs!(psi, L, h, Vxt, dt; nsteps=1, ws, basis=nothing, store=true)

Crank-Nicolson time stepper for a state vector `psi` on the interior grid of length L with spacing h.
- Vxt(xi, t) should return the potential at spatial point xi and time t.
- dt: time-step size.
Keyword args:
- nsteps: number of CN steps to take (default 1).
- ws: SolverWorkspace for assembly (required).
- basis: optional matrix whose columns are basis functions (for projection); if provided,
  the function returns a matrix of coefficients with size (ncols, nsteps+1).
- store: if true and basis provided, store coefficients at t=0 and after each step.

Returns:
- psi (updated in-place)
- coeffs (if basis provided) else nothing
"""
function evolve_dynamic_coeffs!(psi::Vector{ComplexF64},
                                L::Float64, h::Float64,
                                Vxt::Function,
                                dt::Float64,
                                t_start::Float64=0.0; # Added t_start
                                nsteps::Int=1,
                                ws::SolverWorkspace,
                                basis::Union{Nothing,AbstractMatrix{<:Number}}=nothing,
                                store::Bool=true)

    N_expected = round(Int, L / h) - 1
    length(psi) == N_expected || throw(ArgumentError("psi length mismatch"))

    # Prepare storage
    coeffs = nothing
    if basis !== nothing && store
        # Pre-calculate initial state projection before evolution starts
        coeffs = Vector{ComplexF64}[]
        push!(coeffs, project_onto_basis_L2(psi, basis, h)) 
    end

    t = t_start
    
    for step in 1:nsteps
        t_mid = t + dt/2
        
        # Assemble H at the midpoint time
        V_here = x -> Vxt(x, t_mid)
        N, H = assemble_static_hamiltonian!(L, h, V_here, ws)

        # Extract vectors
        diag_r = H.dv
        off_r = H.ev

        # Build CN operators
        # Note: The signs here are correct for (I + iHdt/2)psi_new = (I - iHdt/2)psi_old
        # factor = i * dt / 2
        factor = im * (dt / 2.0)
        
        # Construct diagonals for A (LHS) and B (RHS)
        # LHS: (1 + i dt/2 H)
        diagA = ComplexF64.(1.0 .+ factor .* diag_r)
        offA  = ComplexF64.(factor .* off_r)
        
        # RHS: (1 - i dt/2 H)
        diagB = ComplexF64.(1.0 .- factor .* diag_r)
        offB  = -offA # Simple negation for off-diagonals

        A = Tridiagonal(offA, diagA, offA)
        B = Tridiagonal(offB, diagB, offB)

        # Step: Solve A * psi_new = B * psi_old
        # B * psi is efficient for Tridiagonal
        rhs = B * psi
        
        # In-place update of psi using the backslash operator (efficient for Tridiagonal)
        psi .= A \ rhs

        # Advance full time step
        t += dt

        if basis !== nothing && store
            push!(coeffs, project_onto_basis_L2(psi, basis, h))
        end
    end

    return psi, coeffs
end

"""
    normalise_L2(v, dx; atol=1e-14)


Examples
```julia
dx = 0.01
u = randn(1000)
un = normalise_L2(u, dx)
@assert isapprox(dx * sum(abs2, un), 1.0; rtol=1e-12)

```
"""
function normalise_L2(v::AbstractVector{T}, dx::Real; atol::Real=1e-14) where {T<:Number}
    n2 = dx * sum(abs2, v)
    if !isfinite(n2) || n2 < atol
        throw(ArgumentError("Cannot normalize: dx-weighted L² norm ≈ $(n2)."))
    end
    invn = inv(sqrt(n2))
    return invn .* v
end

function normalise_L2!(v::AbstractVector{T}, dx::Real; atol::Real=1e-14) where {T<:Number}
    dx > 0 || throw(ArgumentError("dx must be positive, got $dx"))
    n2 = dx * sum(abs2, v)
    if !isfinite(n2) || n2 < atol
        throw(ArgumentError("Cannot normalize: dx-weighted L² norm ≈ $(n2)."))
    end
    invn = inv(sqrt(n2))
    v .*= invn
    return v
end

"""
    project_L2(phi, psi, dx) -> c::ComplexF64

dx-weighted L2 projection (inner product) of `psi` onto `phi`:
< phi | psi > = int(conj(phi(x)) * psi(x) dx) approx dx * sum conj(phi_i) * psi_i

"""
function project_L2(phi::AbstractVector{T1}, psi::AbstractVector{T2}, dx::Real) where {T1<:Number,T2<:Number}
    @assert length(phi) == length(psi) "Vectors must have the same length"
    return dx * sum(conj.(phi) .* psi)
end

"""
    project_onto_basis_L2(psi, basis, dx) -> coeffs::Vector{ComplexF64}

Project `psi` onto each column of `basis` using the dx-weighted L2 inner product.
"""
function project_onto_basis_L2(psi::AbstractVector{T}, basis::AbstractMatrix{S}, dx::Real) where {T<:Number,S<:Number}
    @assert size(basis, 1) == length(psi) "Basis row count must match length of psi"
    ncols = size(basis, 2)
    coeffs = Vector{ComplexF64}(undef, ncols)
    @inbounds for i in 1:ncols
        coeffs[i] = dx * sum(conj.(@view basis[:, i]) .* psi)
    end
    return coeffs
end


"""
    fftfreq(N, dt=1.0) -> freqs::Vector{Float64}

Compute the FFT frequency bin centers (in Hz) for a discrete-time signal
of length `N` with sampling spacing `dt` (in seconds). The returned vector
has length `N` and is ordered to correspond to the output of `fft`.

Examples
```
julia> fftfreq(4, 0.25)
[0.0, 1.0, -2.0, -1.0]
```
"""
function fftfreq(N::Int, dt::Real=1.0)
    N >= 1 || throw(ArgumentError("N must be positive, got $(N)"))
    dt > 0 || throw(ArgumentError("dt must be positive, got $(dt)"))
    fs = 1.0 / dt
    freqs = Vector{Float64}(undef, N)
    @inbounds for k in 0:(N - 1)
        if k <= N ÷ 2
            freqs[k + 1] = (k * fs) / N
        else
            freqs[k + 1] = ((k - N) * fs) / N
        end
    end
    return freqs
end

"""
    discrete_fft(y, dt=1.0; norm=:amplitude, one_sided=true)

Compute the discrete Fourier transform of a sampled signal `y` (vector)
using `fft`. Returns a tuple `(freqs, spectrum)` where `freqs` are the
FFT frequency bin centers in Hz and `spectrum` is the (optionally scaled)
complex spectrum values. `norm` controls amplitude normalization (`:none`,
`:amplitude`, or `:unitary`) and `one_sided` (default `true`) will return a
one-sided spectrum for real-valued inputs that contains only non-negative
frequencies (and scales the non-DC / non-Nyquist bins by 2 to conserve
energy).

Examples
```
dt = 0.01
N = 256
t = (0:(N - 1)) .* dt
y = sin.(2π * 10.0 .* t)
f, S = discrete_fft(y, dt; norm=:amplitude, one_sided=true)
```
"""
function discrete_fft(y::AbstractVector{T}, dt::Real=1.0; norm::Symbol=:amplitude, one_sided::Bool=true) where {T<:Number}
    length(y) >= 1 || throw(ArgumentError("y must be non-empty"))
    dt > 0 || throw(ArgumentError("dt must be positive"))
    N = length(y)

    # raw FFT
    Y = fft(ComplexF64.(y))
    freqs = fftfreq(N, dt)

    # Normalisation factor
    scale = nothing
    if norm === :none
        scale = 1.0
    elseif norm === :amplitude
        scale = 1.0 / N
    elseif norm === :unitary
        scale = 1.0 / sqrt(N)
    else
        throw(ArgumentError("norm must be one of :none, :amplitude, :unitary"))
    end

    if one_sided && eltype(y) <: Real
        # One-sided for real signals: keep indices 1 .. floor(N/2)+1
        if iseven(N)
            kmax = N ÷ 2 + 1
        else
            kmax = (N + 1) ÷ 2
        end
        inds = 1:kmax
        S = scale .* Y[inds]
        # Double non-DC and non-Nyquist bins (power in negative freqs)
        if kmax > 1
            # start after DC (index 1), end before Nyquist if even length
            n_fixed = (iseven(N) ? kmax - 2 : kmax - 1)
            if n_fixed > 0
                S[2:(1 + n_fixed)] .= 2.0 .* S[2:(1 + n_fixed)]
            end
        end
        return freqs[inds], S
    else
        return freqs, scale .* Y
    end
end


