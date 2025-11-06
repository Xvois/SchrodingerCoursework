include("../SolutionFunctions.jl")


P = 50.0                 # Choose p so that there are at least three bound states (p ≥ 6 suffices)
N = 500               # number of interior grid points
# DOMAIN SIZE scaled with p (width ∝ 1/q = √p) to capture the tails well
L = 10.0 / q_of_p(P)     # domain size

# Solve the stationary problem (eigenvalues E and eigenvectors ψ on interior grid xi)
E, ψ, xi = solve_schrodinger(N, L, q_of_p(P))

# Ensure x-array and eigenvector lengths match (floating-point rounding can make N differ by one)
nrows = size(ψ, 1)
nxi = length(xi)
if nxi != nrows
    @warn "Mismatch between xi length and eigenvector rows. Aligning arrays by truncation"
    nplot = min(nxi, nrows)
    if nplot < nrows
        # truncate eigenvectors to match xi length
        ψ = ψ[1:nplot, :]
    end
    if nplot < nxi
        xi = xi[1:nplot]
    end
end

# Sanity check: require ≥ 3 bound (negative) states for Q3
nbound = count(<(0.0), E)  # number of E < 0
if nbound < 3
    @warn "Parameter p=$(P) yields only $(nbound) bound states; consider increasing p (p ≥ 6)."
end

# Prepare grid spacing and normalize fundamental mode u0(ξ) by the integral ⟨u0,u0⟩ = ∫ |u0|^2 dξ ≈ dx * Σ |u0_j|^2
dx = L / (N + 1)
u0 = @view ψ[:, 1]
u0_intnorm = sqrt(dx * sum(abs2, u0))
u0n = u0 ./ u0_intnorm            # u0 normalized so that ∫ |u0|^2 dξ = 1

# Initial condition ψ(ξ, 0) = A u0(ξ) with A set so that ∫ |ψ|^2 dξ = 1 → A = 1 for u0n
ψ0 = complex.(u0n)                # promote to complex for time evolution

# Project initial state into eigenbasis coefficients c(0) = V' * ψ0 (Euclidean). These evolve as exact phases in τ.
c0 = complex.(ψ' * ψ0)

# Dimensionless time τ satisfies i ∂ψ/∂τ = H ψ with evolution exp(−i E τ).
# Q3 asks for t ∈ [0, 8π ħ/|E0|] → in τ this is τ ∈ [0, 8π], since τ = t |E0| / ħ.
E0 = E[1]
τ_end = 8pi
dτ = 0.01
τ = 0:dτ:τ_end

# Evolve coefficients exactly in τ and reconstruct ψ(ξ, τ)
coeffs = Matrix{ComplexF64}(undef, length(c0), length(τ))
cT = copy(c0)
for (k, _) in enumerate(τ)
    coeffs[:, k] = cT
    evolve_coeffs!(cT, E, dτ)
end

# Compute the required projection c₀(τ) = ∫ ψ(ξ, τ) u₀(ξ) dξ numerically (dx-weighted inner product)
# Note: with u0n real, conjugation has no effect, but we keep conj for generality.
c0_series = Vector{ComplexF64}(undef, length(τ))
for (k, _) in enumerate(τ)
    ψt = ψ * coeffs[:, k]           # reconstruct ψ(x, τ_k)
    c0_series[k] = dx * sum(conj.(u0n) .* ψt)
end

# Plot Re, Im, and |c₀| versus t/T0 where T0 = 2π ħ/|E0| → t/T0 = τ / (2π)
using Plots
xscaled = collect(τ ./ (2pi))      # equals physical time divided by T0
plt = plot(xscaled, real.(c0_series);
    label = "Re(c₀)", xlabel = "t / T₀", ylabel = "c₀(t)",
    title = "Projection onto fundamental mode (p=$(P), N=$(N))",
    legend = :topright, dpi = 500)
plot!(plt, xscaled, imag.(c0_series); label = "Im(c₀)")
plot!(plt, xscaled, abs.(c0_series); label = "|c₀|")

savefig(plt, "Coursework/Q3/Plots/c0_vs_tOverT0_P$(Int(P))_N$(N).png")
