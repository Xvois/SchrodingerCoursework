include("../SolutionFunctions.jl")


P = 50.0                 # Choose p so that there are at least three bound states (p >= 6 suffices)
N = 1000               # number of interior grid points
# DOMAIN SIZE scaled with p (width proportional to 1/q = sqrt(p)) to capture the tails well
L = 10.0 / q_of_p(P)     # domain size

# Solve the stationary problem (eigenvalues E and eigenvectors psi on interior grid xi)
E, psi, xi = solve_schrodinger(N, L, q_of_p(P))

# Ensure x-array and eigenvector lengths match (floating-point rounding can make N differ by one)
nrows = size(psi, 1)
nxi = length(xi)
if nxi != nrows
    @warn "Mismatch between xi length and eigenvector rows. Aligning arrays by truncation"
    nplot = min(nxi, nrows)
    if nplot < nrows
    # truncate eigenvectors to match xi length
    psi = psi[1:nplot, :]
    end
    if nplot < nxi
        xi = xi[1:nplot]
    end
end

# Sanity check: require >= 3 bound (negative) states for Q3
nbound = count(<(0.0), E)  # number of E < 0
if nbound < 3
    @warn "Parameter p=$(P) yields only $(nbound) bound states; consider increasing p (p >= 6)."
end

# Prepare grid spacing and normalize ALL eigenvectors in the L^2 sense (integral |psi|^2 dxi = 1)
dx = L / (N + 1)

# Normalize all eigenvectors wit h dx-weighted L^2 norm using helper
@views for i in 1:size(psi, 2)
    normalise_L2!(psi[:, i], dx)
end

# Fundamental mode u0(xi) is now L^2-normalized
fundamental_mode = complex.(Vector(psi[:, 1]))

# Project initial state into eigenbasis coefficients c(0) using dx-weighted L^2 inner product for consistency
c = project_onto_basis_L2(fundamental_mode, psi, dx)

# Validation: check that the initial projection is properly normalized (should be ≈ 1)
norm_check = sum(abs2, c)
println("Initial state normalization check: ∑|cₙ(0)|² = $(round(norm_check, digits=6)) (should be ≈ 1)")
if abs(norm_check - 1.0) > 0.01
    @warn "Initial state projection normalization deviates from 1 by $(abs(norm_check - 1.0))"
end

# Dimensionless time tau satisfies i dpsi/dtau = H psi with evolution exp(-i * E * tau).
# Q3 asks for t in [0, 8*pi*hbar/|E0|] -> in tau this is tau in [0, 8*pi], since tau = t * |E0| / hbar.
E0 = E[1]
tau_end = 8pi
dtau = 0.01
tau = 0:dtau:tau_end

# Evolve coefficients exactly in tau and reconstruct psi(xi, tau)
coeffs = Matrix{ComplexF64}(undef, length(c), length(tau))
cT = copy(c)
for (k, _) in enumerate(tau)
    coeffs[:, k] = cT
    evolve_coeffs!(cT, E, dtau)
end

# Compute the required projection c0(tau) = integral psi(xi, tau) * u0(xi) dxi numerically (dx-weighted inner product)
# Note: with u0n real, conjugation has no effect, but we keep conj for generality.
c0_series = Vector{ComplexF64}(undef, length(tau))
for (k, _) in enumerate(tau)
    psit = psi * coeffs[:, k]           # reconstruct psi(xi, tau_k)
    c0_series[k] = dx * sum(conj.(u0n) .* psit)
end

# Plot Re, Im, and |c0| versus t/T0 where T0 = 2*pi*hbar/|E0| -> t/T0 = tau / (2*pi)
using Plots
xscaled = collect(tau ./ (2pi))      # equals physical time divided by T0
plt = plot(xscaled, real.(c0_series);
    label = "Re(c0)", xlabel = "t / T0", ylabel = "c0(t)",
    title = "Projection onto fundamental mode (p=$(P), N=$(N))",
    legend = :topright, dpi = 500)
plot!(plt, xscaled, imag.(c0_series); label = "Im(c0)")
plot!(plt, xscaled, abs.(c0_series); label = "|c0|")

savefig(plt, "./Q3/Plots/c0_vs_tOverT0_P$(Int(P))_N$(N).png")
