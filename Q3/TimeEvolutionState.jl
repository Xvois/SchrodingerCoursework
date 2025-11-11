include("../SolutionFunctions.jl")


P = 50.0                 # Choose p so that there are at least three bound states (p >= 6 suffices)
N = 500               # number of interior grid points
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

# Prepare grid spacing and normalize fundamental mode u0(xi) by the integral <u0,u0> = integral |u0|^2 dxi approx dx * sum |u0_j|^2
dx = L / (N + 1)
u0 = @view psi[:, 1]
u0_intnorm = sqrt(dx * sum(abs2, u0))
# u0n normalized so that integral |u0|^2 dxi = 1
u0n = u0 ./ u0_intnorm

# Initial condition psi(xi, 0) = A u0(xi) with A set so that integral |psi|^2 dxi = 1 -> A = 1 for u0n
psi0 = complex.(u0n)                # promote to complex for time evolution

# Project initial state into eigenbasis coefficients c(0) = V' * psi0 (Euclidean). These evolve as exact phases in tau.
c0 = complex.(psi' * psi0)

# Dimensionless time tau satisfies i dpsi/dtau = H psi with evolution exp(-i * E * tau).
# Q3 asks for t in [0, 8*pi*hbar/|E0|] -> in tau this is tau in [0, 8*pi], since tau = t * |E0| / hbar.
E0 = E[1]
tau_end = 8pi
dtau = 0.01
tau = 0:dtau:tau_end

# Evolve coefficients exactly in tau and reconstruct psi(xi, tau)
coeffs = Matrix{ComplexF64}(undef, length(c0), length(tau))
cT = copy(c0)
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

savefig(plt, "Coursework/Q3/Plots/c0_vs_tOverT0_P$(Int(P))_N$(N).png")
