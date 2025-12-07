include("../SolutionFunctions.jl")
using Plots
using LaTeXStrings
using Printf

# Parameters
P = 30.0                 # Choose p so that there are at least three bound states
h = 0.05                 # spatial step size
q = q_of_p(P)
L = 15.0 / q             # domain size
N = round(Int, L / h) - 1
eta = 0.1             # perturbation amplitude

# Step A: Solve the TISE to get eigenstates and eigenvalues
println("Step A: Solving TISE for static potential...")
V_static(xi) = -sech(q * xi)^2
ws = SolverWorkspace(N)
E, U, xi = solve_static_schrodinger(L, h, V_static, ws)

# Store the eigenvalues we need
epsilon_0 = E[1]
epsilon_2 = E[3]
Omega = epsilon_2 - epsilon_0  # resonant frequency

println("Found $(length(E)) bound states")
println("ε₀ = $(epsilon_0)")
println("ε₂ = $(epsilon_2)")
println("Ω = ε₂ - ε₀ = $(Omega)")

# Define time-dependent potential
V_dynamic(xi, tau) = V_static(xi) * (1.0 + eta * sin(Omega * tau))

# Step B: Set initial condition - start in ground state u₀
println("\nStep B: Setting initial condition Ψ(ξ,0) = u₀(ξ)...")

# Normalise all states using L2 norm
U_norm = similar(U)
for i in 1:size(U, 2)
    U_norm[:, i] = normalise_L2(U[:, i], h)
end

psi = ComplexF64.(U_norm[:, 1])  # ground state eigenvector

# Step C: Evolve using Crank-Nicolson
println("\nStep C: Evolving wavefunction with time-dependent potential...")
dt = 0.1                    # time step
T_final = 8π/Omega              # total evolution time
nsteps = round(Int, T_final / dt)

# Evolve and project onto basis at each step
# Optimization: only project onto the first few states to save time
n_proj = min(10, size(U_norm, 2))
basis_proj = U_norm[:, 1:n_proj]

psi_evolved, coeffs_history = evolve_dynamic_coeffs!(
    psi, L, h, V_dynamic, dt;
    nsteps=nsteps,
    ws=ws,
    basis=basis_proj,
    store=true
)

# Step D: Analyze coefficients
println("\nStep D: Analyzing state coefficients...")
coeffs_matrix = hcat(coeffs_history...)  # shape: (n_proj, n_timesteps)
magnitudes = abs.(coeffs_matrix)         # |cₙ(τ)|

# Time array scaled by Ω/2π
times = range(0, T_final, length=nsteps+1)
t_scaled = times .* Omega ./ (2 * pi)

# Plot magnitudes of first few states
"""
Produce a side-by-side figure: left = full-time traces (as before),
right = a zoomed plot highlighting the micro-oscillation of the n=2 state.
Improve font sizes and overall plot size for paper readability.
"""

function moving_average(x::AbstractVector{T}, window::Int) where T
    n = length(x)
    y = zeros(eltype(x), n)
    half = fld(window, 2)
    for i in 1:n
        lo = max(1, i - half)
        hi = min(n, i + half)
        y[i] = sum(x[lo:hi]) / (hi - lo + 1)
    end
    return y
end

# Compute smoothed envelope and micro-oscillation for n=2 (index 3)
idx_n2 = 3
smoothed_n2 = moving_average(magnitudes[idx_n2, :], 21)
oscillation_n2 = magnitudes[idx_n2, :] .- smoothed_n2

# Choose a zoom window (scaled time). One driving period corresponds to 1 in t_scaled,
# so this zoom shows about one period where oscillations are visible (mid simulation).
Ttot = t_scaled[end]
zoom_start = Ttot * 0.55
zoom_width = 1.0
zoom_end = zoom_start + zoom_width
idx_zoom = findall(t -> (t >= zoom_start) && (t <= zoom_end), t_scaled)

pl_left = plot(t_scaled, magnitudes[1, :], label="n=0 (ground)", lw=2,
    xlabel=L"t\Omega/(2\pi)", ylabel=L"|c_n|", guidefont=font(16), tickfont=font(12),
    legend=:left, legendfontsize=12)
plot!(pl_left, t_scaled, magnitudes[2, :], label="n=1", lw=2)
plot!(pl_left, t_scaled, magnitudes[3, :], label="n=2 (resonant)", lw=2, color=:green)
if size(magnitudes, 1) >= 4
    plot!(pl_left, t_scaled, magnitudes[4, :], label="n=3", lw=2)
end


# Only keep the left/main plot for a cleaner figure
plt_combined = plot(pl_left, size=(620, 350), dpi=500, margin=2Plots.mm)
savefig(plt_combined, "Q4/state_magnitudes.png")
println("Saved Q4/state_magnitudes.png")

# Check total probability conservation (sum of squared magnitudes)
total_prob = sum(abs2.(coeffs_matrix), dims=1)[:]
plt_prob = plot(t_scaled, total_prob, label=L"\sum |c_n|^2", lw=1.5, 
    xlabel=L"t\Omega/(2\pi)", ylabel="Total Probability", 
     title="Probability Conservation Check", dpi=300,
     fontfamily="Computer Modern", guidefontsize=12, tickfontsize=10)
hline!(plt_prob, [1.0], ls=:dash, color=:red, label="Expected")
savefig(plt_prob, "Q4/probability_conservation.png")
println("Saved Q4/probability_conservation.png")

println("\nFinal populations:")
for i in 1:min(5, size(magnitudes, 1))
    println("  State n=$(i-1): |c$(i-1)(T)|² = $(magnitudes[i, end]^2)")
end
