include("../SolutionFunctions.jl")
using Plots

# Parameters
P = 50.0                 # Choose p so that there are at least three bound states
N = 500                  # number of interior grid points
q = q_of_p(P)
L = 10.0 / q             # domain size
h = L / (N + 1)          # grid spacing
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
plt_mag = plot(t_scaled, magnitudes[1, :], label="n=0 (ground)", lw=2, 
     xlabel="tΩ/(2π)", ylabel="|cₙ|", title="State Magnitudes vs Scaled Time", dpi=300)
plot!(plt_mag, t_scaled, magnitudes[2, :], label="n=1", lw=2)
plot!(plt_mag, t_scaled, magnitudes[3, :], label="n=2 (resonant)", lw=2)
if size(magnitudes, 1) >= 4
    plot!(plt_mag, t_scaled, magnitudes[4, :], label="n=3", lw=2)
end
savefig(plt_mag, "Q4/state_magnitudes.png")
println("Saved Q4/state_magnitudes.png")

# Check total probability conservation (sum of squared magnitudes)
total_prob = sum(abs2.(coeffs_matrix), dims=1)[:]
plt_prob = plot(t_scaled, total_prob, label="Σ|cₙ|² (first $n_proj states)", lw=2, 
     xlabel="tΩ/(2π)", ylabel="Total Probability", 
     title="Probability Conservation Check", dpi=300)
hline!(plt_prob, [1.0], ls=:dash, color=:red, label="Expected")
savefig(plt_prob, "Q4/probability_conservation.png")
println("Saved Q4/probability_conservation.png")

println("\nFinal populations:")
for i in 1:min(5, size(magnitudes, 1))
    println("  State n=$(i-1): |c$(i-1)(T)|² = $(magnitudes[i, end]^2)")
end
