include("../SolutionFunctions.jl")
using Plots

# Parameters
P = 50.0                 # Choose p so that there are at least three bound states
N = 400                  # number of interior grid points
q = q_of_p(P)
L = 20.0 / q             # domain size
h = L / (N + 1)          # grid spacing
eta = 0.01               # perturbation amplitude

# Step A: Solve the TISE to get eigenstates and eigenvalues
println("Step A: Solving TISE for static potential...")
V_static(xi) = -sech(q * xi)^2
E, U, xi = solve_static_schrodinger(L, h, V_static)

# Store the eigenvalues we need
epsilon_0 = E[1]
epsilon_2 = E[3]
Omega = epsilon_2 - epsilon_0  # resonant frequency

println("Found $(length(E)) bound states")
println("ε₀ = $(epsilon_0)")
println("ε₂ = $(epsilon_2)")
println("Ω = ε₂ - ε₀ = $(Omega)")

# Define time-dependent potential
V_dynamic(xi, tau) = V_static(xi) * (1.0 - eta * sin(Omega * tau))

# Step B: Set initial condition - start in ground state u₀
println("\nStep B: Setting initial condition Ψ(ξ,0) = u₀(ξ)...")
psi = ComplexF64.(U[:, 1])  # ground state eigenvector
normalise_L2!(psi, h)        # ensure normalization

# Step C: Evolve using Crank-Nicolson
println("\nStep C: Evolving wavefunction with time-dependent potential...")
dt = 0.01                    # time step
T_final = 100.0              # total evolution time
nsteps = round(Int, T_final / dt)

# Evolve and project onto basis at each step
psi_evolved, coeffs_history = evolve_dynamic_coeffs!(
    psi, L, h, V_dynamic, dt;
    nsteps=nsteps,
    ws=ws,
    basis=U,
    store=true
)

# Step D: Analyze coefficients (populations)
println("\nStep D: Analyzing state populations...")
# Convert vector of vectors to matrix for easier plotting
coeffs_matrix = hcat(coeffs_history...)  # shape: (n_states, n_timesteps)
populations = abs2.(coeffs_matrix)       # |cₙ(τ)|²

# Time array
times = range(0, T_final, length=nsteps+1)

# Plot populations of first few states
plot(times, populations[1, :], label="n=0 (ground)", lw=2, 
     xlabel="Time τ", ylabel="|cₙ(τ)|²", title="State Populations vs Time")
plot!(times, populations[2, :], label="n=1", lw=2)
plot!(times, populations[3, :], label="n=2 (resonant)", lw=2)
if size(populations, 1) >= 4
    plot!(times, populations[4, :], label="n=3", lw=2)
end
savefig("Q4/state_populations.png")
println("Saved Q4/state_populations.png")

# Check total probability conservation
total_prob = sum(populations, dims=1)[:]
plot(times, total_prob, label="Σ|cₙ|²", lw=2, 
     xlabel="Time τ", ylabel="Total Probability", 
     title="Probability Conservation Check")
hline!([1.0], ls=:dash, color=:red, label="Expected")
savefig("Q4/probability_conservation.png")
println("Saved Q4/probability_conservation.png")

println("\nFinal populations:")
for i in 1:min(5, size(populations, 1))
    println("  State n=$(i-1): |c$(i-1)(T)|² = $(populations[i, end])")
end