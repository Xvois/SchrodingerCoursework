include("../SolutionFunctions.jl")
using Plots
using LinearAlgebra

# Ensure output directory exists
if !isdir("Q3/Plots")
    mkpath("Q3/Plots")
end

P = 50.0                 # Choose p so that there are at least three bound states
N = 400                  # number of interior grid points
q = q_of_p(P)
L = 20.0 / q             # domain size

println("Solving static problem for P=$P...")
V_static(x) = -sech(q*x)^2
E, psi, xi = solve_static_schrodinger(N, L, V_static)

# Normalize eigenvectors (dx-weighted L2)
h = L / (N + 1)
scale_factor = 1.0 / sqrt(h)
psi_normalized = psi .* scale_factor

# Ground state (for projection)
u0 = ComplexF64.(psi_normalized[:, 1])
E0 = E[1]
println("Ground state energy E0 = $E0")

T0 = 2 * pi / abs(E0)
t_end = 4 * T0
times = collect(range(0.0, t_end, length=500))
dt = times[2] - times[1]

println("Time range: [0, $(round(t_end, digits=2))] (4 periods)")

# Initial state is the ground state
psi_current = copy(u0)

coeffs = project_onto_basis_L2(psi_current, psi_normalized, h)

c0_series = Vector{ComplexF64}(undef, length(times))

psi_reconstructed = Vector{ComplexF64}(undef, N)

for i in 1:length(times)
    t = times[i]
    
    current_coeffs = coeffs .* exp.(-im .* E .* t)
    
    mul!(psi_reconstructed, psi_normalized, current_coeffs)
    
    c0_series[i] = project_L2(u0, psi_reconstructed, h)
end

# Plot vs t / T0
t_scaled = times ./ T0

plt = plot(t_scaled, real.(c0_series), label="Re(c0)", xlabel="t / T0", ylabel="c0(t)",
           title="Time Evolution of Ground State Projection (P=$(Int(P)))",
           lw=2, dpi=300)
plot!(plt, t_scaled, imag.(c0_series), label="Im(c0)", lw=2, ls=:dash)
plot!(plt, t_scaled, abs.(c0_series), label="|c0|", lw=2, color=:black)

savefig(plt, "Q3/Plots/c0_vs_tOverT0.png")
println("Done. Plot saved to Q3/Plots/c0_vs_tOverT0.png")
