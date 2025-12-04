include("../SolutionFunctions.jl")
using Plots
using LinearAlgebra
using LaTeXStrings

# Ensure output directory exists
if !isdir("Q3/Plots")
    mkpath("Q3/Plots")
end

P = 50.0                 # Choose p so that there are at least three bound states
P = 50.0                 # Choose p so that there are at least three bound states
h = 0.05                 # spatial step size
q = q_of_p(P)
L = 10.0 / q             # domain size
N = round(Int, L / h) - 1

println("Solving static problem for P=$P (h=$h, N=$N)...")
V_static(x) = -sech(q*x)^2
E, psi, xi = solve_static_schrodinger(L, h, V_static)
E_A = analytical_energy_levels(P)

# Normalize eigenvectors (dx-weighted L2)
psi_normalized = similar(psi)
for i in 1:size(psi, 2)
    psi_normalized[:, i] = normalise_L2(psi[:, i], h)
end

# Ground state (for projection)
u0 = ComplexF64.(psi_normalized[:, 1])
E0 = E[1]
println("Ground state energy E0 = $E0")

# Analytical period
T0 = 2 * pi / abs(E_A[1])
t_end = 4 * T0
nsteps = 500
dt = t_end / nsteps
times = collect(range(0.0, t_end, length=nsteps+1))

println("Time range: [0, $(round(t_end, digits=2))] (4 periods)")
println("Using Crank-Nicolson method with dt = $(round(dt, digits=6))")

# Initial state is the ground state
psi_current = copy(u0)

# Time-independent potential for CN evolution
Vxt(x, t) = V_static(x)

# Create workspace for CN solver
ws = SolverWorkspace(N)

# Evolve using Crank-Nicolson and store coefficients at each step
psi_evolved, coeffs_history = evolve_dynamic_coeffs!(
    psi_current, L, h, Vxt, dt, 0.0;
    nsteps=nsteps, ws=ws, basis=psi_normalized, store=true
)

# Extract c0 (ground state coefficient) at each time step
c0_series = [coeffs[1] for coeffs in coeffs_history]

# Plot vs t / T0 (main time-series)
t_scaled = times ./ T0

# Main time-series plot
p_main = plot(t_scaled, real.(c0_series), label=L"Re(c_0)", xlabel=L"t / T_0", ylabel=L"c_0(t)",
           lw=2, dpi=300, fontfamily="Computer Modern", guidefontsize=12, tickfontsize=10)
plot!(p_main, t_scaled, imag.(c0_series), label=L"Im(c_0)", lw=2, ls=:dash)

# Prepare Argand inset bbox (bottom-right)
# Compute scale for inset (unit circle or larger to fit trajectory)
rmax = maximum(abs.(c0_series))
rplot = max(1.0, rmax)
theta = range(0, 2pi, length=360)
circlex = rplot .* cos.(theta)
circley = rplot .* sin.(theta)

# Trajectory
xtraj = real.(c0_series)
ytraj = imag.(c0_series)

# Inset placement (bottom-right inside main axes)
inset_bbox = bbox(0.62, 0.06, 0.34, 0.34, :bottom, :right, :inner)

# Build Argand plot separately and insert as inset to avoid drawing on main axes
p_arg = plot(circlex, circley, linecolor=:lightgray, lw=1.2, legend=false,
             xlabel="", ylabel="", aspect_ratio=:equal, framestyle=:box,
             guidefontsize=8, tickfontsize=8, grid=false,
             xlim=(-rplot-0.05, rplot+0.05), ylim=(-rplot-0.05, rplot+0.05))
plot!(p_arg, xtraj, ytraj, linecolor=:blue, lw=1.2, legend=false)
scatter!(p_arg, [xtraj[end]], [ytraj[end]], color=:red, markersize=4, label=false)

# Add a few tick marks to inset axes for reference
nticks = 5
ticks_vals = collect(range(-rplot, rplot, length=nticks))
plot!(p_arg, xticks=ticks_vals, yticks=ticks_vals)

# Insert Argand as an inset in the bottom-right of the main plot
plot!(p_main, p_arg, inset=(1, inset_bbox))

savefig(p_main, "Q3/Plots/c0_vs_tOverT0.png")
println("Done. Plot saved to Q3/Plots/c0_vs_tOverT0.png")
