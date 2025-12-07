include("../SolutionFunctions.jl")
using Plots

# Ensure output directory exists
if !isdir("Q3/Plots")
    mkpath("Q3/Plots")
end

P = 30.0                 # Choose p so that there are at least three bound states
h = 0.05                 # spatial step size
q = q_of_p(P)
L = 15.0 / q             # domain size
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

# Prepare Argand inset (unit circle to show |c0| = 1 is preserved)
theta = range(0, 2pi, length=360)
unit_circlex = cos.(theta)
unit_circley = sin.(theta)

# Trajectory in complex plane
xtraj = real.(c0_series)
ytraj = imag.(c0_series)

# Create main plot (left subplot)
p_main = plot(t_scaled, real.(c0_series), label=math_label("\\mathrm{Re}(c_0)"), xlabel=math_label("t / T_0"), ylabel=math_label("c_0(t)"),
           lw=1.5, dpi=500, fontfamily="Computer Modern",
           guidefontsize=16, tickfontsize=12, legendfontsize=12,
           legend=:bottomleft)
plot!(p_main, t_scaled, imag.(c0_series), label=math_label("\\mathrm{Im}(c_0)"), lw=1.5, ls=:dash)

# Prepare directional arrows evenly spaced around the unit circle
# Place arrows at evenly spaced angles around the circle
n_arrows = 10
arrow_angles = range(0, 2pi, length=n_arrows+1)[1:end-1]  # evenly spaced angles
arrow_x = Float64[]
arrow_y = Float64[]
arrow_dx = Float64[]
arrow_dy = Float64[]
arrow_scale = 0.08  # size of arrows

for angle in arrow_angles
    # Position on unit circle
    x_pos = cos(angle)
    y_pos = sin(angle)
    # Direction is tangent to circle (clockwise rotation, so negative angular direction)
    # Tangent vector for clockwise motion: (sin(angle), -cos(angle))
    dx = sin(angle) * arrow_scale
    dy = -cos(angle) * arrow_scale
    push!(arrow_x, x_pos)
    push!(arrow_y, y_pos)
    push!(arrow_dx, dx)
    push!(arrow_dy, dy)
end

# Create Argand diagram (right subplot) with matching style
p_argand = plot(unit_circlex, unit_circley, 
      linecolor=:gray, lw=1, ls=:dash, legend=false,
      aspect_ratio=:equal, framestyle=:box,
      fontfamily="Computer Modern", 
      guidefontsize=16, tickfontsize=12,
    xlabel=math_label("\\mathrm{Re}(c_0)"), ylabel=math_label("\\mathrm{Im}(c_0)"),
      xlim=(-1.18, 1.18), ylim=(-1.18, 1.18),
      xticks=[-1, 0, 1], yticks=[-1, 0, 1], titlefontsize=12)

# Plot the trajectory on the Argand diagram
plot!(p_argand, xtraj, ytraj, lw=1.5, color=:blue)

# Add directional arrows
for i in 1:length(arrow_x)
    quiver!(p_argand, [arrow_x[i]], [arrow_y[i]], quiver=([arrow_dx[i]], [arrow_dy[i]]),
            color=:red, lw=1)
end


# Combine plots side by side with left plot larger
p_combined = plot(p_main, p_argand, layout=grid(1, 2, widths=[0.7, 0.3]), size=(620, 350), dpi=500, margin=2Plots.mm)

savefig(p_combined, "Q3/Plots/c0_vs_tOverT0.png")
println("Done. Plot saved to Q3/Plots/c0_vs_tOverT0.png")
