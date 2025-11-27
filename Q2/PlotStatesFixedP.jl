include("../SolutionFunctions.jl")

## ALL PARAMETERS TO BE SET HERE

P = 50.0 # Our dimensionless parameter
N = 1000  # number of interior points
# DOMAIN SIZE HAS TO BE ADJUSTED BASED ON P
q = q_of_p(P)
L = 10.0*1/q  # domain size
println("Using domain size L = $L for P = $P")

# Solve the Schrodinger equation numerically
V(x) = -sech(q*x)^2
E, psi, xi = @time solve_static_schrodinger(N, L, V)

# Ensure x-array and eigenvector lengths match (floating-point rounding can make N differ by one)
nrows = size(psi, 1)
nxi = length(xi)
if nxi != nrows
    @warn "Mismatch between xi length and eigenvector rows. Aligning arrays by truncation." xi_len=nxi psi_rows=nrows
    nplot = min(nxi, nrows)
    if nplot < nrows
        # truncate eigenvectors to match xi length
        psi = psi[1:nplot, :]
    end
    if nplot < nxi
        xi = xi[1:nplot]
    end
end

# Solve analytically
E_analytical = analytical_energy_levels(P)

using Statistics

# Compute and print the average percentage error for the first 5 states
ncomp = min(5, length(E), length(E_analytical))
percent_errors = [percent_error(E_analytical[i], E[i]) for i in 1:ncomp]
avg_percent_error = mean(percent_errors)
println("Average percentage error across first $ncomp states: $(round(avg_percent_error, digits=4))%")

using Plots

println("Lowest 5 energy eigenvalues:")
println("Analytical: ", E_analytical)
println("Numerical:  ", E[1:5])
# Plot the lowest energy eigenfunctions with thicker lines and distinct styles
styles = [:solid, :dash, :dot, :dashdot, :dashdotdot]
plt = plot(
    xi, psi[:, 1];
    label = "n=0", xlabel = "xi", ylabel = "psi(xi)",
    legend = :topright, dpi = 500,
    lw = 3, linestyle = styles[1],
    guidefont = font(16), tickfont = font(12)
)
# Plot next few eigenfunctions with different line styles
for (i, n) in enumerate(2:5)
    plot!(
        xi, psi[:, n];
        label = "n=$(n-1)", lw = 3, linestyle = styles[i + 1]
    )
end
savefig(plt, "./Q2/Plots/PlotStates_P$(Int(P))_N$(N).png")


nstates = min(5, size(psi, 2))

maxamp = maximum(abs, psi[:, 1:nstates])
gap = 1.6 * maxamp                       # a bit more separation for clarity
offsets = (0:nstates-1) .* gap
colors = palette(:tab10)
plt_offset = plot(; legend = false, dpi = 500, size = (720, 480),
    xlabel = "xi", ylabel = "state offsets",
    guidefont = font(16), tickfont = font(12))
for i in 1:nstates
    # light baseline at each offset with state label as y-tick
    hline!([offsets[i]]; c = :gray, alpha = 0.35, lw = 1, label = "")
    plot!(xi, psi[:, i] .+ offsets[i];
        lw = 5, linestyle = styles[i], color = colors[(i - 1) % length(colors) + 1], label = "")
end
yticks_vals = offsets
yticks_labs = ["n=$(i-1)" for i in 1:nstates]
yticks!(plt_offset, yticks_vals, yticks_labs)
annot_x = xi[1] + 0.1 * (xi[end] - xi[1])
for i in 1:nstates
    annotate!(annot_x, offsets[i] + 0.2 * gap,
    text("E=$(round(E[i], digits = 3))", 16, :black))
end
savefig(plt_offset, "./Q2/Plots/PlotStates_offset_P$(Int(P))_N$(N)_clear.png")
