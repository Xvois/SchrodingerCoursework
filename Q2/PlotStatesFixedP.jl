include("../SolutionFunctions.jl")
using Plots

## ALL PARAMETERS TO BE SET HERE

P = 30.0 # Our dimensionless parameter
h = 0.05  # desired spatial step size
q = q_of_p(P)
L = 15.0 / q  # domain size
N = round(Int, L / h) - 1
println("Using domain size L = $L for P = $P (h=$h, N=$N)")

# Solve the Schrodinger equation numerically
V(x) = -sech(q*x)^2
E, psi, xi = @time solve_static_schrodinger(L, h, V)

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

# Compute and print the average percentage error for the first 5 states
ncomp = min(5, length(E), length(E_analytical))
percent_errors = [percent_error(E_analytical[i], E[i]) for i in 1:ncomp]
avg_percent_error = mean_value(percent_errors)
println("Average percentage error across first $ncomp states: $(round(avg_percent_error, digits=4))%")

println("Lowest 5 energy eigenvalues:")
println("Analytical: ", E_analytical)
println("Numerical:  ", E[1:5])

nstates = min(5, size(psi, 2))

styles = [:solid, :dash, :dot, :dashdot, :dashdotdot]
maxamp = maximum(abs, psi[:, 1:nstates])
gap = 1.6 * maxamp                       
offsets = (0:nstates-1) .* gap
colors = palette(:viridis, nstates)
plt_offset = plot(; legend = false, dpi = 500, size = (540, 360),
    xlabel = math_label("\\xi"),
    fontfamily = "Computer Modern",
    guidefontsize = 16,
    tickfontsize = 12)
for i in 1:nstates
    hline!([offsets[i]]; c = :gray, alpha = 0.35, lw = 1, label = "")
    plot!(xi, psi[:, i] .+ offsets[i];
        lw = 5, linestyle = styles[i], color = colors[(i - 1) % length(colors) + 1], label = "")
end
yticks_vals = offsets
yticks_labs = ["n=$(i-1)" for i in 1:nstates]
yticks!(plt_offset, yticks_vals, yticks_labs)
savefig(plt_offset, "./Q2/Plots/PlotStates_offset_P$(Int(P))_h$(h)_N$(N)_clear.png")
