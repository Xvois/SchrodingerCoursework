include("../SolutionFunctions.jl")

using Plots

## PARAMETERS
h = 0.05  # Fixed spatial step size for all runs
L_fixed = 500.0  # Fixed domain size L
const P_values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]  # Multiple P values to test
N = round(Int, L_fixed / h) - 1  # Number of grid points based on fixed L and h

# Number of states to compare (will use minimum available across all P values)
max_states = 10

println("Fixed parameters: N=$N, L=$L_fixed")
println("Testing P values: $P_values")
println()

# Storage for results
results = Dict{Float64, Vector{Float64}}()

for P in P_values
    q = q_of_p(P)
    L = L_fixed  # Use fixed L for all P values
    
    # Solve numerically using fixed h
    V(x) = -sech(q*x)^2
    E, psi, xi = solve_static_schrodinger(L, h, V)
    
    # Get analytical energies
    E_analytical = analytical_energy_levels(P)
    
    # Compute per-state percentage errors
    ncomp = min(max_states, length(E), length(E_analytical))
    percent_errors = Float64[]
    for i in 1:ncomp
        err = percent_error(E_analytical[i], E[i])
        # Filter out Inf/NaN values (can occur if analytical energy is zero or numerical solver failed)
        if isfinite(err)
            push!(percent_errors, err)
        else
            @warn "Non-finite error for P=$P, state n=$(i-1): E_analytical=$(E_analytical[i]), E_numerical=$(E[i])"
            push!(percent_errors, NaN)
        end
    end
    
    # Store results
    results[P] = percent_errors
    
    valid_errors = filter(isfinite, percent_errors)
    avg_err = isempty(valid_errors) ? NaN : mean_value(abs.(valid_errors))
    println("  Computed $ncomp states, avg error = $(round(avg_err, digits=4))%")
end

println("\nGenerating plot...")

# Create plot (scatter-only, jittered x to avoid overlap, improved styling)
max_n = maximum(length(results[P]) for P in P_values) - 1
colors = palette(:viridis, length(P_values))
markers = [:circle, :square, :diamond, :utriangle, :dtriangle, :hexagon]

plt = plot(
    xlabel = math_label("n"),
    ylabel = math_label("\\mathrm{Error\\ (\\%)}"),
    legend = :bottomright,
    legendfontsize = 12,
    legend_background_color_alpha = 0.0,
    dpi = 500,
    yscale = :log10,
    fontfamily = "Computer Modern",
    guidefontsize = 16,
    tickfontsize = 12,
    size = (540, 360),
    ygrid = true,
    xgrid = false,
    xticks = 0:1:max_n
)

# Plot error vs state number for each P (scatter series, no connecting lines)
for (idx, P) in enumerate(P_values)
    errors = results[P]
    n_values = 0:(length(errors)-1)  # State indices starting from 0

    # Filter out NaN/Inf values for plotting
    valid_indices = findall(isfinite, errors)
    if !isempty(valid_indices)
        x = n_values[valid_indices]
        y = abs.(errors[valid_indices])
        plot!(plt, x, y;
            seriestype = :scatter,
            label = "P=$(Int(P))",
            markersize = 6,
            marker = markers[idx],
            color = colors[idx],
            markerstrokecolor = :black,
            markerstrokewidth = 0.4,
            alpha = 0.95
        )
    else
        @warn "No valid errors to plot for P=$P"
    end
end

# Ensure target directory exists, then save
mkpath("./Q2/Plots")
outfile = "./Q2/Plots/ErrorVsState_MultipleP_h$(h)_L$(Int(L_fixed)).png"
savefig(plt, outfile)
println("Plot saved to $outfile")

# Print summary table
println("\nSummary of percentage errors by state:")
println("─"^70)
print("n    ")
for P in P_values
    print("P=$(Int(P))      ")
end
println()
println("─"^70)

max_rows = maximum(length(results[P]) for P in P_values)
for n in 0:(max_rows-1)
    print("$n    ")
    for P in P_values
        errors = results[P]
        if n < length(errors)
            err = errors[n+1]
            if isfinite(err)
                print(format_percentage(abs(err); digits=4, pad=9), "   ")
            else
                print("N/A       ")
            end
        else
            print("N/A       ")
        end
    end
    println()
end
println("─"^70)
