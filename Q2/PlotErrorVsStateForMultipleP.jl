include("../SolutionFunctions.jl")

using Plots
using Statistics
using Printf
using LaTeXStrings

## PARAMETERS
h = 0.05  # Fixed spatial step size for all runs
L_fixed = 500.0  # Fixed domain size L
P_values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]  # Multiple P values to test
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
    avg_err = isempty(valid_errors) ? NaN : mean(abs.(valid_errors))
    println("  Computed $ncomp states, avg error = $(round(avg_err, digits=4))%")
end

println("\nGenerating plot...")

# Create plot
plt = plot(
    xlabel = L"n", 
    ylabel = L"\mathrm{Error\ (\%)}",
    legend = :topleft,
    dpi = 500,
    yscale = :log10,
    fontfamily="Computer Modern", guidefontsize=12, tickfontsize=10
)

# Plot error vs state number for each P
colors = palette(:viridis, length(P_values))
for (idx, P) in enumerate(P_values)
    errors = results[P]
    n_values = 0:(length(errors)-1)  # State indices starting from 0
    
    # Filter out NaN/Inf values for plotting
    valid_indices = findall(isfinite, errors)
    if !isempty(valid_indices)
        plot!(plt, n_values[valid_indices], abs.(errors[valid_indices]);
            label = "P=$(Int(P))",
            lw = 2,
            marker = :circle,
            markersize = 4,
            color = colors[idx]
        )
    else
        @warn "No valid errors to plot for P=$P"
    end
end

savefig(plt, "./Q2/Plots/ErrorVsState_MultipleP_h$(h)_L$(Int(L_fixed)).png")
# Save plot
println("Plot saved to ./Q2/Plots/ErrorVsState_MultipleP_h$(h)_L$(Int(L_fixed)).png")

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
                @printf("%.4f%%   ", abs(err))
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
