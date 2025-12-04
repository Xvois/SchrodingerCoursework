using Base.Threads
using Plots
using LinearAlgebra
using LaTeXStrings

include("../SolutionFunctions.jl")

# Ensure output directory exists
if !isdir("Q2/Plots")
    mkpath("Q2/Plots")
end

P = 50.0
q = q_of_p(P)

# Fix L to be large enough so boundary error is negligible
# From previous analysis, Lq > 15 is usually safe.
L = 10.0 / q 

println("Analyzing discretization error for P=$P, L=$L")

# Analytical energies
E_analytical_full = analytical_energy_levels(P)
levels_to_plot = min(5, length(E_analytical_full))
E_analytical = E_analytical_full[1:levels_to_plot]

# Range of N values (grid points)
# h = L / (N+1) -> N = L/h - 1
N_values = round.(Int, 10 .^ range(1.5, 4.0, length=50)) # 30 to 10000

# Storage for errors
# E_errors[level, N_idx]
E_errors = zeros(Float64, levels_to_plot, length(N_values))

println("Computing errors for $(length(N_values)) different grid sizes...")

Threads.@threads for i in 1:length(N_values)
    N = N_values[i]
    h = L / (N + 1)
    
    # Solve
    # We use the static solver directly
    # Note: solve_static_schrodinger(N, L, V) computes h internally
    V(x) = -sech(q*x)^2
    
    # We can't use the workspace efficiently here because N changes every time.
    # Just call the allocating version or create a one-off workspace.
    # Since N varies, we just call the convenience function.
    
    # However, solve_static_schrodinger(N, L, V) returns (evals, evecs, xi)
    # We only need evals.
    
    # To avoid allocating eigenvectors (which are large for large N), 
    # we might want to call assemble directly and use eigvals.
    
    # Assemble
    # We need a workspace for assemble_static_hamiltonian!
    ws = SolverWorkspace(N)
    _, H = assemble_static_hamiltonian!(L, h, V, ws)
    
    if H !== nothing
        # Compute eigenvalues only
        # We only need the lowest few. 
        # For large matrices, computing all eigenvalues is slow.
        # But SymTridiagonal eigvals is quite fast.
        vals = eigvals(H)
        
        for level in 1:levels_to_plot
            if level <= length(vals)
                E_errors[level, i] = percent_error(E_analytical[level], vals[level])
            else
                E_errors[level, i] = NaN
            end
        end
    else
        E_errors[:, i] .= NaN
    end
end

# Plotting
println("Plotting results...")

plt = plot(
    xlabel = L"N",
    ylabel = L"\mathrm{Energy\ Error\ (\%)}",
    xscale = :log10,
    yscale = :log10,
    legend = :bottomleft,
    dpi = 300,
    fontfamily="Computer Modern", guidefontsize=12, tickfontsize=10
)

colors = palette(:viridis, levels_to_plot)

for level in 1:levels_to_plot
    n = level - 1
    plot!(plt, N_values, E_errors[level, :], label="n=$n", lw=2, marker=:circle, ms=1.5, color=colors[level])
end

# Add a reference line for O(1/N^2) or O(h^2)
# h ~ 1/N, so error ~ 1/N^2
# Log-log slope should be -2
ref_x = N_values
ref_y = 1e4 .* (1.0 ./ ref_x).^2
plot!(plt, ref_x, ref_y, label=L"O(1/N^2)", ls=:dash, lw=3.5, color=:black)

# Add an annotation near the right-hand side so the reference line is obvious
idx_annot = clamp(round(Int, 0.75 * length(ref_x)), 1, length(ref_x))
x_annot = ref_x[idx_annot]
y_annot = ref_y[idx_annot]
annotate!(plt, x_annot, y_annot * 1.25, text(L"O(1/N^2)", 10, :black, "Computer Modern"))

savefig(plt, "Q2/Plots/Error_vs_StepSize_P$(round(Int, P))_L$(round(Int, L)).png")
println("Done. Plot saved to Q2/Plots/Error_vs_StepSize_P$(round(Int, P))_L$(round(Int, L)).png")