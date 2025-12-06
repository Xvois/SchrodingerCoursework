using Base.Threads

include("../SolutionFunctions.jl")

P = 30.0 # Our dimensionless parameter
h = 0.05 # step size

q = q_of_p(P)
E_analytical_full = analytical_energy_levels(P)
levels_to_plot = min(5, length(E_analytical_full))
E_analytical = E_analytical_full[1:levels_to_plot]

# Create a range of L values to test
L_values = range(0.1 * 1 / q, stop=20 * 1 / q, length=500)
Lq_values = L_values .* q

nthreads_running = nthreads()
if nthreads_running == 1
    println("Running with 1 thread. To use multiple threads set JULIA_NUM_THREADS before starting Julia, e.g.:")
    println("  JULIA_NUM_THREADS=4 julia PlotErrorWithWindowSize.jl")
    println("or use the -t flag:")
    println("  julia -t 4 PlotErrorWithWindowSize.jl")
else
    println("Using $(nthreads_running) threads for computation.")
end

N_candidates = round.(Int, L_values ./ h) .- 1
maxN = max(1, maximum(N_candidates))
workspaces = [SolverWorkspace(maxN) for _ in 1:nthreads_running]

function calculate_single_errors!(dest::Vector{Float64}, L::Float64, V::Function, h::Float64, E_analytical::Vector{Float64}, ws::SolverWorkspace, _buffer::Vector{Float64})
    # Assemble tridiagonal Hamiltonian reusing workspace (no large dense matrices)
    N, H = assemble_static_hamiltonian!(L, h, V, ws)
    if N == 0
        fill!(dest, NaN)
        return dest
    end
    all_vals = eigvals(H)
    nlevels = length(E_analytical)
    @assert length(dest) >= nlevels "Destination array too small for number of levels"
    @assert length(all_vals) >= nlevels "Computed energies fewer than expected levels"
    @inbounds @simd for i in 1:nlevels
        ev = all_vals[i]
        dest[i] = isfinite(ev) ? percent_error(E_analytical[i], ev) : NaN
    end
    return dest
end

function calculate_errors(L_values::AbstractVector{Float64}, h::Float64, V::Function, E_analytical::Vector{Float64}, workspaces::Vector{SolverWorkspace})
    nL = length(L_values)
    nlevels = length(E_analytical)
    E_errors = Array{Float64}(undef, nlevels, nL)
    next_index = Threads.Atomic{Int}(0)

    Threads.@sync for ws in workspaces
        Threads.@spawn begin
            local_ws = ws
            local_errors = Vector{Float64}(undef, nlevels)
            energy_buffer = Vector{Float64}(undef, nlevels)
            while true
                idx = Threads.atomic_add!(next_index, 1) + 1
                if idx > nL
                    break
                end
                L = L_values[idx]
                calculate_single_errors!(local_errors, L, V, h, E_analytical, local_ws, energy_buffer)
                @inbounds @simd for level in 1:nlevels
                    E_errors[level, idx] = local_errors[level]
                end
            end
        end
    end

    println("Completed calculations for $(nL) window sizes (threads: $(nthreads())).")
    return E_errors
end

V(x) = -sech(q*x)^2
E_errors = @time calculate_errors(L_values, h, V, E_analytical, workspaces)

# Plot the error against window size L with shaded regions
using Plots
using LaTeXStrings

highlight_n = 4
highlight_level = highlight_n + 1

# Clean NaNs and compute index of minimal absolute error (boundary/discretisation boundary)
ground_errors = @view E_errors[highlight_level, :]
abs_err = map(x -> isnan(x) ? Inf : abs(x), ground_errors)
min_idx = argmin(abs_err)
Lq_min = Lq_values[min_idx]

# Prepare plot and add background shaded regions. The left region is boundary-dominated
# (Lq < Lq_min) and the right region is discretisation-dominated (Lq > Lq_min).
# Compute logged error values, handling NaNs and zeros for plotting
log_vals = similar(E_errors)
@inbounds for level in axes(E_errors, 1)
    for idx in axes(E_errors, 2)
        val = E_errors[level, idx]
        log_vals[level, idx] = (isfinite(val) && val > 0) ? log10(val) : NaN
    end
end

finite_vals = filter(isfinite, vec(log_vals))
if isempty(finite_vals)
    ymin, ymax = -10.0, 0.0
else
    ymin = minimum(finite_vals)
    ymax = maximum(finite_vals)
    # add a small margin
    rng = ymax - ymin
    if rng == 0.0
        ymin -= 1.0
        ymax += 1.0
    else
        ymin -= 0.05 * rng
        ymax += 0.05 * rng
    end
end

# Prepare plot and draw shaded rectangles (use seriestype=:shape for compatibility)
plt = plot(xlabel=L"Lq", ylabel=L"\log_{10}(\mathrm{Error\ \%})", xlim=(minimum(Lq_values), maximum(Lq_values)), ylim=(ymin, ymax), dpi=500,
           palette = :viridis, fontfamily="Computer Modern", guidefontsize=16, tickfontsize=12, legendfontsize=10, size = (540, 360))

# Left rectangle (boundary-dominated)
xs_left = [minimum(Lq_values), Lq_min, Lq_min, minimum(Lq_values)]
ys_left = [ymin, ymin, ymax, ymax]
plot!(plt, xs_left, ys_left, seriestype = :shape, color = :lightblue, alpha = 0.18, label = "Boundary-dominated")

# Right rectangle (discretisation-dominated)
xs_right = [Lq_min, maximum(Lq_values), maximum(Lq_values), Lq_min]
ys_right = [ymin, ymin, ymax, ymax]
plot!(plt, xs_right, ys_right, seriestype = :shape, color = :lightgreen, alpha = 0.18, label = "Discretisation-dominated")

# Draw the vertical marker at the minimum and then the error line on top
## Highlighting: Use viridis palette

if levels_to_plot >= 1
    # Get colors from viridis palette
    # We want n=0 (dark) to n=4 (bright)
    # levels_to_plot is usually 5 (n=0 to n=4)
    colors = palette(:viridis, levels_to_plot)
    
    for level in 1:levels_to_plot
        n = level - 1
        
        # Use the color from the palette
        c = colors[level]
        
        plot!(plt, Lq_values, log_vals[level, :], lw = 2.0, color = c, label = "n = $(n)")
    end
end

savefig(plt, "./Q2/Plots/Error_vs_WindowSize_P$(Int(P))_h$(h).png")
